"""Survival analysis using lifelines (Kaplan-Meier and Cox Proportional Hazards)."""

import os
from pathlib import Path
import joblib
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def run_survival_analysis(
    transactions_path: str = "data/processed/clean_transactions.csv",
    features_path: str = "data/processed/features.csv",
    models_dir: str = "models",
    figures_dir: str = "reports/figures"
):
    """Fit Kaplan-Meier curves and Cox Proportional Hazards model on customer histories."""
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("Loading data for survival analysis...")
    df_tx = pd.read_csv(transactions_path)
    df_tx["InvoiceDate"] = pd.to_datetime(df_tx["InvoiceDate"])
    df_feat = pd.read_csv(features_path)

    # Core event definition from features.csv
    churn_dict = dict(zip(df_feat["CustomerID"].astype(str), df_feat["Churn"]))

    # Use full history to calculate customer duration and event
    data_end = df_tx["InvoiceDate"].max()

    survival_records = []
    for cid, group in df_tx.groupby("CustomerID"):
        cid_str = str(cid)
        if cid_str not in churn_dict:
            continue

        first_purchase = group["InvoiceDate"].min()
        last_purchase = group["InvoiceDate"].max()
        event = churn_dict[cid_str]

        # duration: days from first purchase to last purchase (if churned) or to data end (if not churned)
        if event == 1:
            duration = (last_purchase - first_purchase).days
        else:
            duration = (data_end - first_purchase).days

        # Ensure duration is at least 1 day for numerical stability in Cox
        duration = max(1, duration)

        invoices = group.drop_duplicates(subset=["Invoice"])
        freq = len(invoices)
        monetary = group["Revenue"].sum()
        aov = monetary / freq if freq > 0 else 0.0
        uniq_prods = group["StockCode"].nunique()

        tenure = max(1, (data_end - first_purchase).days)
        if freq > 1:
            diffs = invoices["InvoiceDate"].sort_values().diff().dropna().dt.total_seconds() / 86400.0
            avg_days_between = float(diffs.mean())
        else:
            avg_days_between = float(tenure)

        country = group["Country"].mode()[0] if not group["Country"].empty else "United Kingdom"
        freq_tier = "High Frequency (>3)" if freq > 3 else "Low Frequency (<=3)"
        country_group = "United Kingdom" if country == "United Kingdom" else "International"

        survival_records.append({
            "CustomerID": cid_str,
            "duration": duration,
            "event": event,
            "Frequency": freq,
            "AvgOrderValue": aov,
            "UniqueProducts": uniq_prods,
            "AvgDaysBetweenOrders": avg_days_between,
            "FrequencyTier": freq_tier,
            "CountryGroup": country_group
        })

    surv_df = pd.DataFrame(survival_records)
    print(f"Survival dataset prepared: {surv_df.shape}")

    # 1. Kaplan-Meier Curves
    kmf = KaplanMeierFitter()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall KM
    kmf.fit(surv_df["duration"], event_observed=surv_df["event"], label="Overall Cohort")
    kmf.plot_survival_function(ax=axes[0], color="#1f77b4", linewidth=2.5)
    axes[0].set_title("Overall Kaplan-Meier Customer Survival Curve", fontweight="bold")
    axes[0].set_xlabel("Timeline (Days)", fontweight="bold")
    axes[0].set_ylabel("Customer Survival Probability", fontweight="bold")
    axes[0].set_ylim(0, 1.05)

    # KM by Frequency Tier
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    mask_high = surv_df["FrequencyTier"] == "High Frequency (>3)"
    kmf_high.fit(surv_df.loc[mask_high, "duration"], surv_df.loc[mask_high, "event"], label="High Frequency (>3)")
    kmf_low.fit(surv_df.loc[~mask_high, "duration"], surv_df.loc[~mask_high, "event"], label="Low Frequency (<=3)")

    kmf_high.plot_survival_function(ax=axes[1], color="#2ca02c", linewidth=2)
    kmf_low.plot_survival_function(ax=axes[1], color="#e74c3c", linewidth=2)
    axes[1].set_title("Survival Curves by Purchase Frequency Tier", fontweight="bold")
    axes[1].set_xlabel("Timeline (Days)", fontweight="bold")
    axes[1].set_ylabel("Survival Probability", fontweight="bold")
    axes[1].set_ylim(0, 1.05)

    # Log-rank test
    lr_res = logrank_test(
        surv_df.loc[mask_high, "duration"], surv_df.loc[~mask_high, "duration"],
        surv_df.loc[mask_high, "event"], surv_df.loc[~mask_high, "event"]
    )
    print(f"Log-rank test p-value between Frequency Tiers: {lr_res.p_value:.4e}")
    axes[1].text(0.05, 0.15, f"Log-Rank p-value: {lr_res.p_value:.2e}", transform=axes[1].transAxes,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=9)

    plt.tight_layout()
    plots_path = Path(figures_dir) / "survival_curves.png"
    plt.savefig(plots_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Kaplan-Meier plot to {plots_path}")

    # 2. Cox Proportional Hazards Model
    cox_cols = ["duration", "event", "Frequency", "AvgOrderValue", "UniqueProducts", "AvgDaysBetweenOrders"]
    cph_df = surv_df[cox_cols].copy()

    # Log transform heavy-tailed covariates for proportional hazards fit stability
    cph_df["Frequency"] = np.log1p(cph_df["Frequency"])
    cph_df["AvgOrderValue"] = np.log1p(np.maximum(0, cph_df["AvgOrderValue"]))
    cph_df["UniqueProducts"] = np.log1p(cph_df["UniqueProducts"])
    cph_df["AvgDaysBetweenOrders"] = np.log1p(cph_df["AvgDaysBetweenOrders"])

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(cph_df, duration_col="duration", event_col="event")

    print("\n=== Cox Proportional Hazards Summary ===")
    hazard_ratios = np.exp(cph.params_)
    print("Hazard Ratios (exp(coef)):")
    for k, v in hazard_ratios.items():
        print(f"  {k}: {v:.4f}")
    c_index = float(cph.concordance_index_)
    print(f"Cox Concordance Index: {c_index:.4f}")

    # Save model and survival dataset for downstream customer expected survival lookups
    model_path = Path(models_dir) / "cox.joblib"
    joblib.dump({"cox_model": cph, "surv_df": surv_df, "concordance_index": c_index}, model_path)
    print(f"Saved CoxPHFitter model to {model_path}")

    return c_index, surv_df


if __name__ == "__main__":
    curr_dir = Path(__file__).resolve().parent
    base_dir = curr_dir.parent

    tx_path = base_dir / "data" / "processed" / "clean_transactions.csv"
    feat_path = base_dir / "data" / "processed" / "features.csv"
    mod_dir = base_dir / "models"
    fig_dir = base_dir / "reports" / "figures"

    run_survival_analysis(str(tx_path), str(feat_path), str(mod_dir), str(fig_dir))
