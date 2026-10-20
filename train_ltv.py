"""Train Beta-Geometric / Negative Binomial (BG/NBD) and Gamma-Gamma LTV models."""

import os
from pathlib import Path
import joblib
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import calibration_and_holdout_data, summary_data_from_transaction_data
import numpy as np
import pandas as pd


def train_ltv_pipeline(
    transactions_path: str = "data/processed/clean_transactions.csv",
    features_path: str = "data/processed/features.csv",
    churn_model_path: str = "models/churn_xgb.joblib",
    cox_model_path: str = "models/cox.joblib",
    models_dir: str = "models",
    output_scores_path: str = "data/processed/customer_scores.csv"
):
    """Train BG/NBD and Gamma-Gamma models, compute holdout validation, and assemble customer_scores.csv."""
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(Path(output_scores_path).parent, exist_ok=True)

    print("Loading cleaned transactions...")
    df_tx = pd.read_csv(transactions_path)
    df_tx["InvoiceDate"] = pd.to_datetime(df_tx["InvoiceDate"])

    # 1. Validation with calibration vs holdout split
    max_date = df_tx["InvoiceDate"].max()
    cal_end = max_date - pd.Timedelta(days=90)
    print(f"Holdout validation split: calibration ending at {cal_end}...")

    cal_hold = calibration_and_holdout_data(
        df_tx, "CustomerID", "InvoiceDate",
        calibration_period_end=cal_end,
        monetary_value_col="Revenue",
        freq="W"
    )

    bgf_val = BetaGeoFitter(penalizer_coef=0.001)
    bgf_val.fit(cal_hold["frequency_cal"], cal_hold["recency_cal"], cal_hold["T_cal"])
    holdout_weeks = int(np.ceil((max_date - cal_end).days / 7))
    predicted_holdout = bgf_val.predict(holdout_weeks, cal_hold["frequency_cal"], cal_hold["recency_cal"], cal_hold["T_cal"])
    holdout_corr = float(predicted_holdout.corr(cal_hold["frequency_holdout"]))
    print(f"Holdout Correlation (Predicted vs Actual holdout purchases): {holdout_corr:.4f}")

    # 2. Build full RFM summary in weeks
    print("Building full weekly RFM summary...")
    summary = summary_data_from_transaction_data(
        df_tx, "CustomerID", "InvoiceDate",
        monetary_value_col="Revenue",
        freq="W"
    )
    summary.index = summary.index.astype(str)

    # 3. Fit BetaGeoFitter on full dataset
    print("Fitting BetaGeoFitter on full history...")
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])

    # 4. Fit GammaGammaFitter on returning customers with frequency > 0
    print("Fitting GammaGammaFitter on repeat customers...")
    returning_customers = summary[(summary["frequency"] > 0) & (summary["monetary_value"] > 0)]
    ggf = GammaGammaFitter(penalizer_coef=0.001)
    ggf.fit(returning_customers["frequency"], returning_customers["monetary_value"])

    # Save models using lifetimes native save_model method
    bgf_path = str(Path(models_dir) / "bgf.pkl")
    ggf_path = str(Path(models_dir) / "ggf.pkl")
    bgf.save_model(bgf_path, save_data=False, save_generate_data_method=False)
    ggf.save_model(ggf_path, save_data=False, save_generate_data_method=False)
    print(f"Saved BG/NBD and Gamma-Gamma models to {models_dir}")

    # 5. Predict metrics across all customers
    summary["ProbAlive"] = bgf.conditional_probability_alive(
        summary["frequency"], summary["recency"], summary["T"]
    )

    # Impute zero/negative monetary values with cohort median for stable LTV calculation
    median_monetary = returning_customers["monetary_value"].median()
    monetary_imputed = summary["monetary_value"].apply(lambda v: v if v > 0 else median_monetary)

    summary["LTV_6m"] = ggf.customer_lifetime_value(
        bgf,
        summary["frequency"],
        summary["recency"],
        summary["T"],
        monetary_imputed,
        time=6,
        discount_rate=0.01,
        freq="W"
    )

    summary["LTV_12m"] = ggf.customer_lifetime_value(
        bgf,
        summary["frequency"],
        summary["recency"],
        summary["T"],
        monetary_imputed,
        time=12,
        discount_rate=0.01,
        freq="W"
    )

    summary["LTV_6m"] = summary["LTV_6m"].fillna(0.0).clip(lower=0.0).round(2)
    summary["LTV_12m"] = summary["LTV_12m"].fillna(0.0).clip(lower=0.0).round(2)

    # 6. Load churn predictions
    print(f"Loading churn model from {churn_model_path}...")
    churn_artifact = joblib.load(churn_model_path)
    churn_model = churn_artifact["model"]
    feature_names = churn_artifact["feature_names"]
    threshold = churn_artifact["threshold"]

    df_feat = pd.read_csv(features_path)
    X = df_feat[feature_names]
    churn_probs = churn_model.predict_proba(X)[:, 1]
    churn_flags = (churn_probs >= threshold).astype(int)

    churn_score_map = dict(zip(df_feat["CustomerID"].astype(str), churn_probs))
    churn_flag_map = dict(zip(df_feat["CustomerID"].astype(str), churn_flags))

    # 7. Load Cox model to estimate expected survival days
    print(f"Loading Cox model from {cox_model_path}...")
    cox_artifact = joblib.load(cox_model_path)
    surv_df = cox_artifact["surv_df"]
    # Mean duration for active/retained vs churned
    surv_duration_map = dict(zip(surv_df["CustomerID"].astype(str), surv_df["duration"]))

    scores_list = []
    customer_ids = summary.index.astype(str)

    # RFM Segment computation
    r_labels = [4, 3, 2, 1]
    f_labels = [1, 2, 3, 4]
    m_labels = [1, 2, 3, 4]

    r_quart = pd.qcut(summary["recency"].rank(method="first"), q=4, labels=r_labels)
    f_quart = pd.qcut(summary["frequency"].rank(method="first"), q=4, labels=f_labels)
    m_quart = pd.qcut(monetary_imputed.rank(method="first"), q=4, labels=m_labels)

    def assign_segment(r, f, m):
        rfm_score = int(r) * 100 + int(f) * 10 + int(m)
        if r >= 3 and f >= 3:
            return "Champions"
        elif f >= 3:
            return "Loyal Customers"
        elif r >= 3 and f <= 2:
            return "Potential Loyalists"
        elif r <= 2 and f >= 2:
            return "At Risk"
        else:
            return "Hibernating"

    segments = [assign_segment(r, f, m) for r, f, m in zip(r_quart, f_quart, m_quart)]
    summary["RFM_segment"] = segments

    # Value tier by LTV_12m quantiles (Top 20% High, middle 30% Mid, bottom 50% Low)
    value_tiers = pd.qcut(summary["LTV_12m"], q=[0.0, 0.50, 0.80, 1.0], labels=["Low", "Mid", "High"])
    summary["Value_tier"] = value_tiers

    for cid in customer_ids:
        c_churn_prob = float(churn_score_map.get(cid, 0.5))
        c_churn_flag = int(churn_flag_map.get(cid, 1 if c_churn_prob >= threshold else 0))
        c_prob_alive = float(summary.loc[cid, "ProbAlive"])
        c_ltv_6 = float(summary.loc[cid, "LTV_6m"])
        c_ltv_12 = float(summary.loc[cid, "LTV_12m"])
        c_rfm = str(summary.loc[cid, "RFM_segment"])
        c_tier = str(summary.loc[cid, "Value_tier"])

        # Expected remaining survival days estimation based on ProbAlive and historical tenure
        hist_days = surv_duration_map.get(cid, 180)
        expected_survival_days = int(round(hist_days * (c_prob_alive if c_churn_flag == 0 else (1.0 - c_churn_prob))))
        expected_survival_days = max(14, expected_survival_days)

        scores_list.append({
            "CustomerID": cid,
            "ChurnProb": round(c_churn_prob, 4),
            "ChurnFlag": c_churn_flag,
            "ProbAlive": round(c_prob_alive, 4),
            "LTV_6m": c_ltv_6,
            "LTV_12m": c_ltv_12,
            "ExpectedSurvivalDays": expected_survival_days,
            "RFM segment": c_rfm,
            "Value tier": c_tier
        })

    scores_df = pd.DataFrame(scores_list)
    scores_df.to_csv(output_scores_path, index=False)
    print(f"Customer scores table saved to {output_scores_path} with {len(scores_df)} rows")

    # High-risk high-value count
    high_risk_high_value = scores_df[(scores_df["ChurnFlag"] == 1) & (scores_df["Value tier"] == "High")]
    print(f"Number of High-Risk, High-Value customers: {len(high_risk_high_value)}")

    return holdout_corr, len(high_risk_high_value)


if __name__ == "__main__":
    curr_dir = Path(__file__).resolve().parent
    base_dir = curr_dir.parent

    tx_path = base_dir / "data" / "processed" / "clean_transactions.csv"
    feat_path = base_dir / "data" / "processed" / "features.csv"
    churn_mod = base_dir / "models" / "churn_xgb.joblib"
    cox_mod = base_dir / "models" / "cox.joblib"
    mod_dir = base_dir / "models"
    out_scores = base_dir / "data" / "processed" / "customer_scores.csv"

    train_ltv_pipeline(str(tx_path), str(feat_path), str(churn_mod), str(cox_mod), str(mod_dir), str(out_scores))
