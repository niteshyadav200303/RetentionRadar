"""Feature engineering pipeline for customer churn and LTV prediction."""

import os
from pathlib import Path
import numpy as np
import pandas as pd


def generate_customer_features(
    input_path: str = "data/processed/clean_transactions.csv",
    output_path: str = "data/processed/features.csv",
    cutoff_days: int = 90
) -> pd.DataFrame:
    """Engineer customer-level features before cutoff date and compute non-leaking churn label.

    Args:
        input_path: Path to cleaned transaction data.
        output_path: Destination path for engineered features CSV.
        cutoff_days: Number of days before the max transaction date defining the cutoff.

    Returns:
        pd.DataFrame: Engineered feature dataset ready for modeling.
    """
    print(f"Loading cleaned transactions from {input_path}...")
    df = pd.read_csv(input_path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Core definition: Cutoff date = last InvoiceDate minus 90 days
    max_date = df["InvoiceDate"].max()
    cutoff_date = max_date - pd.Timedelta(days=cutoff_days)
    print(f"Max transaction date: {max_date}, Cutoff date: {cutoff_date}")

    # Split transactions strictly before and after cutoff
    df_obs = df[df["InvoiceDate"] < cutoff_date].copy()
    df_perf = df[df["InvoiceDate"] >= cutoff_date].copy()

    # Churn label: 1 if customer made no purchase in 90 days after cutoff, else 0
    obs_customers = set(df_obs["CustomerID"].unique())
    perf_customers = set(df_perf["CustomerID"].unique())
    churn_map = {cid: (1 if cid not in perf_customers else 0) for cid in obs_customers}

    print(f"Observed customers: {len(obs_customers)}, Churned: {sum(churn_map.values())} ({sum(churn_map.values())/len(obs_customers):.2%})")

    # Group by customer in observation window
    records = []
    top_5_countries = ["United Kingdom", "Germany", "France", "EIRE", "Spain"]

    for cid, group in df_obs.groupby("CustomerID"):
        group = group.sort_values("InvoiceDate")
        first_date = group["InvoiceDate"].min()
        last_date = group["InvoiceDate"].max()

        tenure = (cutoff_date - first_date).days
        recency = (cutoff_date - last_date).days

        # Distinct invoices
        invoices = group.drop_duplicates(subset=["Invoice"])
        frequency = len(invoices)
        monetary = group["Revenue"].sum()
        avg_order_value = monetary / frequency if frequency > 0 else 0.0

        # Avg days between orders
        if frequency > 1:
            order_dates = invoices["InvoiceDate"].sort_values()
            diffs = order_dates.diff().dropna().dt.total_seconds() / 86400.0
            avg_days_between = float(diffs.mean())
        else:
            avg_days_between = float(tenure)

        unique_products = group["StockCode"].nunique()

        # Purchases in last 30, 60, 90 days before cutoff
        d30 = cutoff_date - pd.Timedelta(days=30)
        d60 = cutoff_date - pd.Timedelta(days=60)
        d90 = cutoff_date - pd.Timedelta(days=90)

        purchases_30 = group[group["InvoiceDate"] >= d30]["Invoice"].nunique()
        purchases_60 = group[group["InvoiceDate"] >= d60]["Invoice"].nunique()
        purchases_90 = group[group["InvoiceDate"] >= d90]["Invoice"].nunique()

        # Primary country
        country = group["Country"].mode()[0] if not group["Country"].empty else "United Kingdom"
        country_clean = country if country in top_5_countries else "Other"

        records.append({
            "CustomerID": str(cid),
            "Recency": recency,
            "Frequency": frequency,
            "Monetary": round(monetary, 2),
            "AvgOrderValue": round(avg_order_value, 2),
            "Tenure": tenure,
            "AvgDaysBetweenOrders": round(avg_days_between, 2),
            "UniqueProducts": unique_products,
            "ReturnRate": 0.0,
            "PurchasesLast30Days": purchases_30,
            "PurchasesLast60Days": purchases_60,
            "PurchasesLast90Days": purchases_90,
            "Country": country_clean,
            "Churn": churn_map[cid]
        })

    feat_df = pd.DataFrame(records)

    # One-hot encode Country (top 5 + Other)
    country_dummies = pd.get_dummies(feat_df["Country"], prefix="Country", dtype=int)
    for c in ["Country_United Kingdom", "Country_Germany", "Country_France", "Country_EIRE", "Country_Spain", "Country_Other"]:
        if c not in country_dummies.columns:
            country_dummies[c] = 0

    feat_df = pd.concat([feat_df, country_dummies], axis=1)

    os.makedirs(Path(output_path).parent, exist_ok=True)
    feat_df.to_csv(output_path, index=False)
    print(f"Features successfully saved to {output_path} with shape {feat_df.shape}")
    return feat_df


if __name__ == "__main__":
    curr_dir = Path(__file__).resolve().parent
    base_dir = curr_dir.parent

    in_path = base_dir / "data" / "processed" / "clean_transactions.csv"
    out_path = base_dir / "data" / "processed" / "features.csv"

    generate_customer_features(str(in_path), str(out_path))
