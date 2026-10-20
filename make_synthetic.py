"""Synthetic transaction data generator for E-Commerce Customer Churn & LTV Predictor."""

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def generate_synthetic_data(output_path: str = "data/raw/online_retail_II.xlsx", n_customers: int = 1000, n_transactions: int = 25000) -> None:
    """Generate realistic fake transactions matching UCI Online Retail II schema.

    Args:
        output_path: Path where Excel file with sheets will be saved.
        n_customers: Number of unique customer IDs to generate.
        n_transactions: Number of transactions to synthesize.
    """
    np.random.seed(42)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    customer_ids = np.random.choice(range(10000, 10000 + n_customers), size=n_transactions)
    countries = ["United Kingdom"] * int(0.9 * n_transactions) + list(
        np.random.choice(["Germany", "France", "EIRE", "Spain", "Netherlands"], size=n_transactions - int(0.9 * n_transactions))
    )
    np.random.shuffle(countries)

    start_date = datetime(2009, 12, 1)
    date_offsets = [timedelta(days=int(np.random.exponential(scale=180)) % 730,
                              hours=int(np.random.randint(8, 20)),
                              minutes=int(np.random.randint(0, 60)))
                    for _ in range(n_transactions)]
    invoice_dates = [start_date + offset for offset in date_offsets]

    stock_codes = [f"STK_{np.random.randint(100, 500)}" for _ in range(n_transactions)]
    descriptions = [f"Product Description {sc}" for sc in stock_codes]
    quantities = np.random.geometric(p=0.2, size=n_transactions)
    prices = np.round(np.random.exponential(scale=15.0, size=n_transactions) + 0.5, 2)

    invoices = [f"{500000 + (hash(str(cid) + str(dt.date())) % 50000)}" for cid, dt in zip(customer_ids, invoice_dates)]

    # Add ~2% cancellations
    cancel_mask = np.random.rand(n_transactions) < 0.02
    invoices = [f"C{inv}" if is_c else inv for inv, is_c in zip(invoices, cancel_mask)]
    quantities = [-q if is_c else q for q, is_c in zip(quantities, cancel_mask)]

    # Split roughly across two years
    midpoint = datetime(2010, 12, 1)
    df_all = pd.DataFrame({
        "Invoice": invoices,
        "StockCode": stock_codes,
        "Description": descriptions,
        "Quantity": quantities,
        "InvoiceDate": invoice_dates,
        "Price": prices,
        "Customer ID": customer_ids,
        "Country": countries
    })

    df_year1 = df_all[df_all["InvoiceDate"] < midpoint]
    df_year2 = df_all[df_all["InvoiceDate"] >= midpoint]

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_year1.to_excel(writer, sheet_name="Year 2009-2010", index=False)
        df_year2.to_excel(writer, sheet_name="Year 2010-2011", index=False)

    print(f"Synthetic dataset saved to {output_path}")


if __name__ == "__main__":
    generate_synthetic_data()
