"""Data cleaning and preparation pipeline for E-Commerce Churn & LTV predictor."""

import os
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
try:
    from make_synthetic import generate_synthetic_data
except ImportError:
    from src.make_synthetic import generate_synthetic_data


def load_and_clean_data(raw_data_path: str = "data/raw/online_retail_II.xlsx",
                        output_path: str = "data/processed/clean_transactions.csv") -> pd.DataFrame:
    """Load both sheets from Excel file, clean records, and save to CSV.

    Args:
        raw_data_path: Path to the raw UCI Excel workbook.
        output_path: Destination path for cleaned transaction CSV.

    Returns:
        pd.DataFrame: Cleaned transactions dataframe.
    """
    raw_file = Path(raw_data_path)
    if not raw_file.exists():
        print(f"File {raw_data_path} not found. Generating realistic synthetic data...")
        generate_synthetic_data(output_path=raw_data_path)

    print(f"Loading {raw_data_path}...")
    xl = pd.ExcelFile(raw_file)
    sheets = xl.sheet_names
    print(f"Found sheets: {sheets}")

    dfs = []
    for sheet in sheets:
        print(f"Reading sheet: {sheet}...")
        df_sheet = xl.parse(sheet)
        dfs.append(df_sheet)

    df_raw = pd.concat(dfs, ignore_index=True)
    initial_shape = df_raw.shape
    print(f"Raw data shape: {initial_shape}")

    # Standardize column names if needed
    col_mapping = {"Customer ID": "CustomerID", "Customer ID ": "CustomerID"}
    df_raw = df_raw.rename(columns=col_mapping)

    # 1. Drop rows with missing Customer ID
    df = df_raw.dropna(subset=["CustomerID"]).copy()
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)

    # 2. Remove cancelled invoices (Invoice starting with 'C') and rows with Quantity <= 0 or Price <= 0
    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    df = df[~df["Invoice"].str.startswith("C", na=False)]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

    # Ensure InvoiceDate is datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # 3. Add Revenue = Quantity * Price
    df["Revenue"] = df["Quantity"] * df["Price"]

    # Sort chronological
    df = df.sort_values("InvoiceDate").reset_index(drop=True)

    # 4. Save to data/processed/clean_transactions.csv
    os.makedirs(Path(output_path).parent, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path} with shape {df.shape} (dropped {initial_shape[0] - df.shape[0]} rows)")

    return df


if __name__ == "__main__":
    # Resolve paths relative to churn-ltv-predictor folder if run from src or root
    curr_dir = Path(__file__).resolve().parent
    base_dir = curr_dir.parent

    raw_path = base_dir / "data" / "raw" / "online_retail_II.xlsx"
    out_path = base_dir / "data" / "processed" / "clean_transactions.csv"

    load_and_clean_data(str(raw_path), str(out_path))
