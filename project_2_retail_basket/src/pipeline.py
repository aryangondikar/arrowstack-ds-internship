import logging
from pathlib import Path
from typing import List
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: List[str] = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "Country"
]

NON_PRODUCT_CODES = {"POST", "D", "DOT", "M", "BANK CHARGES", "AMAZONFEE", "CRUK", "C2"}


class RetailSchemaError(Exception):
    """Raised when retail dataset fails required schema or integrity checks."""
    pass


def validate_retail_schema(df: pd.DataFrame) -> None:
    if df.empty:
        raise RetailSchemaError("Retail dataset is empty.")
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise RetailSchemaError(f"Missing required columns: {sorted(missing)}")


def generate_data_dictionary(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    records = []
    for col in df.columns:
        non_null = df[col].dropna()
        sample_val = non_null.iloc[0] if not non_null.empty else "N/A"
        records.append({
            "column_name": col,
            "data_type": str(df[col].dtype),
            "non_null_count": int(df[col].notnull().sum()),
            "null_count": int(df[col].isnull().sum()),
            "null_percentage": round(float(df[col].isnull().mean() * 100), 2),
            "unique_values": int(df[col].nunique()),
            "sample_value": sample_val
        })
    dict_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dict_df.to_csv(output_path, index=False)
    logger.info(f"Data dictionary exported to {output_path}")
    return dict_df


def clean_retail_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans historical retail transactions and engineers temporal/revenue features."""
    validate_retail_schema(df)
    df = df.copy()
    initial_rows = len(df)

    # 1. Remove missing descriptions and normalize whitespace
    df = df.dropna(subset=["InvoiceNo", "Description", "StockCode"])
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["Description"] = df["Description"].astype(str).str.strip().str.upper()

    # 2. Edge Case: Filter cancelled invoices ('C' prefix) and non-product fees
    df = df[~df["InvoiceNo"].str.startswith("C")]
    df = df[~df["StockCode"].isin(NON_PRODUCT_CODES)]

    # 3. Edge Case: Enforce strictly positive quantities and unit prices
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # 4. Parse timestamps & engineer demand features
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    df["TotalRevenue"] = (df["Quantity"] * df["UnitPrice"]).round(2)
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
    df["HourOfDay"] = df["InvoiceDate"].dt.hour

    # Deduplicate exact line-item repeats
    df = df.drop_duplicates()

    logger.info(f"Cleaned retail data: retained {len(df)}/{initial_rows} valid rows.")
    return df
