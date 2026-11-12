import logging
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: List[str] = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "InternetService", "Contract",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
]


class SchemaValidationError(Exception):
    """Raised when input dataset violates expected schema or business rules."""
    pass


def validate_raw_schema(df: pd.DataFrame) -> None:
    """Ensures required columns exist and dataset is non-empty."""
    if df.empty:
        raise SchemaValidationError("Input dataset is empty.")
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise SchemaValidationError(f"Missing required columns: {sorted(missing)}")


def generate_data_dictionary(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Exports a structured data dictionary with null rates and sample values."""
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


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans realistic anomalies and engineers business cohort features."""
    validate_raw_schema(df)
    df = df.copy()

    # 1. Deduplicate primary key
    initial_rows = len(df)
    df = df.dropna(subset=["customerID"]).drop_duplicates(subset=["customerID"])
    if len(df) < initial_rows:
        logger.warning(f"Removed {initial_rows - len(df)} duplicate/null customerID rows.")

    # 2. Edge Case: TotalCharges contains whitespace strings for tenure == 0 customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    zero_tenure_mask = (df["tenure"] == 0) & (df["TotalCharges"].isnull())
    df.loc[zero_tenure_mask, "TotalCharges"] = 0.0
    if df["TotalCharges"].isnull().any():
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # 3. Edge Case: Filter impossible negative values
    df = df[(df["tenure"] >= 0) & (df["MonthlyCharges"] >= 0)]

    # 4. Feature Engineering for Business Segmentation
    df["tenure_cohort"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, np.inf],
        labels=["0-6m (Onboarding)", "6-12m (Early)", "12-24m (Mid)", "24-48m (Established)", "48m+ (Loyal)"]
    ).astype(str)

    service_cols = [
        c for c in ["PhoneService", "MultipleLines", "OnlineSecurity",
                    "OnlineBackup", "DeviceProtection", "TechSupport",
                    "StreamingTV", "StreamingMovies"] if c in df.columns
    ]
    df["active_services_count"] = df[service_cols].apply(lambda row: sum(row == "Yes"), axis=1)
    df["churn_binary"] = df["Churn"].map({"Yes": 1, "No": 0})

    if df["churn_binary"].isnull().any():
        raise SchemaValidationError("Target column 'Churn' contains unexpected values outside {'Yes', 'No'}.")

    logger.info(f"Cleaned dataset ready: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df
