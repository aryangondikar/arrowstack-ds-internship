import hashlib
import logging
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: List[str] = [
    "Age", "Attrition", "Department", "JobRole", "JobSatisfaction",
    "MonthlyIncome", "OverTime", "PerformanceRating", "TotalWorkingYears",
    "YearsAtCompany", "YearsSinceLastPromotion"
]

CONSTANT_COLS_TO_DROP = ["EmployeeCount", "Over18", "StandardHours"]


class WorkforceSchemaError(Exception):
    """Raised when HR dataset fails schema or logical consistency checks."""
    pass


def validate_workforce_schema(df: pd.DataFrame) -> None:
    if df.empty:
        raise WorkforceSchemaError("Workforce dataset is empty.")
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise WorkforceSchemaError(f"Missing required columns: {sorted(missing)}")


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


def _anonymize_id(val: object) -> str:
    return hashlib.sha256(str(val).encode("utf-8")).hexdigest()[:12]


def clean_and_engineer_workforce_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans HR data, applies privacy pseudonymization, and validates career timelines."""
    validate_workforce_schema(df)
    df = df.copy()
    initial_rows = len(df)

    # 1. Privacy Guardrail: Pseudonymize EmployeeNumber if present
    if "EmployeeNumber" in df.columns:
        df = df.drop_duplicates(subset=["EmployeeNumber"])
        df["employee_hash"] = df["EmployeeNumber"].apply(_anonymize_id)
        df = df.drop(columns=["EmployeeNumber"])

    # 2. Drop zero-variance columns
    cols_to_drop = [c for c in CONSTANT_COLS_TO_DROP if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # 3. Logical Timeline Edge Cases: Age >= 18, Income > 0, Company Tenure <= Total Career
    df = df[(df["Age"] >= 18) & (df["MonthlyIncome"] > 0)]
    df = df[df["YearsAtCompany"] <= df["TotalWorkingYears"]]

    if len(df) < initial_rows:
        logger.warning(f"Filtered {initial_rows - len(df)} inconsistent or duplicate HR records.")

    # 4. Feature Engineering: Hiring Cohort, Promotion Stagnation & Workload Strain
    df["hiring_tenure_stage"] = pd.cut(
        df["YearsAtCompany"],
        bins=[-1, 1, 3, 7, np.inf],
        labels=["0-1y (New Hire)", "1-3y (Ramp-Up)", "3-7y (Core)", "7y+ (Veteran)"]
    ).astype(str)

    df["promotion_stagnation_ratio"] = (
        df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
    ).round(3)

    df["attrition_binary"] = df["Attrition"].map({"Yes": 1, "No": 0})
    if df["attrition_binary"].isnull().any():
        raise WorkforceSchemaError("Target column 'Attrition' must only contain 'Yes' or 'No'.")

    logger.info(f"Cleaned workforce dataset ready: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df
