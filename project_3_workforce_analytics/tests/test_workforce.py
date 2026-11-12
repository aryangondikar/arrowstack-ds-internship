import pandas as pd
import pytest
from src.pipeline import clean_and_engineer_workforce_data, WorkforceSchemaError
from src.analysis import test_categorical_driver


def _make_hr_df() -> pd.DataFrame:
    return pd.DataFrame({
        "EmployeeNumber": [101, 102, 103],
        "Age": [34, 28, 41],
        "Attrition": ["No", "Yes", "No"],
        "Department": ["Sales", "R&D", "HR"],
        "JobRole": ["Sales Executive", "Research Scientist", "HR Manager"],
        "JobSatisfaction": [4, 2, 3],
        "MonthlyIncome": [6500, 3200, 8100],
        "OverTime": ["No", "Yes", "No"],
        "PerformanceRating": [3, 3, 4],
        "TotalWorkingYears": [10, 4, 2],  # Row 3 has YearsAtCompany (5) > TotalWorkingYears (2) -> invalid
        "YearsAtCompany": [6, 1, 5],
        "YearsSinceLastPromotion": [1, 0, 2],
        "EmployeeCount": [1, 1, 1]
    })


def test_clean_workforce_anonymizes_and_filters_invalid_timeline():
    df = _make_hr_df()
    cleaned = clean_and_engineer_workforce_data(df)

    # Row 3 removed due to impossible tenure > total experience
    assert len(cleaned) == 2
    # Raw EmployeeNumber stripped and replaced with hashed ID
    assert "EmployeeNumber" not in cleaned.columns
    assert "employee_hash" in cleaned.columns
    # Zero-variance EmployeeCount dropped
    assert "EmployeeCount" not in cleaned.columns


def test_governance_warning_on_small_subgroup():
    df = pd.DataFrame({
        "Department": ["Sales"] * 20 + ["Legal"] * 3,
        "attrition_binary": [0] * 16 + [1] * 4 + [0] * 2 + [1] * 1
    })
    res = test_categorical_driver(df, "Department", min_subgroup_size=10)
    assert res["governance_warning"] is not None


def test_missing_hr_column_raises_error():
    df = _make_hr_df().drop(columns=["OverTime"])
    with pytest.raises(WorkforceSchemaError):
        clean_and_engineer_workforce_data(df)
