import pandas as pd
import pytest
from src.pipeline import clean_and_engineer_features, SchemaValidationError


def _make_sample_df(**overrides) -> pd.DataFrame:
    base = {
        "customerID": ["001", "002", "003"],
        "gender": ["Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes"],
        "tenure": [0, 15, -4],
        "PhoneService": ["Yes", "Yes", "No"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        "MonthlyCharges": [29.85, 85.50, 20.00],
        "TotalCharges": [" ", "1282.50", "80.00"],
        "Churn": ["No", "Yes", "No"]
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_clean_pipeline_handles_edge_cases():
    df = _make_sample_df()
    cleaned = clean_and_engineer_features(df)

    # Negative tenure row ('003') should be dropped -> 2 rows left
    assert len(cleaned) == 2
    # Zero-tenure blank TotalCharges ('001') should be imputed to 0.0
    assert cleaned.loc[cleaned["customerID"] == "001", "TotalCharges"].iloc[0] == 0.0
    assert cleaned["churn_binary"].tolist() == [0, 1]


def test_missing_column_raises_schema_error():
    df = _make_sample_df().drop(columns=["Contract"])
    with pytest.raises(SchemaValidationError):
        clean_and_engineer_features(df)
