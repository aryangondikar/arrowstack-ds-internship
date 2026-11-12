import pandas as pd
import pytest
from src.pipeline import clean_retail_transactions, RetailSchemaError


def test_clean_retail_filters_cancellations_and_fees():
    raw = pd.DataFrame({
        "InvoiceNo": ["536365", "C536366", "536367", "536368"],
        "StockCode": ["85123A", "85123A", "POST", "22423"],
        "Description": ["WHITE HANGING HEART", "WHITE HANGING HEART", "POSTAGE", "REGENCY CAKESTAND"],
        "Quantity": [6, -6, 1, 2],
        "InvoiceDate": ["12/1/2010 8:26", "12/1/2010 8:28", "12/1/2010 8:30", "12/1/2010 8:34"],
        "UnitPrice": [2.55, 2.55, 18.00, 12.75],
        "Country": ["United Kingdom"] * 4
    })
    cleaned = clean_retail_transactions(raw)
    assert len(cleaned) == 2
    assert set(cleaned["StockCode"]) == {"85123A", "22423"}
    assert cleaned["TotalRevenue"].tolist() == [15.30, 25.50]


def test_missing_retail_column_raises_error():
    bad_df = pd.DataFrame({"InvoiceNo": ["536365"], "Quantity": [5]})
    with pytest.raises(RetailSchemaError):
        clean_retail_transactions(bad_df)
