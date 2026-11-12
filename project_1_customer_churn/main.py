import logging
from pathlib import Path
from src.fetch_data import download_dataset
from src.pipeline import generate_data_dictionary, clean_and_engineer_features
from src.analysis import generate_eda_charts, run_driver_modeling
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_project_1() -> None:
    base_dir = Path(__file__).resolve().parent
    raw_path = base_dir / "data/raw/telco_churn.csv"
    processed_path = base_dir / "data/processed/churn_cleaned.csv"
    dict_path = base_dir / "reports/data_dictionary.csv"
    figures_dir = base_dir / "reports/figures"
    metrics_path = base_dir / "reports/model_metrics.json"

    logger.info("=== Starting Project 1: Customer Churn Investigation ===")
    download_dataset(raw_path)

    raw_df = pd.read_csv(raw_path)
    generate_data_dictionary(raw_df, dict_path)

    clean_df = clean_and_engineer_features(raw_df)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(processed_path, index=False)

    generate_eda_charts(clean_df, figures_dir)
    metrics = run_driver_modeling(clean_df, metrics_path, figures_dir)

    logger.info(f"Logistic Regression Test ROC-AUC: {metrics['LogisticRegression']['test_roc_auc']}")
    logger.info(f"Random Forest Test ROC-AUC: {metrics['RandomForest']['test_roc_auc']}")
    logger.info("=== Project 1 Pipeline Completed Successfully ===")


if __name__ == "__main__":
    run_project_1()
