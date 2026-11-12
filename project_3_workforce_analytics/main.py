import logging
from pathlib import Path
import pandas as pd
from src.fetch_data import download_workforce_dataset
from src.pipeline import generate_data_dictionary, clean_and_engineer_workforce_data
from src.analysis import generate_workforce_charts, run_workforce_study

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_project_3() -> None:
    base_dir = Path(__file__).resolve().parent
    raw_path = base_dir / "data/raw/hr_attrition.csv"
    processed_path = base_dir / "data/processed/workforce_cleaned.csv"
    dict_path = base_dir / "reports/data_dictionary.csv"
    report_path = base_dir / "reports/workforce_analytics_report.json"
    figures_dir = base_dir / "reports/figures"

    logger.info("=== Starting Project 3: Workforce Analytics Study ===")
    download_workforce_dataset(raw_path)

    raw_df = pd.read_csv(raw_path)
    generate_data_dictionary(raw_df, dict_path)

    clean_df = clean_and_engineer_workforce_data(raw_df)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(processed_path, index=False)

    generate_workforce_charts(clean_df, figures_dir)
    report = run_workforce_study(clean_df, report_path)

    logger.info(f"Overall Attrition Rate: {report['overall_attrition_rate_pct']}% | 5-Fold ROC-AUC: {report['multivariate_5fold_roc_auc']}")
    logger.info("=== Project 3 Pipeline Completed Successfully ===")


if __name__ == "__main__":
    run_project_3()
