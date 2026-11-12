import logging
from pathlib import Path
import pandas as pd
from src.fetch_data import download_retail_dataset
from src.pipeline import generate_data_dictionary, clean_retail_transactions
from src.analysis import generate_demand_charts, run_basket_association_analysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_project_2() -> None:
    base_dir = Path(__file__).resolve().parent
    raw_path = base_dir / "data/raw/online_retail.csv"
    processed_path = base_dir / "data/processed/retail_cleaned.csv"
    dict_path = base_dir / "reports/data_dictionary.csv"
    rules_path = base_dir / "reports/association_rules.csv"
    metrics_path = base_dir / "reports/basket_metrics.json"
    figures_dir = base_dir / "reports/figures"

    logger.info("=== Starting Project 2: Retail Demand & Basket Analysis ===")
    download_retail_dataset(raw_path)

    raw_df = pd.read_csv(raw_path, encoding="ISO-8859-1")
    generate_data_dictionary(raw_df, dict_path)

    clean_df = clean_retail_transactions(raw_df)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(processed_path, index=False)

    generate_demand_charts(clean_df, figures_dir)
    metrics = run_basket_association_analysis(clean_df, rules_path, metrics_path, figures_dir)

    logger.info(f"Qualified Association Rules Found: {metrics['qualified_rules_count']} (Max Lift: {metrics['max_lift']})")
    logger.info("=== Project 2 Pipeline Completed Successfully ===")


if __name__ == "__main__":
    run_project_2()
