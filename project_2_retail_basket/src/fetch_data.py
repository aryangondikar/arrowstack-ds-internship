import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_URL = "https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/master/data/retail-data/all/online-retail-dataset.csv"


def download_retail_dataset(output_path: Path) -> Path:
    """Downloads public historical retail transaction dataset to raw storage."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        logger.info(f"Dataset already exists at {output_path}")
        return output_path

    logger.info(f"Fetching retail dataset from {DATA_URL}...")
    df = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
    df.to_csv(output_path, index=False)
    logger.info(f"Saved raw dataset ({df.shape[0]} rows, {df.shape[1]} cols) to {output_path}")
    return output_path


if __name__ == "__main__":
    download_retail_dataset(Path("data/raw/online_retail.csv"))
