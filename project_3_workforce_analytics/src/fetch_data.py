import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_URL = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"


def download_workforce_dataset(output_path: Path) -> Path:
    """Downloads public HR workforce attrition dataset to raw storage."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        logger.info(f"Dataset already exists at {output_path}")
        return output_path

    logger.info(f"Fetching workforce dataset from {DATA_URL}...")
    df = pd.read_csv(DATA_URL)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved raw dataset ({df.shape[0]} rows, {df.shape[1]} cols) to {output_path}")
    return output_path


if __name__ == "__main__":
    download_workforce_dataset(Path("data/raw/hr_attrition.csv"))
