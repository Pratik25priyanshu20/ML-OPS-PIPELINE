#src/data/ingestion.py
"""
Data Ingestion Module — Bank Marketing Dataset (UCI)
Loads raw CSV, applies basic sanity checks, and hands off to validation.
"""

from pathlib import Path
from typing import Dict
import pandas as pd
import yaml
from loguru import logger


class DataIngestion:
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        self.config = self._load_config(config_path)
        self.raw_path = Path(self.config["raw_data"]["path"])
        logger.info(f"DataIngestion initialized — reading from {self.raw_path}")

    def _load_config(self, path: str) -> Dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def load_raw(self) -> pd.DataFrame:
        """
        Load raw bank marketing dataset.
        Returns:
            df (pd.DataFrame)
        """
        if not self.raw_path.exists():
            raise FileNotFoundError(f"❌ Raw data missing: {self.raw_path}")

        logger.info("Loading dataset with semicolon delimiter...")
        df = pd.read_csv(self.raw_path, sep=";")

        logger.info(f"Loaded dataset — {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def basic_checks(self, df: pd.DataFrame) -> None:
        """Simple non-heavy checks before Great Expectations."""
        logger.info("Running basic schema checks...")

        required_cols = [
            "age", "job", "marital", "education",
            "default", "housing", "loan",
            "contact", "month", "day_of_week",
            "duration", "campaign", "pdays",
            "previous", "poutcome",
            "emp.var.rate", "cons.price.idx", "cons.conf.idx",
            "euribor3m", "nr.employed",
            "y"
                ]

        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"❌ Missing required columns: {missing}")

        if df.empty:
            raise ValueError("❌ Dataset is empty — check raw file.")

        if df["y"].isin(["yes", "no"]).sum() != len(df):
            raise ValueError("❌ Target column contains invalid values.")

        logger.info("Basic checks passed.")

    def save_clean_raw(self, df: pd.DataFrame) -> Path:
        output_dir = Path("data/processed")
        output_dir.mkdir(exist_ok=True, parents=True)

        output_path = output_dir / "bank_raw_clean.csv"
        df.to_csv(output_path, index=False)

        logger.info(f"Clean raw saved → {output_path}")
        return output_path

    def run(self) -> pd.DataFrame:
        logger.info("=" * 60)
        logger.info("🚀 Starting Data Ingestion")
        logger.info("=" * 60)

        df = self.load_raw()
        self.basic_checks(df)
        saved_path = self.save_clean_raw(df)

        logger.info("Data Ingestion Completed Successfully.")
        logger.info(f"File available at {saved_path}")

        return df
    


if __name__ == "__main__":
    from loguru import logger

    ingestion = DataIngestion()
    df = ingestion.run()

    logger.info(f"Head of ingested data:\n{df.head().to_string()}")