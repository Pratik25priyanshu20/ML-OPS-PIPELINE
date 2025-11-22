#src/data/validation.py
"""
Data Validation using Great Expectations (V3 API)
"""

from pathlib import Path
from typing import Dict
import pandas as pd
import yaml
from loguru import logger
import great_expectations as ge
from great_expectations.core.batch import BatchRequest


class DataValidator:
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        self.config = self._load_config(config_path)
        self.context = ge.get_context()

        logger.info("Great Expectations context initialized.")

    def _load_config(self, path: str) -> Dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def validate(self, df: pd.DataFrame) -> Dict:
        """
        Validates the dataframe using an GE Validator.
        """
        logger.info("Starting Great Expectations validation...")

        validator = self.context.sources.pandas_default.read_dataframe(df)

        results = {}

        # 1. Check required columns exist
        required_cols = (
            self.config["columns"]["numerical"]
            + self.config["columns"]["categorical"]
            + [self.config["target"]["column"]]
        )

        results["columns_exist"] = validator.expect_table_columns_to_match_set(
            column_set=required_cols
        ).to_json_dict()

        # 2. Age range validation
        results["age_valid"] = validator.expect_column_values_to_be_between(
            "age", min_value=17, max_value=100
        ).to_json_dict()

        # 3. Target values must be yes/no
        results["target_set"] = validator.expect_column_values_to_be_in_set(
            self.config["target"]["column"], ["yes", "no"]
        ).to_json_dict()

        # 4. No missing values in important columns
        for col in ["age", "job", "marital", self.config["target"]["column"]]:
            results[f"{col}_not_null"] = validator.expect_column_values_to_not_be_null(col).to_json_dict()

        # 5. No duplicate rows
        duplicates = df.duplicated().sum()
        results["duplicate_rows"] = {"duplicates": int(duplicates)}

        # Save HTML report
        output_dir = Path("data/validation_reports")
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / "validation_report.html"
        self.context.build_data_docs()
        logger.info(f"Validation report saved → {report_path}")

        logger.success("Validation completed.")

        return results