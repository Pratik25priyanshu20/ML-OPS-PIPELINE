#src/data/feature_engineering.py 
"""
Feature Engineering Module for Bank Marketing Dataset

Creates advanced features and interactions to improve model performance.
This module is called from the main preprocessing pipeline.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from loguru import logger


class FeatureEngineer:
    """
    Advanced feature engineering for bank marketing prediction.
    Controlled by configs/data_config.yaml -> data.feature_engineering
    """

    def __init__(self, config_path: str = "configs/data_config.yaml") -> None:
        with open(config_path, "r") as f:
            self.config: Dict = yaml.safe_load(f)
        self.feature_names: List[str] = []
        logger.info("FeatureEngineer initialized with config from {}", config_path)

    @property
    def fe_cfg(self) -> Dict:
        return self.config["data"].get("feature_engineering", {})

    # ---------------------- feature blocks ----------------------

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating interaction features...")
        df_new = df.copy()

        # Age-based interactions
        if "age" in df.columns:
            df_new["age_group"] = pd.cut(
                df["age"],
                bins=[0, 25, 35, 45, 55, 65, 100],
                labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
            )

            if "job" in df.columns:
                df_new["young_professional"] = (
                    (df["age"] <= 35)
                    & (df["job"].isin(["management", "technician", "admin."]))
                ).astype(int)

        # Campaign effectiveness features
        if {"campaign", "previous"}.issubset(df.columns):
            df_new["total_contacts"] = df["campaign"] + df["previous"]
            df_new["contact_intensity"] = df["campaign"] / (df["previous"] + 1)
            df_new["has_previous_contact"] = (df["previous"] > 0).astype(int)

        # Financial situation features
        if {"default", "housing", "loan"}.issubset(df.columns):
            df_new["financial_burden"] = (
                (df["default"] == "yes").astype(int)
                + (df["housing"] == "yes").astype(int)
                + (df["loan"] == "yes").astype(int)
            )
            df_new["debt_free"] = (
                (df["default"] == "no")
                & (df["housing"] == "no")
                & (df["loan"] == "no")
            ).astype(int)

        # Previous campaign success
        if "poutcome" in df.columns:
            df_new["prev_success"] = (df["poutcome"] == "success").astype(int)
            df_new["prev_failure"] = (df["poutcome"] == "failure").astype(int)

        # Contact timing features
        if "pdays" in df.columns:
            df_new["recent_contact"] = ((df["pdays"] > 0) & (df["pdays"] <= 30)).astype(
                int
            )
            df_new["never_contacted"] = (df["pdays"] == 999).astype(int)
            df_new["log_pdays"] = np.log1p(df["pdays"])

        # Education-Job interaction
        if {"education", "job"}.issubset(df.columns):
            df_new["educated_professional"] = (
                df["education"].isin(["university.degree", "professional.course"])
                & df["job"].isin(["management", "technician", "admin."])
            ).astype(int)

        # Economic indicator interactions
        if {"emp.var.rate", "cons.price.idx"}.issubset(df.columns):
            df_new["economic_health"] = df["emp.var.rate"] * df["cons.price.idx"]

        if {"cons.conf.idx", "euribor3m"}.issubset(df.columns):
            df_new["sentiment_rate_interaction"] = (
                df["cons.conf.idx"] * df["euribor3m"]
            )

        # Marital status interactions
        if {"marital", "age"}.issubset(df.columns):
            df_new["young_married"] = (
                (df["marital"] == "married") & (df["age"] < 35)
            ).astype(int)

        logger.info(
            "Interaction features added. New total columns: {} -> {}",
            df.shape[1],
            df_new.shape[1],
        )
        return df_new

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating temporal features...")
        df_new = df.copy()

        # Month features
        if "month" in df.columns:
            month_map = {
                "jan": 1,
                "feb": 2,
                "mar": 3,
                "apr": 4,
                "may": 5,
                "jun": 6,
                "jul": 7,
                "aug": 8,
                "sep": 9,
                "oct": 10,
                "nov": 11,
                "dec": 12,
            }
            df_new["month_numeric"] = df["month"].map(month_map)

            df_new["quarter"] = pd.cut(
                df_new["month_numeric"],
                bins=[0, 3, 6, 9, 12],
                labels=["Q1", "Q2", "Q3", "Q4"],
            )

            df_new["is_summer"] = df["month"].isin(["jun", "jul", "aug"]).astype(int)
            df_new["is_winter"] = df["month"].isin(["dec", "jan", "feb"]).astype(int)
            df_new["is_year_end"] = df["month"].isin(["nov", "dec"]).astype(int)

        # Day-of-week features
        if "day_of_week" in df.columns:
            df_new["is_weekend"] = df["day_of_week"].isin(["fri"]).astype(int)
            df_new["is_monday"] = (df["day_of_week"] == "mon").astype(int)

        logger.info(
            "Temporal features added. New total columns: {} -> {}",
            df.shape[1],
            df_new.shape[1],
        )
        return df_new

    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating statistical features...")
        df_new = df.copy()

        # Campaign efficiency
        if {"campaign", "previous"}.issubset(df.columns):
            df_new["campaign_efficiency"] = df["previous"] / (df["campaign"] + 1)

        # Economic indicators normalized
        economic_cols = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m"]
        if all(col in df.columns for col in economic_cols):
            for col in economic_cols:
                std = df[col].std()
                if std == 0 or np.isnan(std):
                    df_new[f"{col}_std"] = 0.0
                else:
                    df_new[f"{col}_std"] = (df[col] - df[col].mean()) / std

            df_new["economic_sentiment"] = (
                df_new["emp.var.rate_std"]
                + df_new["cons.price.idx_std"]
                + df_new["cons.conf.idx_std"]
                + df_new["euribor3m_std"]
            ) / 4.0

        # Age stats
        if "age" in df.columns:
            df_new["age_deviation"] = df["age"] - df["age"].mean()
            df_new["age_percentile"] = df["age"].rank(pct=True)

        logger.info(
            "Statistical features added. New total columns: {} -> {}",
            df.shape[1],
            df_new.shape[1],
        )
        return df_new

    def create_binned_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating binned features...")
        df_new = df.copy()

        if "campaign" in df.columns:
            df_new["campaign_frequency"] = pd.cut(
                df["campaign"],
                bins=[0, 1, 3, 5, 100],
                labels=["low", "medium", "high", "very_high"],
            )

        if "previous" in df.columns:
            df_new["previous_contacts_group"] = pd.cut(
                df["previous"],
                bins=[-1, 0, 1, 3, 100],
                labels=["none", "low", "medium", "high"],
            )

        logger.info(
            "Binned features added. New total columns: {} -> {}",
            df.shape[1],
            df_new.shape[1],
        )
        return df_new

    # ---------------------- main entry ----------------------

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all enabled feature engineering steps.
        """
        logger.info("=" * 60)
        logger.info("🚀 Starting Feature Engineering Pipeline")
        logger.info("=" * 60)

        fe_cfg = self.fe_cfg
        if not fe_cfg.get("enabled", False):
            logger.info("Feature engineering disabled in config. Returning original DF.")
            self.feature_names = df.columns.tolist()
            return df

        df_engineered = df.copy()
        initial_cols = df_engineered.shape[1]

        if fe_cfg.get("create_interactions", True):
            df_engineered = self.create_interaction_features(df_engineered)

        if fe_cfg.get("date_features", True):
            df_engineered = self.create_temporal_features(df_engineered)

        if fe_cfg.get("statistical_features", True):
            df_engineered = self.create_statistical_features(df_engineered)

        if fe_cfg.get("binned_features", False):
            df_engineered = self.create_binned_features(df_engineered)

        final_cols = df_engineered.shape[1]
        self.feature_names = df_engineered.columns.tolist()

        logger.info("Feature Engineering Complete")
        logger.info("Initial features: {}", initial_cols)
        logger.info("Final features: {}", final_cols)
        logger.info("New features created: {}", final_cols - initial_cols)
        logger.info("=" * 60)

        return df_engineered

    def get_feature_names(self) -> List[str]:
        return self.feature_names