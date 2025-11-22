# src/data/preprocessing.py
"""
Preprocessing Module for Bank Marketing Dataset
- Uses configs/data_config.yaml (data.* keys)
- Handles:
  * target encoding
  * leakage column drop
  * automatic categorical detection (original + FE-added)
  * one-hot encoding
  * scaling
  * train/test split
  * saving scaler + feature_columns + npy arrays
"""

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self, config_path: str = "configs/data_config.yaml") -> None:
        with open(config_path, "r") as f:
            self.config: Dict = yaml.safe_load(f)

        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []

        logger.info("DataPreprocessor initialized with config from {}", config_path)

    # ---------- helpers over config ----------

    @property
    def _data_cfg(self) -> Dict:
        return self.config["data"]

    @property
    def target_col(self) -> str:
        return self._data_cfg["target_column"]

    @property
    def target_mapping(self) -> Dict:
        return self._data_cfg["target_mapping"]

    @property
    def orig_cat_cols(self) -> List[str]:
        """Categorical columns defined in config BEFORE FE."""
        return self._data_cfg["categorical_features"]

    @property
    def num_cols(self) -> List[str]:
        return self._data_cfg["numerical_features"]

    # ---------- core steps ----------

    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Encoding target column '{}' with mapping {}", self.target_col, self.target_mapping)
        df = df.copy()
        df[self.target_col] = df[self.target_col].map(self.target_mapping)
        if df[self.target_col].isnull().any():
            raise ValueError("Target mapping produced NaNs – check target_mapping in config.")
        return df

    def drop_leakage(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = self.config.get("drop_columns", [])
        if drop_cols:
            logger.warning("Dropping potential leakage/unused columns: {}", drop_cols)
        return df.drop(columns=drop_cols, errors="ignore")

    def train_test_split(self, df: pd.DataFrame):
        logger.info("Splitting data into train/test...")
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        test_size = self._data_cfg["test_size"]
        random_state = self._data_cfg["random_state"]
        stratify_flag = self._data_cfg["stratify"]

        stratify = y if stratify_flag else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        logger.info("Train size: {} rows, Test size: {} rows", len(X_train), len(X_test))
        logger.info("Train target distribution: {}", y_train.value_counts().to_dict())
        logger.info("Test target distribution: {}", y_test.value_counts().to_dict())

        return X_train, X_test, y_train, y_test

    # ---------------------------------------------------------------------
    # AUTO-DETECT CATEGORICAL COLUMNS (critical fix)
    # ---------------------------------------------------------------------
    def detect_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Automatically detect new FE-created categorical columns + original ones.
        """
        auto_cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
        merged = list(set(self.orig_cat_cols + auto_cats))

        logger.info("Auto-detected categorical columns: {}", auto_cats)
        logger.info("Final categorical columns to encode: {}", merged)
        return merged

    def encode_categorical(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        One-hot encode categorical columns.
        fit=True  → learn final columns
        fit=False → align to learned columns
        """
        # detect FE + original categories
        cat_cols = self.detect_categorical_columns(X)

        logger.info("One-hot encoding categorical columns: {}", cat_cols)
        X = X.copy()

        if fit:
            X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)
            self.feature_columns = X_enc.columns.tolist()
        else:
            X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)

            # add missing columns
            for col in self.feature_columns:
                if col not in X_enc.columns:
                    X_enc[col] = 0

            # keep training column order
            X_enc = X_enc[self.feature_columns]

        logger.info("Encoded feature dimensionality: {} columns", X_enc.shape[1])
        return X_enc

    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        logger.info("Scaling features with StandardScaler (fit = {})", fit)
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)

    def save_artifacts(self) -> None:
        feat_dir = Path(self.config["processed_data"]["features_path"])
        feat_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.scaler, feat_dir / "scaler.pkl")
        joblib.dump(self.feature_columns, feat_dir / "feature_columns.pkl")

        logger.info("Saved scaler and feature_columns to {}", feat_dir)

    # ------------------------------------------------------------------
    # MAIN PIPELINE
    # ------------------------------------------------------------------
    def preprocess_for_training(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logger.info("=" * 60)
        logger.info("🚀 Starting Preprocessing Pipeline (Training)")
        logger.info("=" * 60)

        # 1) drop leakage / unused cols
        df = self.drop_leakage(df)

        # 2) encode target
        df = self.encode_target(df)

        # 3) split
        X_train, X_test, y_train, y_test = self.train_test_split(df)

        # 4) encode categoricals
        X_train_enc = self.encode_categorical(X_train, fit=True)
        X_test_enc = self.encode_categorical(X_test, fit=False)

        # 5) scale
        X_train_scaled = self.scale_features(X_train_enc, fit=True)
        X_test_scaled = self.scale_features(X_test_enc, fit=False)

        # 6) save artifacts
        self.save_artifacts()

        logger.info("✅ Preprocessing complete.")
        logger.info("X_train shape: {}", X_train_scaled.shape)
        logger.info("X_test shape: {}", X_test_scaled.shape)

        return X_train_scaled, X_test_scaled, y_train.values, y_test.values


# ---------------------------------------------------------------------
# EXECUTION BLOCK
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from src.data.feature_engineering import FeatureEngineer

    raw_clean_path = Path("data/processed/bank_raw_clean.csv")
    if not raw_clean_path.exists():
        raise FileNotFoundError(
            f"{raw_clean_path} not found. Run `python -m src.data.ingestion` first."
        )

    # load cleaned raw
    df_raw = pd.read_csv(raw_clean_path)

    # apply feature engineering
    fe = FeatureEngineer()
    df_fe = fe.engineer_features(df_raw)

    # preprocess engineered data
    pre = DataPreprocessor()
    X_train, X_test, y_train, y_test = pre.preprocess_for_training(df_fe)

    # save numpy arrays
    feats_dir = Path("data/features")
    feats_dir.mkdir(parents=True, exist_ok=True)
    np.save(feats_dir / "X_train.npy", X_train)
    np.save(feats_dir / "X_test.npy", X_test)
    np.save(feats_dir / "y_train.npy", y_train)
    np.save(feats_dir / "y_test.npy", y_test)

    logger.info("Saved preprocessed arrays to {}", feats_dir)
    logger.info("Train samples: {}, Test samples: {}", len(X_train), len(X_test))
    logger.info("Feature dimension: {}", X_train.shape[1])