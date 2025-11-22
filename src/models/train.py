# src/models/train.py
"""
Model Training Module for Bank Marketing Churn

- Loads preprocessed features from data/features
- Trains multiple models: Logistic Regression, Random Forest, XGBoost, LightGBM
- Uses Optuna for hyperparameter optimization (per model, controlled by YAML)
- Logs metrics and params to MLflow (tracking URI from model_config.yaml)
- Chooses best model based on training.metric (e.g. 'roc_auc')
- Saves best model + metadata to ./models
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import mlflow
import numpy as np
import optuna
import yaml
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ============================================================
# Dataclasses for config
# ============================================================


@dataclass
class ExperimentConfig:
    name: str
    tracking_uri: str


@dataclass
class TrainingConfig:
    metric: str
    n_splits: int
    random_state: int
    test_size: float


class ModelTrainer:
    """
    Multi-model trainer with Optuna + MLflow.
    Models configured via configs/model_config.yaml:
      - experiment
      - training
      - models: logistic_regression, random_forest, xgboost, lightgbm
    """

    def __init__(
        self,
        model_config_path: str = "configs/model_config.yaml",
        data_config_path: str = "configs/data_config.yaml",
    ) -> None:
        self.model_config_path = model_config_path
        self.data_config_path = data_config_path

        cfg = self._load_yaml(model_config_path)
        self.exp_cfg = ExperimentConfig(
            name=cfg["experiment"]["name"],
            tracking_uri=cfg["experiment"]["tracking_uri"],
        )
        self.train_cfg = TrainingConfig(
            metric=cfg["training"]["metric"],
            n_splits=cfg["training"]["n_splits"],
            random_state=cfg["training"]["random_state"],
            test_size=cfg["training"]["test_size"],
        )
        self.models_cfg: Dict[str, Any] = cfg["models"]

        self.data_cfg = self._load_yaml(data_config_path)["data"]

        self.best_model_name: str | None = None
        self.best_model: Any | None = None
        self.best_score: float = -1.0

        self._setup_mlflow()

        logger.info(
            "ModelTrainer initialized — experiment '{}', primary metric '{}'",
            self.exp_cfg.name,
            self.train_cfg.metric,
        )

    # -----------------------------------------------------
    # Config helpers
    # -----------------------------------------------------

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _setup_mlflow(self) -> None:
        """
        Configure MLflow tracking based on model_config.yaml.
        Example:
          tracking_uri: "file:mlruns"
          name: "bank_marketing_churn_experiment"
        """
        mlflow.set_tracking_uri(self.exp_cfg.tracking_uri)
        mlflow.set_experiment(self.exp_cfg.name)

        logger.info("MLflow tracking URI: {}", self.exp_cfg.tracking_uri)
        logger.info("MLflow experiment: {}", self.exp_cfg.name)

    # -----------------------------------------------------
    # Data loading
    # -----------------------------------------------------

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load preprocessed train/test arrays from data/features.
        Produced by: python -m src.data.preprocessing
        """
        feature_dir = Path("data/features")
        logger.info("Loading preprocessed arrays from {}", feature_dir)

        X_train_path = feature_dir / "X_train.npy"
        X_test_path = feature_dir / "X_test.npy"
        y_train_path = feature_dir / "y_train.npy"
        y_test_path = feature_dir / "y_test.npy"

        if not X_train_path.exists():
            raise FileNotFoundError(
                f"{X_train_path} not found — run `python -m src.data.preprocessing` first."
            )

        X_train = np.load(X_train_path)
        X_test = np.load(X_test_path)
        y_train = np.load(y_train_path)
        y_test = np.load(y_test_path)

        logger.info("Train shape: X={}, y={}", X_train.shape, y_train.shape)
        logger.info("Test shape:  X={}, y={}", X_test.shape, y_test.shape)

        # Log class balance
        unique, counts = np.unique(y_train, return_counts=True)
        logger.info("Train class distribution: {}", dict(zip(unique, counts)))

        return X_train, X_test, y_train, y_test

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
        }

    def get_primary_metric(self, metrics: Dict[str, float]) -> float:
        metric_name = self.train_cfg.metric
        if metric_name not in metrics:
            raise ValueError(
                f"Requested primary metric '{metric_name}' not in metrics {list(metrics.keys())}"
            )
        return float(metrics[metric_name])

    # ============================================================
    # LOGISTIC REGRESSION
    # ============================================================

    def _optuna_objective_lr(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        C = trial.suggest_float("C", 1e-3, 10.0, log=True)
        solver = "lbfgs"

        model = LogisticRegression(
            C=C,
            solver=solver,
            max_iter=1000,
            class_weight="balanced",
            random_state=self.train_cfg.random_state,
        )

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = self.compute_metrics(y_val, y_pred, y_proba)
        return self.get_primary_metric(metrics)

    def train_logistic_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        cfg = self.models_cfg.get("logistic_regression", {})
        if not cfg.get("enabled", False):
            logger.info("Logistic Regression disabled in config; skipping.")
            return {}

        n_trials = cfg.get("optuna", {}).get("n_trials", 0)
        logger.info("Training Logistic Regression (Optuna trials = {})", n_trials)

        # Simple holdout-based tuning (train vs test) – okay for this project
        study = None
        best_params: Dict[str, Any] = {}
        if n_trials > 0:
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: self._optuna_objective_lr(
                    trial, X_train, y_train, X_test, y_test
                ),
                n_trials=n_trials,
            )
            best_params = study.best_params
            logger.info(
                "[LR] Best {} = {:.4f} with params: {}",
                self.train_cfg.metric,
                study.best_value,
                study.best_params,
            )
        else:
            best_params = {"C": 1.0}

        with mlflow.start_run(run_name="logistic_regression"):
            mlflow.log_param("model_type", "logistic_regression")
            for k, v in best_params.items():
                mlflow.log_param(k, v)

            model = LogisticRegression(
                C=best_params.get("C", 1.0),
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced",
                random_state=self.train_cfg.random_state,
            )

            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            metrics = self.compute_metrics(y_test, y_pred, y_proba)
            mlflow.log_metrics(metrics)

            primary = self.get_primary_metric(metrics)
            logger.info(
                "[LR] {} = {:.4f}, F1 = {:.4f}, ROC-AUC = {:.4f}",
                self.train_cfg.metric,
                primary,
                metrics["f1"],
                metrics["roc_auc"],
            )

            mlflow.sklearn.log_model(model, artifact_path="model")

        self._update_best_model("logistic_regression", model, metrics)
        return metrics

    # ============================================================
    # RANDOM FOREST
    # ============================================================

    def _optuna_objective_rf(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        n_estimators = trial.suggest_int("n_estimators", 100, 400)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=self.train_cfg.random_state,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = self.compute_metrics(y_val, y_pred, y_proba)
        return self.get_primary_metric(metrics)

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        cfg = self.models_cfg.get("random_forest", {})
        if not cfg.get("enabled", False):
            logger.info("Random Forest disabled in config; skipping.")
            return {}

        n_trials = cfg.get("optuna", {}).get("n_trials", 0)
        logger.info("Training Random Forest (Optuna trials = {})", n_trials)

        study = None
        best_params: Dict[str, Any] = {}
        if n_trials > 0:
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: self._optuna_objective_rf(
                    trial, X_train, y_train, X_test, y_test
                ),
                n_trials=n_trials,
            )
            best_params = study.best_params
            logger.info(
                "[RF] Best {} = {:.4f} with params: {}",
                self.train_cfg.metric,
                study.best_value,
                study.best_params,
            )
        else:
            best_params = {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 10,
                "min_samples_leaf": 4,
            }

        with mlflow.start_run(run_name="random_forest"):
            mlflow.log_param("model_type", "random_forest")
            for k, v in best_params.items():
                mlflow.log_param(k, v)

            model = RandomForestClassifier(
                n_estimators=best_params.get("n_estimators", 200),
                max_depth=best_params.get("max_depth", 10),
                min_samples_split=best_params.get("min_samples_split", 10),
                min_samples_leaf=best_params.get("min_samples_leaf", 4),
                class_weight="balanced",
                random_state=self.train_cfg.random_state,
                n_jobs=-1,
            )

            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            metrics = self.compute_metrics(y_test, y_pred, y_proba)
            mlflow.log_metrics(metrics)

            primary = self.get_primary_metric(metrics)
            logger.info(
                "[RF] {} = {:.4f}, F1 = {:.4f}, ROC-AUC = {:.4f}",
                self.train_cfg.metric,
                primary,
                metrics["f1"],
                metrics["roc_auc"],
            )

            mlflow.sklearn.log_model(model, artifact_path="model")

        self._update_best_model("random_forest", model, metrics)
        return metrics

    # ============================================================
    # XGBOOST
    # ============================================================

    def _compute_scale_pos_weight(self, y: np.ndarray) -> float:
        unique, counts = np.unique(y, return_counts=True)
        freq = dict(zip(unique, counts))
        neg = freq.get(0, 0)
        pos = freq.get(1, 1)
        spw = neg / pos
        logger.info("scale_pos_weight = {:.3f} (neg={} / pos={})", spw, neg, pos)
        return spw

    def _optuna_objective_xgb(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        scale_pos_weight: float,
    ) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 450),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "scale_pos_weight": scale_pos_weight,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": self.train_cfg.random_state,
            "use_label_encoder": False,
        }

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = self.compute_metrics(y_val, y_pred, y_proba)
        return self.get_primary_metric(metrics)

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        cfg = self.models_cfg.get("xgboost", {})
        if not cfg.get("enabled", False):
            logger.info("XGBoost disabled in config; skipping.")
            return {}

        n_trials = cfg.get("optuna", {}).get("n_trials", 0)
        logger.info("Training XGBoost (Optuna trials = {})", n_trials)

        scale_pos_weight = self._compute_scale_pos_weight(y_train)

        study = None
        best_params: Dict[str, Any] = {}
        if n_trials > 0:
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: self._optuna_objective_xgb(
                    trial, X_train, y_train, X_test, y_test, scale_pos_weight
                ),
                n_trials=n_trials,
            )
            best_params = study.best_params
            logger.info(
                "[XGB] Best {} = {:.4f} with params: {}",
                self.train_cfg.metric,
                study.best_value,
                study.best_params,
            )
        else:
            best_params = {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1,
                "gamma": 0.0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
            }

        with mlflow.start_run(run_name="xgboost"):
            mlflow.log_param("model_type", "xgboost")
            for k, v in best_params.items():
                mlflow.log_param(k, v)
            mlflow.log_param("scale_pos_weight", scale_pos_weight)

            final_params = {
                **best_params,
                "scale_pos_weight": scale_pos_weight,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "n_jobs": -1,
                "random_state": self.train_cfg.random_state,
                "use_label_encoder": False,
            }

            model = XGBClassifier(**final_params)
            model.fit(X_train, y_train)

            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            metrics = self.compute_metrics(y_test, y_pred, y_proba)
            mlflow.log_metrics(metrics)

            primary = self.get_primary_metric(metrics)
            logger.info(
                "[XGB] {} = {:.4f}, F1 = {:.4f}, ROC-AUC = {:.4f}",
                self.train_cfg.metric,
                primary,
                metrics["f1"],
                metrics["roc_auc"],
            )

            mlflow.xgboost.log_model(model, artifact_path="model")

        self._update_best_model("xgboost", model, metrics)
        return metrics

    # ============================================================
    # LIGHTGBM
    # ============================================================

    def _optuna_objective_lgbm(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 450),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 16, 64),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "class_weight": "balanced",
            "random_state": self.train_cfg.random_state,
            "n_jobs": -1,
        }

        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = self.compute_metrics(y_val, y_pred, y_proba)
        return self.get_primary_metric(metrics)

    def train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        cfg = self.models_cfg.get("lightgbm", {})
        if not cfg.get("enabled", False):
            logger.info("LightGBM disabled in config; skipping.")
            return {}

        n_trials = cfg.get("optuna", {}).get("n_trials", 0)
        logger.info("Training LightGBM (Optuna trials = {})", n_trials)

        study = None
        best_params: Dict[str, Any] = {}
        if n_trials > 0:
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: self._optuna_objective_lgbm(
                    trial, X_train, y_train, X_test, y_test
                ),
                n_trials=n_trials,
            )
            best_params = study.best_params
            logger.info(
                "[LGBM] Best {} = {:.4f} with params: {}",
                self.train_cfg.metric,
                study.best_value,
                study.best_params,
            )
        else:
            best_params = {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }

        with mlflow.start_run(run_name="lightgbm"):
            mlflow.log_param("model_type", "lightgbm")
            for k, v in best_params.items():
                mlflow.log_param(k, v)

            final_params = {
                **best_params,
                "class_weight": "balanced",
                "random_state": self.train_cfg.random_state,
                "n_jobs": -1,
            }

            model = LGBMClassifier(**final_params)
            model.fit(X_train, y_train)

            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            metrics = self.compute_metrics(y_test, y_pred, y_proba)
            mlflow.log_metrics(metrics)

            primary = self.get_primary_metric(metrics)
            logger.info(
                "[LGBM] {} = {:.4f}, F1 = {:.4f}, ROC-AUC = {:.4f}",
                self.train_cfg.metric,
                primary,
                metrics["f1"],
                metrics["roc_auc"],
            )

            mlflow.sklearn.log_model(model, artifact_path="model")

        self._update_best_model("lightgbm", model, metrics)
        return metrics

    # ============================================================
    # Best model tracking & saving
    # ============================================================

    def _update_best_model(
        self, name: str, model: Any, metrics: Dict[str, float]
    ) -> None:
        score = self.get_primary_metric(metrics)
        if score > self.best_score:
            logger.info(
                "🏆 '{}' is new best model: {} = {:.4f} (previous best = {:.4f})",
                name,
                self.train_cfg.metric,
                score,
                self.best_score,
            )
            self.best_score = score
            self.best_model_name = name
            self.best_model = model

    def save_best_model(self) -> None:
        if self.best_model is None or self.best_model_name is None:
            logger.warning("No best model to save — training may have skipped all models.")
            return

        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)

        model_path = models_dir / "best_model.pkl"
        meta_path = models_dir / "best_model_metadata.yaml"

        joblib.dump(self.best_model, model_path)

        metadata = {
            "model_name": self.best_model_name,
            "primary_metric": self.train_cfg.metric,
            "best_score": float(self.best_score),
            "saved_at": datetime.utcnow().isoformat(),
            "data": {
                "target_column": self.data_cfg["target_column"],
            },
        }

        with open(meta_path, "w") as f:
            yaml.safe_dump(metadata, f)

        logger.info("💾 Saved best model → {}", model_path)
        logger.info("💾 Saved metadata → {}", meta_path)

    # ============================================================
    # Main pipeline
    # ============================================================

    def run_training_pipeline(self) -> Dict[str, Dict[str, float]]:
        """
        Full training entrypoint:
          1. Load data
          2. Train each enabled model (LR, RF, XGB, LGBM)
          3. Track best model based on training.metric
          4. Save best model + metadata
        """
        logger.info("=" * 80)
        logger.info("🚀 Starting Multi-Model Training Pipeline")
        logger.info("=" * 80)

        X_train, X_test, y_train, y_test = self.load_data()

        results: Dict[str, Dict[str, float]] = {}

        # 1) Logistic Regression
        metrics_lr = self.train_logistic_regression(X_train, y_train, X_test, y_test)
        if metrics_lr:
            results["logistic_regression"] = metrics_lr

        # 2) Random Forest
        metrics_rf = self.train_random_forest(X_train, y_train, X_test, y_test)
        if metrics_rf:
            results["random_forest"] = metrics_rf

        # 3) XGBoost
        metrics_xgb = self.train_xgboost(X_train, y_train, X_test, y_test)
        if metrics_xgb:
            results["xgboost"] = metrics_xgb

        # 4) LightGBM
        metrics_lgbm = self.train_lightgbm(X_train, y_train, X_test, y_test)
        if metrics_lgbm:
            results["lightgbm"] = metrics_lgbm

        # Save best model
        self.save_best_model()

        logger.info("=" * 80)
        logger.info("✅ Training pipeline complete.")
        logger.info("Best model: {} with {} = {:.4f}", self.best_model_name, self.train_cfg.metric, self.best_score)
        logger.info("=" * 80)

        return results


if __name__ == "__main__":
    trainer = ModelTrainer()
    results = trainer.run_training_pipeline()

    # Nice CLI summary
    print("\n📊 MODEL TRAINING RESULTS")
    print("=" * 80)
    if results:
        header = f"{'Model':<22} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC-AUC':<10}"
        print(header)
        print("-" * 80)
        for name, m in results.items():
            print(
                f"{name:<22} "
                f"{m['accuracy']:<10.4f} "
                f"{m['precision']:<10.4f} "
                f"{m['recall']:<10.4f} "
                f"{m['f1']:<10.4f} "
                f"{m['roc_auc']:<10.4f}"
            )
    else:
        print("No models were trained (all disabled in config).")
    print("=" * 80)