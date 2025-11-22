"""
Evaluation utilities for classification models.
Computes metrics and saves diagnostic plots for MLflow.
"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn import metrics


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute a standard set of binary classification metrics.
    """
    logger.info("Computing classification metrics...")

    acc = metrics.accuracy_score(y_true, y_pred)
    prec = metrics.precision_score(y_true, y_pred, zero_division=0)
    rec = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)

    metrics_dict: Dict[str, float] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }

    # ROC-AUC & PR-AUC require probabilities
    if y_proba is not None:
        try:
            roc_auc = metrics.roc_auc_score(y_true, y_proba)
            pr_auc = metrics.average_precision_score(y_true, y_proba)
            metrics_dict["roc_auc"] = roc_auc
            metrics_dict["pr_auc"] = pr_auc
        except Exception as e:
            logger.warning(f"Could not compute ROC/PR AUC: {e}")

    logger.info(f"Metrics: {metrics_dict}")
    return metrics_dict


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    labels: Optional[list] = None,
) -> None:
    """
    Save confusion matrix plot to disk.
    """
    logger.info(f"Saving confusion matrix plot to {out_path}")

    cm = metrics.confusion_matrix(y_true, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_path: Path,
) -> None:
    """
    Save ROC curve plot to disk.
    """
    logger.info(f"Saving ROC curve plot to {out_path}")

    fpr, tpr, _ = metrics.roc_curve(y_true, y_proba)
    auc = metrics.roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)