"""Classification metric computation on scored evaluation frames.

Consumes the scored frames produced by ``model_factory`` (columns ``y_true``,
``y_prob``, ``y_pred``) and returns a flat name -> value mapping.

Metric conventions:

* Every metric records its ``higher_is_better`` orientation in
  ``METRIC_ORIENTATION`` so the comparison engine treats log loss correctly.
* Threshold metrics (precision/recall/F1) use the frame's ``y_pred``;
  ranking metrics (ROC-AUC, PR-AUC, log loss) use ``y_prob``.
* Degenerate cases (single-class truth) return ``math.nan`` for undefined
  metrics rather than raising - the comparison engine skips NaNs explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.model_factory import Y_PRED, Y_PROB, Y_TRUE

CLASSIFICATION_METRICS: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "log_loss",
)

METRIC_ORIENTATION: dict[str, bool] = {
    "accuracy": True,
    "precision": True,
    "recall": True,
    "f1": True,
    "roc_auc": True,
    "pr_auc": True,
    "log_loss": False,  # lower is better
}


@dataclass(frozen=True)
class MetricReport:
    """All metrics for one scored evaluation frame."""

    values: dict[str, float]
    n_samples: int
    positive_rate: float
    confusion: tuple[tuple[int, int], tuple[int, int]]  # ((tn, fp), (fn, tp))


def _single_class(y_true: np.ndarray) -> bool:
    return np.unique(y_true).size < 2


def compute_metrics(scores: pd.DataFrame) -> MetricReport:
    """Compute the full classification metric set for one scored frame."""
    required = {Y_TRUE, Y_PROB, Y_PRED}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"scored frame missing columns: {sorted(missing)}")
    if len(scores) == 0:
        raise ValueError("scored frame is empty")

    y_true = scores[Y_TRUE].to_numpy()
    y_prob = scores[Y_PROB].to_numpy()
    y_pred = scores[Y_PRED].to_numpy()

    values: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if _single_class(y_true):
        values["roc_auc"] = math.nan
        values["pr_auc"] = math.nan
        values["log_loss"] = math.nan
    else:
        values["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        values["pr_auc"] = float(average_precision_score(y_true, y_prob))
        values["log_loss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return MetricReport(
        values=values,
        n_samples=int(len(scores)),
        positive_rate=float(np.mean(y_true)),
        confusion=((int(cm[0, 0]), int(cm[0, 1])), (int(cm[1, 0]), int(cm[1, 1]))),
    )
