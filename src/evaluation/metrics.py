"""Metric computation on scored evaluation frames, for both task types.

Consumes the scored frames produced by ``model_factory``:

* classification frames carry ``y_true``, ``y_prob``, ``y_pred``;
* regression frames carry ``y_true``, ``y_pred`` (no ``y_prob``).

Metric conventions:

* Every metric records its ``higher_is_better`` orientation in
  ``METRIC_ORIENTATION`` so the comparison engine treats error metrics
  (log loss, MAE, RMSE) correctly.
* Classification threshold metrics (precision/recall/F1) use ``y_pred``;
  ranking metrics (ROC-AUC, PR-AUC, log loss) use ``y_prob``.
* Regression metrics: MAE, RMSE, R-squared. MAPE is deliberately excluded:
  the synthetic target is zero-crossing, so division by ``y_true`` is
  undefined in expectation; MAPE is only meaningful for strictly positive
  targets and would be statistical theater here.
* Degenerate cases (single-class truth, zero-variance truth) return
  ``math.nan`` for undefined metrics rather than raising - the comparison
  engine skips NaNs explicitly.
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
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.data.synthetic_generator import TaskType
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

REGRESSION_METRICS: tuple[str, ...] = (
    "mae",
    "rmse",
    "r2",
)

METRICS_BY_TASK: dict[TaskType, tuple[str, ...]] = {
    TaskType.CLASSIFICATION: CLASSIFICATION_METRICS,
    TaskType.REGRESSION: REGRESSION_METRICS,
}

METRIC_ORIENTATION: dict[str, bool] = {
    "accuracy": True,
    "precision": True,
    "recall": True,
    "f1": True,
    "roc_auc": True,
    "pr_auc": True,
    "log_loss": False,  # lower is better
    "mae": False,       # lower is better
    "rmse": False,      # lower is better
    "r2": True,
}


@dataclass(frozen=True)
class MetricReport:
    """All metrics for one scored evaluation frame.

    ``positive_rate`` and ``confusion`` are classification-only;
    for regression frames they are ``nan`` and ``None`` respectively.
    """

    task: TaskType
    values: dict[str, float]
    n_samples: int
    positive_rate: float
    confusion: tuple[tuple[int, int], tuple[int, int]] | None


def _single_class(y_true: np.ndarray) -> bool:
    return np.unique(y_true).size < 2


def infer_task(scores: pd.DataFrame) -> TaskType:
    """A scored frame with ``y_prob`` is classification; without, regression."""
    return TaskType.CLASSIFICATION if Y_PROB in scores.columns else TaskType.REGRESSION


def _classification_values(scores: pd.DataFrame) -> dict[str, float]:
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
    return values


def _regression_values(scores: pd.DataFrame) -> dict[str, float]:
    y_true = scores[Y_TRUE].to_numpy(dtype=float)
    y_pred = scores[Y_PRED].to_numpy(dtype=float)
    values: dict[str, float] = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
    }
    # R-squared is undefined for zero-variance truth.
    values["r2"] = (
        math.nan if float(np.var(y_true)) == 0.0 else float(r2_score(y_true, y_pred))
    )
    return values


def compute_metrics(scores: pd.DataFrame, task: TaskType | None = None) -> MetricReport:
    """Compute the full metric set for one scored frame.

    ``task=None`` infers the task from the frame's columns.
    """
    if task is None:
        task = infer_task(scores)
    required = {Y_TRUE, Y_PRED} | ({Y_PROB} if task is TaskType.CLASSIFICATION else set())
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"scored frame missing columns: {sorted(missing)}")
    if len(scores) == 0:
        raise ValueError("scored frame is empty")

    if task is TaskType.CLASSIFICATION:
        values = _classification_values(scores)
        y_true = scores[Y_TRUE].to_numpy()
        y_pred = scores[Y_PRED].to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        return MetricReport(
            task=task,
            values=values,
            n_samples=int(len(scores)),
            positive_rate=float(np.mean(y_true)),
            confusion=((int(cm[0, 0]), int(cm[0, 1])), (int(cm[1, 0]), int(cm[1, 1]))),
        )

    return MetricReport(
        task=task,
        values=_regression_values(scores),
        n_samples=int(len(scores)),
        positive_rate=math.nan,
        confusion=None,
    )
