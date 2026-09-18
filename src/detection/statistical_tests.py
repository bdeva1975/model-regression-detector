"""Statistical validation of metric differences via bootstrap resampling.

Approach
--------
The baseline and candidate are evaluated on *independent* evaluation windows,
so we use a two-sample bootstrap: resample each window with replacement,
recompute the metric on each resample, and study the distribution of the
difference ``candidate - baseline``.

Reported quantities:

* percentile confidence interval for the difference at ``1 - alpha``;
* an achieved significance level (bootstrap p-value):
  ``p = 2 * min(P(diff <= 0), P(diff >= 0))``, clipped to (1/B, 1].

Assumptions and limitations (documented, not hidden):

* observations within each window are i.i.d. - true for our synthetic data,
  an approximation for real production windows with temporal correlation;
* percentile intervals can be slightly off for strongly skewed statistics at
  small n; we require ``config.min_sample_size`` before testing at all;
* resamples that collapse to a single class are skipped for metrics that are
  undefined there (AUC, log loss).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.model_factory import Y_PRED, Y_PROB, Y_TRUE
from src.utils.config import DetectionConfig


def _metric_value(name: str, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> float:
    if name == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if name == "precision":
        return float(precision_score(y_true, y_pred, zero_division=0))
    if name == "recall":
        return float(recall_score(y_true, y_pred, zero_division=0))
    if name == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    single_class = np.unique(y_true).size < 2
    if single_class:
        return float("nan")
    if name == "roc_auc":
        return float(roc_auc_score(y_true, y_prob))
    if name == "pr_auc":
        return float(average_precision_score(y_true, y_prob))
    if name == "log_loss":
        return float(log_loss(y_true, y_prob, labels=[0, 1]))
    raise ValueError(f"unknown metric: {name}")


@dataclass(frozen=True)
class BootstrapResult:
    """Two-sample bootstrap comparison of one metric."""

    metric: str
    baseline_value: float
    candidate_value: float
    diff: float                   # candidate - baseline (raw, unoriented)
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool             # p_value < alpha
    n_baseline: int
    n_candidate: int
    n_effective_iterations: int   # iterations that produced a defined diff


def bootstrap_metric_diff(
    baseline_scores: pd.DataFrame,
    candidate_scores: pd.DataFrame,
    metric: str,
    config: DetectionConfig,
) -> BootstrapResult:
    """Bootstrap the difference ``candidate - baseline`` for one metric."""
    nb, nc = len(baseline_scores), len(candidate_scores)
    if min(nb, nc) < config.min_sample_size:
        raise ValueError(
            f"sample sizes ({nb}, {nc}) below min_sample_size={config.min_sample_size}; "
            "statistical comparison would be unreliable"
        )

    bt = baseline_scores[Y_TRUE].to_numpy()
    bp = baseline_scores[Y_PROB].to_numpy()
    bd = baseline_scores[Y_PRED].to_numpy()
    ct = candidate_scores[Y_TRUE].to_numpy()
    cp = candidate_scores[Y_PROB].to_numpy()
    cd = candidate_scores[Y_PRED].to_numpy()

    base_val = _metric_value(metric, bt, bp, bd)
    cand_val = _metric_value(metric, ct, cp, cd)

    rng = np.random.default_rng(config.random_seed)
    diffs = np.empty(config.bootstrap_iterations)
    k = 0
    for _ in range(config.bootstrap_iterations):
        ib = rng.integers(0, nb, size=nb)
        ic = rng.integers(0, nc, size=nc)
        b = _metric_value(metric, bt[ib], bp[ib], bd[ib])
        c = _metric_value(metric, ct[ic], cp[ic], cd[ic])
        d = c - b
        if np.isnan(d):
            continue
        diffs[k] = d
        k += 1

    if k < max(100, config.bootstrap_iterations // 2):
        raise ValueError(
            f"only {k} effective bootstrap iterations for {metric}; "
            "data too degenerate for a reliable interval"
        )
    diffs = diffs[:k]

    alpha = config.significance_level
    ci_low, ci_high = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_le = float(np.mean(diffs <= 0.0))
    p_ge = float(np.mean(diffs >= 0.0))
    p_value = float(np.clip(2.0 * min(p_le, p_ge), 1.0 / k, 1.0))

    return BootstrapResult(
        metric=metric,
        baseline_value=base_val,
        candidate_value=cand_val,
        diff=cand_val - base_val,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=p_value,
        significant=p_value < alpha,
        n_baseline=nb,
        n_candidate=nc,
        n_effective_iterations=k,
    )
