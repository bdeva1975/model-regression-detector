"""Segment-level performance comparison, for both task types.

Compares baseline vs candidate performance within each level of each
segment column (region, customer_type, risk_band) to catch degradation that
overall metrics hide.

Method
------
For each segment level with at least ``config.min_segment_size`` rows in
BOTH windows:

* classification: accuracy, precision, recall and F1 per side; the tested
  statistic is ACCURACY (per-row correctness, mean-of-losses bootstrap);
* regression: MAE, RMSE and bias (mean error) per side; the tested
  statistic is MAE (per-row absolute error, mean-of-losses bootstrap).

The bootstrap operates on per-row loss values (correctness / absolute
error), so one procedure serves both tasks. Degradation direction follows
the statistic's orientation: accuracy degrades downward, MAE upward.

Multiple comparisons: with ~9 segment levels tested, a raw alpha of 0.05
would produce false alarms; p-values are Bonferroni-adjusted across the
levels actually tested. Conservative and simple - documented trade-off.

Small segments are reported as SKIPPED, never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.synthetic_generator import TaskType
from src.evaluation.metrics import infer_task
from src.models.model_factory import Y_PRED, Y_TRUE
from src.utils.config import DetectionConfig


@dataclass(frozen=True)
class SegmentResult:
    column: str
    level: str
    n_baseline: int
    n_candidate: int
    baseline: dict[str, float]
    candidate: dict[str, float]
    tested_metric: str              # "accuracy" | "mae"
    metric_diff: float              # candidate - baseline, raw
    p_value_adjusted: float
    degraded: bool                  # practically AND statistically degraded
    skipped: bool
    skip_reason: str


@dataclass(frozen=True)
class SegmentReport:
    results: tuple[SegmentResult, ...]
    degraded_segments: tuple[str, ...]   # "column=level" labels
    any_segment_regression: bool


def _classification_metrics(df: pd.DataFrame) -> dict[str, float]:
    y, p = df[Y_TRUE].to_numpy(), df[Y_PRED].to_numpy()
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": float((y == p).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _regression_metrics(df: pd.DataFrame) -> dict[str, float]:
    y = df[Y_TRUE].to_numpy(dtype=float)
    p = df[Y_PRED].to_numpy(dtype=float)
    err = p - y
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
    }


def _row_losses(df: pd.DataFrame, task: TaskType) -> np.ndarray:
    """Per-row statistic whose mean is the tested segment metric."""
    y = df[Y_TRUE].to_numpy(dtype=float)
    p = df[Y_PRED].to_numpy(dtype=float)
    if task is TaskType.CLASSIFICATION:
        return (y == p).astype(float)      # mean = accuracy
    return np.abs(p - y)                    # mean = MAE


def _bootstrap_mean_diff_p(
    base_vals: np.ndarray,
    cand_vals: np.ndarray,
    config: DetectionConfig,
    seed: int,
) -> float:
    """Two-sample bootstrap p-value for mean(cand) - mean(base)."""
    rng = np.random.default_rng(seed)
    nb, nc = len(base_vals), len(cand_vals)
    iters = config.bootstrap_iterations
    diffs = np.empty(iters)
    for i in range(iters):
        diffs[i] = (
            cand_vals[rng.integers(0, nc, nc)].mean()
            - base_vals[rng.integers(0, nb, nb)].mean()
        )
    p = 2.0 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
    return float(np.clip(p, 1.0 / iters, 1.0))


def analyze_segments(
    baseline_scores: pd.DataFrame,
    candidate_scores: pd.DataFrame,
    config: DetectionConfig,
    task: TaskType | None = None,
) -> SegmentReport:
    """Per-segment baseline vs candidate comparison with adjusted p-values."""
    if task is None:
        task = infer_task(baseline_scores)
    is_classification = task is TaskType.CLASSIFICATION
    metrics_fn = _classification_metrics if is_classification else _regression_metrics
    tested_metric = "accuracy" if is_classification else "mae"
    higher_is_better = is_classification   # accuracy up-good; MAE down-good

    raw: list[tuple[str, str, pd.DataFrame, pd.DataFrame]] = []
    skipped: list[SegmentResult] = []

    for col in config.segment_columns:
        if col not in baseline_scores.columns or col not in candidate_scores.columns:
            continue
        levels = sorted(set(baseline_scores[col]) | set(candidate_scores[col]))
        for level in levels:
            b = baseline_scores[baseline_scores[col] == level]
            c = candidate_scores[candidate_scores[col] == level]
            if min(len(b), len(c)) < config.min_segment_size:
                skipped.append(
                    SegmentResult(
                        column=col, level=str(level),
                        n_baseline=len(b), n_candidate=len(c),
                        baseline={}, candidate={},
                        tested_metric=tested_metric,
                        metric_diff=float("nan"),
                        p_value_adjusted=float("nan"),
                        degraded=False, skipped=True,
                        skip_reason=(
                            f"needs >= {config.min_segment_size} rows per side, "
                            f"got ({len(b)}, {len(c)})"
                        ),
                    )
                )
            else:
                raw.append((col, str(level), b, c))

    n_tests = max(len(raw), 1)
    results: list[SegmentResult] = []
    for i, (col, level, b, c) in enumerate(raw):
        mb, mc = metrics_fn(b), metrics_fn(c)
        diff = mc[tested_metric] - mb[tested_metric]
        degradation = -diff if higher_is_better else diff   # positive = worse
        p_adj = min(
            1.0,
            n_tests
            * _bootstrap_mean_diff_p(
                _row_losses(b, task), _row_losses(c, task),
                config, seed=config.random_seed + i,
            ),
        )
        base_ref = abs(mb[tested_metric])
        rel = degradation / base_ref if base_ref > 0 else (
            float("inf") if degradation > 0 else 0.0
        )
        practically = (
            degradation > 0
            and degradation >= config.min_absolute_degradation
            and rel >= config.min_relative_degradation
        )
        results.append(
            SegmentResult(
                column=col, level=level,
                n_baseline=len(b), n_candidate=len(c),
                baseline=mb, candidate=mc,
                tested_metric=tested_metric,
                metric_diff=float(diff),
                p_value_adjusted=p_adj,
                degraded=bool(practically and p_adj < config.significance_level),
                skipped=False, skip_reason="",
            )
        )

    results.extend(skipped)
    degraded = tuple(f"{r.column}={r.level}" for r in results if r.degraded)
    return SegmentReport(
        results=tuple(results),
        degraded_segments=degraded,
        any_segment_regression=len(degraded) > 0,
    )
