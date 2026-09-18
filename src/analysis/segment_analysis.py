"""Segment-level performance comparison.

Compares baseline vs candidate performance within each level of each
segment column (region, customer_type, risk_band) to catch degradation that
overall metrics hide.

Method
------
For each segment level with at least ``config.min_segment_size`` rows in
BOTH windows, compute accuracy, recall, precision and F1 per side, then
bootstrap the accuracy difference within the segment for a p-value.

Multiple comparisons: with ~9 segment levels tested, a raw alpha of 0.05
would produce false alarms; p-values are Bonferroni-adjusted across the
levels actually tested. Conservative and simple - documented trade-off.

Small segments are reported as SKIPPED, never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.model_factory import Y_PRED, Y_TRUE
from src.utils.config import DetectionConfig


@dataclass(frozen=True)
class SegmentResult:
    column: str
    level: str
    n_baseline: int
    n_candidate: int
    baseline: dict[str, float]      # accuracy, precision, recall, f1
    candidate: dict[str, float]
    accuracy_diff: float            # candidate - baseline
    p_value_adjusted: float
    degraded: bool                  # practically AND statistically degraded
    skipped: bool
    skip_reason: str


@dataclass(frozen=True)
class SegmentReport:
    results: tuple[SegmentResult, ...]
    degraded_segments: tuple[str, ...]   # "column=level" labels
    any_segment_regression: bool


def _threshold_metrics(df: pd.DataFrame) -> dict[str, float]:
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


def _bootstrap_accuracy_p(
    base: pd.DataFrame, cand: pd.DataFrame, config: DetectionConfig, seed: int
) -> float:
    rng = np.random.default_rng(seed)
    by = (base[Y_TRUE].to_numpy() == base[Y_PRED].to_numpy()).astype(float)
    cy = (cand[Y_TRUE].to_numpy() == cand[Y_PRED].to_numpy()).astype(float)
    nb, nc = len(by), len(cy)
    iters = config.bootstrap_iterations
    diffs = np.empty(iters)
    for i in range(iters):
        diffs[i] = cy[rng.integers(0, nc, nc)].mean() - by[rng.integers(0, nb, nb)].mean()
    p = 2.0 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
    return float(np.clip(p, 1.0 / iters, 1.0))


def analyze_segments(
    baseline_scores: pd.DataFrame,
    candidate_scores: pd.DataFrame,
    config: DetectionConfig,
) -> SegmentReport:
    """Per-segment baseline vs candidate comparison with adjusted p-values."""
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
                        accuracy_diff=float("nan"),
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
        mb, mc = _threshold_metrics(b), _threshold_metrics(c)
        diff = mc["accuracy"] - mb["accuracy"]
        p_adj = min(
            1.0,
            n_tests * _bootstrap_accuracy_p(b, c, config, seed=config.random_seed + i),
        )
        rel = abs(diff) / mb["accuracy"] if mb["accuracy"] > 0 else float("inf")
        practically = (
            diff < 0
            and abs(diff) >= config.min_absolute_degradation
            and rel >= config.min_relative_degradation
        )
        results.append(
            SegmentResult(
                column=col, level=level,
                n_baseline=len(b), n_candidate=len(c),
                baseline=mb, candidate=mc,
                accuracy_diff=float(diff),
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
