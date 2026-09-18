"""Distribution drift detection between baseline and candidate windows.

Techniques (each with explicit assumptions):

* Kolmogorov-Smirnov two-sample test - numeric features and prediction
  scores. Nonparametric; sensitive to any distributional difference; at
  large n it flags trivially small shifts, which is why the PSI magnitude
  accompanies every p-value.
* Chi-square test of homogeneity - categorical features. Requires expected
  counts >= 5 per cell; sparse levels are pooled into "__other__".
* Population Stability Index (PSI) - magnitude measure for numerics
  (quantile bins from the baseline) and categoricals (level frequencies).
  Conventional bands: < 0.10 stable, 0.10-0.25 moderate, >= 0.25 major.
  PSI has no p-value; it complements the tests, not replaces them.

A feature is flagged as DRIFTED only if the test is significant AND
PSI >= psi_warning: statistical detectability plus material magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.models.model_factory import Y_PROB
from src.utils.config import DetectionConfig

_EPS = 1e-6


@dataclass(frozen=True)
class FeatureDriftResult:
    feature: str
    kind: str            # "numeric" | "categorical" | "prediction"
    test: str            # "ks" | "chi_square"
    statistic: float
    p_value: float
    psi: float
    psi_band: str        # "stable" | "moderate" | "major"
    drifted: bool
    baseline_summary: str
    candidate_summary: str


@dataclass(frozen=True)
class DriftReport:
    features: tuple[FeatureDriftResult, ...]
    prediction: FeatureDriftResult
    n_drifted_features: int
    any_feature_drift: bool
    prediction_drift: bool


def _psi_band(psi: float, config: DetectionConfig) -> str:
    if psi >= config.psi_alert:
        return "major"
    if psi >= config.psi_warning:
        return "moderate"
    return "stable"


def _psi_from_proportions(p_base: np.ndarray, p_cand: np.ndarray) -> float:
    p = np.clip(p_base, _EPS, None)
    q = np.clip(p_cand, _EPS, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum((q - p) * np.log(q / p)))


def _numeric_psi(base: np.ndarray, cand: np.ndarray, bins: int) -> float:
    """PSI over quantile bins defined on the BASELINE distribution."""
    edges = np.unique(np.quantile(base, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:  # near-constant feature; PSI undefined -> no shift
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    b_counts, _ = np.histogram(base, bins=edges)
    c_counts, _ = np.histogram(cand, bins=edges)
    return _psi_from_proportions(
        b_counts.astype(float) / max(len(base), 1),
        c_counts.astype(float) / max(len(cand), 1),
    )


def _pool_sparse_levels(
    base: pd.Series, cand: pd.Series, min_expected: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    levels = sorted(set(base.unique()) | set(cand.unique()))
    b = np.array([(base == lv).sum() for lv in levels], dtype=float)
    c = np.array([(cand == lv).sum() for lv in levels], dtype=float)
    keep = (b >= min_expected) & (c >= min_expected)
    if keep.all():
        return b, c
    pooled_b = np.append(b[keep], b[~keep].sum())
    pooled_c = np.append(c[keep], c[~keep].sum())
    if pooled_b[-1] == 0 and pooled_c[-1] == 0:
        pooled_b, pooled_c = pooled_b[:-1], pooled_c[:-1]
    return pooled_b, pooled_c


def _numeric_drift(
    feature: str, base: np.ndarray, cand: np.ndarray, config: DetectionConfig
) -> FeatureDriftResult:
    ks = stats.ks_2samp(base, cand, method="asymp")
    psi = _numeric_psi(base, cand, config.psi_bins)
    significant = ks.pvalue < config.drift_significance_level
    return FeatureDriftResult(
        feature=feature,
        kind="numeric",
        test="ks",
        statistic=float(ks.statistic),
        p_value=float(ks.pvalue),
        psi=psi,
        psi_band=_psi_band(psi, config),
        drifted=bool(significant and psi >= config.psi_warning),
        baseline_summary=f"mean={base.mean():.3f}, std={base.std():.3f}",
        candidate_summary=f"mean={cand.mean():.3f}, std={cand.std():.3f}",
    )


def _categorical_drift(
    feature: str, base: pd.Series, cand: pd.Series, config: DetectionConfig
) -> FeatureDriftResult:
    b_counts, c_counts = _pool_sparse_levels(base, cand)
    if b_counts.size < 2:
        chi2, p_value = 0.0, 1.0
    else:
        table = np.vstack([b_counts, c_counts])
        chi2, p_value, _, _ = stats.chi2_contingency(table)
    psi = _psi_from_proportions(b_counts / b_counts.sum(), c_counts / c_counts.sum())
    significant = p_value < config.drift_significance_level
    top_b = base.value_counts(normalize=True).round(2).to_dict()
    top_c = cand.value_counts(normalize=True).round(2).to_dict()
    return FeatureDriftResult(
        feature=feature,
        kind="categorical",
        test="chi_square",
        statistic=float(chi2),
        p_value=float(p_value),
        psi=psi,
        psi_band=_psi_band(psi, config),
        drifted=bool(significant and psi >= config.psi_warning),
        baseline_summary=str(top_b),
        candidate_summary=str(top_c),
    )


def detect_drift(
    eval_baseline: pd.DataFrame,
    eval_candidate: pd.DataFrame,
    feature_columns: tuple[str, ...],
    segment_columns: tuple[str, ...],
    baseline_scores: pd.DataFrame,
    candidate_scores: pd.DataFrame,
    config: DetectionConfig,
) -> DriftReport:
    """Compare feature and prediction distributions across the two windows."""
    results: list[FeatureDriftResult] = []
    for col in feature_columns:
        results.append(
            _numeric_drift(
                col,
                eval_baseline[col].to_numpy(dtype=float),
                eval_candidate[col].to_numpy(dtype=float),
                config,
            )
        )
    for col in segment_columns:
        results.append(
            _categorical_drift(col, eval_baseline[col], eval_candidate[col], config)
        )

    pred = _numeric_drift(
        "prediction_score",
        baseline_scores[Y_PROB].to_numpy(dtype=float),
        candidate_scores[Y_PROB].to_numpy(dtype=float),
        config,
    )
    pred = FeatureDriftResult(**{**pred.__dict__, "kind": "prediction"})

    n_drifted = sum(r.drifted for r in results)
    return DriftReport(
        features=tuple(results),
        prediction=pred,
        n_drifted_features=n_drifted,
        any_feature_drift=n_drifted > 0,
        prediction_drift=pred.drifted,
    )
