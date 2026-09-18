"""The regression decision engine.

Combines per-metric comparison, bootstrap significance, drift analysis and
segment analysis into a single structured ``RegressionAssessment``.

Decision rule (per metric)
--------------------------
A metric counts as REGRESSED only if ALL hold, in its own orientation
(log loss degrades upward, everything else downward):

1. degradation >= config.min_absolute_degradation (absolute), AND
2. degradation >= config.min_relative_degradation (relative to baseline), AND
3. the bootstrap difference is statistically significant at
   config.significance_level.

Practical-but-not-statistical -> "inconclusive" (more data needed).
Statistical-but-not-practical -> "negligible" (explicitly called out; this
is the STAT_SIG_SMALL scenario's expected outcome).

Overall verdict
---------------
REGRESSION if any metric regressed OR any segment degraded. Severity is
graded on the worst relative metric degradation (config.severity_*); a
segment-only regression is at least MODERATE. Drift never triggers a
regression verdict by itself - it is reported as supporting context.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from src.analysis.segment_analysis import SegmentReport, analyze_segments
from src.data.synthetic_generator import ScenarioBundle
from src.detection.drift_detector import DriftReport, detect_drift
from src.detection.statistical_tests import BootstrapResult, bootstrap_metric_diff
from src.evaluation.metrics import (
    CLASSIFICATION_METRICS,
    METRIC_ORIENTATION,
)
from src.models.model_factory import TrainedPair
from src.utils.config import DetectionConfig


class MetricStatus(StrEnum):
    OK = "ok"
    IMPROVED = "improved"
    NEGLIGIBLE = "negligible"          # statistically significant, practically small
    INCONCLUSIVE = "inconclusive"      # practically large, not statistically confirmed
    REGRESSED = "regressed"


class Severity(StrEnum):
    NONE = "NONE"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class MetricAssessment:
    metric: str
    higher_is_better: bool
    baseline_value: float
    candidate_value: float
    absolute_change: float          # candidate - baseline (raw)
    degradation_abs: float          # >= 0, in the degrading direction
    degradation_rel: float          # fraction of |baseline|
    practically_significant: bool
    statistically_significant: bool
    p_value: float
    ci_low: float
    ci_high: float
    status: MetricStatus


@dataclass(frozen=True)
class RegressionAssessment:
    regression_detected: bool
    severity: Severity
    metrics: tuple[MetricAssessment, ...]
    affected_metrics: tuple[str, ...]
    largest_degradation: str | None      # metric name, by relative degradation
    drift: DriftReport
    segments: SegmentReport
    affected_segments: tuple[str, ...]
    config: DetectionConfig

    def to_dict(self) -> dict:
        """JSON-ready summary (the schema shown in the README)."""
        worst = self.largest_degradation
        worst_rel = next(
            (m.degradation_rel for m in self.metrics if m.metric == worst), 0.0
        )
        return {
            "regression_detected": self.regression_detected,
            "severity": self.severity.value,
            "affected_metrics": list(self.affected_metrics),
            "largest_degradation": worst,
            "relative_change_pct": round(-100.0 * worst_rel, 1) if worst else 0.0,
            "statistically_significant": any(
                m.statistically_significant for m in self.metrics
                if m.metric in self.affected_metrics
            ),
            "practically_significant": bool(self.affected_metrics),
            "affected_segments": list(self.affected_segments),
            "drifted_features": [f.feature for f in self.drift.features if f.drifted],
            "prediction_drift": self.drift.prediction_drift,
        }


def _assess_metric(boot: BootstrapResult, config: DetectionConfig) -> MetricAssessment:
    higher_is_better = METRIC_ORIENTATION[boot.metric]
    change = boot.diff
    degradation = -change if higher_is_better else change   # positive = worse
    deg_abs = max(degradation, 0.0)
    base = abs(boot.baseline_value)
    deg_rel = deg_abs / base if base > 0 else (float("inf") if deg_abs > 0 else 0.0)

    practically = (
        deg_abs >= config.min_absolute_degradation
        and deg_rel >= config.min_relative_degradation
    )
    statistically = boot.significant

    if degradation <= 0:
        status = MetricStatus.IMPROVED if statistically else MetricStatus.OK
    elif practically and statistically:
        status = MetricStatus.REGRESSED
    elif statistically:
        status = MetricStatus.NEGLIGIBLE
    elif practically:
        status = MetricStatus.INCONCLUSIVE
    else:
        status = MetricStatus.OK

    return MetricAssessment(
        metric=boot.metric,
        higher_is_better=higher_is_better,
        baseline_value=boot.baseline_value,
        candidate_value=boot.candidate_value,
        absolute_change=change,
        degradation_abs=deg_abs,
        degradation_rel=deg_rel,
        practically_significant=practically,
        statistically_significant=statistically,
        p_value=boot.p_value,
        ci_low=boot.ci_low,
        ci_high=boot.ci_high,
        status=status,
    )


def _grade_severity(
    regressed: list[MetricAssessment],
    segments: SegmentReport,
    config: DetectionConfig,
) -> Severity:
    worst_rel = max((m.degradation_rel for m in regressed), default=0.0)
    if worst_rel >= config.severity_critical:
        return Severity.CRITICAL
    if worst_rel >= config.severity_high:
        return Severity.HIGH
    if worst_rel >= config.severity_moderate or regressed:
        return Severity.MODERATE
    if segments.any_segment_regression:
        return Severity.MODERATE
    return Severity.NONE


def assess(
    bundle: ScenarioBundle,
    pair: TrainedPair,
    config: DetectionConfig,
) -> RegressionAssessment:
    """Run the full comparison pipeline and return one structured verdict."""
    boots = [
        bootstrap_metric_diff(pair.baseline_scores, pair.candidate_scores, m, config)
        for m in CLASSIFICATION_METRICS
    ]
    metrics = tuple(_assess_metric(b, config) for b in boots)
    regressed = [m for m in metrics if m.status is MetricStatus.REGRESSED]

    drift = detect_drift(
        bundle.eval_baseline,
        bundle.eval_candidate,
        bundle.feature_columns,
        bundle.segment_columns,
        pair.baseline_scores,
        pair.candidate_scores,
        config,
    )
    segments = analyze_segments(pair.baseline_scores, pair.candidate_scores, config)

    detected = bool(regressed) or segments.any_segment_regression
    largest = max(regressed, key=lambda m: m.degradation_rel).metric if regressed else None

    return RegressionAssessment(
        regression_detected=detected,
        severity=_grade_severity(regressed, segments, config),
        metrics=metrics,
        affected_metrics=tuple(m.metric for m in regressed),
        largest_degradation=largest,
        drift=drift,
        segments=segments,
        affected_segments=segments.degraded_segments,
        config=config,
    )
