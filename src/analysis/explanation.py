"""Deterministic, human-readable explanation of a RegressionAssessment.

No LLM, no templates library - plain structured prose generated from the
evidence, answering: what happened, how bad, what supports it, what to
investigate next. The same assessment always yields the same text.
"""

from __future__ import annotations

from src.detection.regression_detector import (
    MetricStatus,
    RegressionAssessment,
)


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _metric_lines(a: RegressionAssessment) -> list[str]:
    lines: list[str] = []
    for m in a.metrics:
        if m.status is MetricStatus.REGRESSED:
            direction = "decreased" if m.higher_is_better else "increased"
            lines.append(
                f"- {m.metric} {direction} from {m.baseline_value:.3f} to "
                f"{m.candidate_value:.3f} ({_fmt_pct(m.degradation_rel)} relative "
                f"degradation; p={m.p_value:.4f}; "
                f"95% CI for the change [{m.ci_low:.3f}, {m.ci_high:.3f}])."
            )
    return lines


def _negligible_lines(a: RegressionAssessment) -> list[str]:
    lines: list[str] = []
    for m in a.metrics:
        if m.status is MetricStatus.NEGLIGIBLE:
            lines.append(
                f"- {m.metric} changed by {_fmt_pct(m.degradation_rel)} relative "
                f"(p={m.p_value:.4f}): statistically significant at this sample size, "
                f"but below the practical threshold of "
                f"{_fmt_pct(a.config.min_relative_degradation)} - reported, not alerted."
            )
    return lines


def _inconclusive_lines(a: RegressionAssessment) -> list[str]:
    return [
        f"- {m.metric} shows a {_fmt_pct(m.degradation_rel)} relative degradation that "
        f"is not statistically confirmed (p={m.p_value:.4f}); more evaluation data "
        "would settle it."
        for m in a.metrics
        if m.status is MetricStatus.INCONCLUSIVE
    ]


def _segment_lines(a: RegressionAssessment) -> list[str]:
    lines: list[str] = []
    for r in a.segments.results:
        if r.degraded:
            lines.append(
                f"- {r.column}={r.level}: accuracy {r.baseline['accuracy']:.3f} -> "
                f"{r.candidate['accuracy']:.3f}, recall {r.baseline['recall']:.3f} -> "
                f"{r.candidate['recall']:.3f} "
                f"(adjusted p={r.p_value_adjusted:.4f}, "
                f"n={r.n_baseline}/{r.n_candidate})."
            )
    return lines


def _drift_lines(a: RegressionAssessment) -> list[str]:
    lines: list[str] = []
    for f in a.drift.features:
        if f.drifted:
            lines.append(
                f"- {f.feature} ({f.kind}): PSI={f.psi:.3f} ({f.psi_band}), "
                f"{f.test} p={f.p_value:.2e}. Baseline {f.baseline_summary}; "
                f"candidate {f.candidate_summary}."
            )
    if a.drift.prediction_drift:
        p = a.drift.prediction
        lines.append(
            f"- prediction scores: PSI={p.psi:.3f} ({p.psi_band}), ks p={p.p_value:.2e}. "
            "A shifted score distribution warns of changed model behaviour but does not "
            "by itself prove regression."
        )
    return lines


def _recommendations(a: RegressionAssessment) -> list[str]:
    recs: list[str] = []
    if a.affected_metrics:
        recs.append(
            "Compare candidate and baseline training data: label distributions, "
            "class balance, and any relabelling or pipeline changes."
        )
    if "recall" in a.affected_metrics or "precision" in a.affected_metrics:
        recs.append(
            "Inspect the decision threshold: a boundary shift (precision up, recall "
            "down, ranking metrics stable) points to calibration, not lost signal."
        )
    if a.affected_segments:
        recs.append(
            f"Audit training data for the degraded segment(s) "
            f"{', '.join(a.affected_segments)}: coverage, label quality, and any "
            "segment-correlated pipeline change."
        )
    drifted = [f.feature for f in a.drift.features if f.drifted]
    if drifted:
        recs.append(
            f"Investigate upstream sources of the drifted feature(s) "
            f"{', '.join(drifted)}; confirm whether the shift is expected "
            "(seasonality, new population) or a data-quality fault."
        )
    if a.drift.prediction_drift and not a.regression_detected:
        recs.append(
            "Track the shifted prediction distribution over the next windows; "
            "recalibrate the operating threshold if the shift persists."
        )
    if not recs:
        recs.append("No action required; continue routine monitoring.")
    return recs


def explain(a: RegressionAssessment) -> str:
    """Render the full plain-text explanation for one assessment."""
    parts: list[str] = []

    if a.regression_detected:
        head = f"REGRESSION DETECTED - severity {a.severity.value}."
        if a.largest_degradation:
            worst = next(m for m in a.metrics if m.metric == a.largest_degradation)
            head += (
                f" Largest degradation: {worst.metric} "
                f"({_fmt_pct(worst.degradation_rel)} relative)."
            )
        elif a.affected_segments:
            head += " Driven by segment-level degradation; overall metrics held."
        parts.append(head)
    else:
        parts.append("NO MATERIAL REGRESSION DETECTED.")

    metric_lines = _metric_lines(a)
    if metric_lines:
        parts.append("Degraded metrics (practically AND statistically significant):")
        parts.extend(metric_lines)

    seg_lines = _segment_lines(a)
    if seg_lines:
        parts.append("Degraded segments (Bonferroni-adjusted):")
        parts.extend(seg_lines)

    neg_lines = _negligible_lines(a)
    if neg_lines:
        parts.append("Statistically significant but practically negligible:")
        parts.extend(neg_lines)

    inc_lines = _inconclusive_lines(a)
    if inc_lines:
        parts.append("Inconclusive (practically large, statistically unconfirmed):")
        parts.extend(inc_lines)

    drift_lines = _drift_lines(a)
    if drift_lines:
        parts.append("Distribution shifts (context, not proof of regression):")
        parts.extend(drift_lines)
    elif not a.regression_detected:
        parts.append("No feature or prediction drift of note.")

    parts.append("Recommended investigation:")
    parts.extend(f"{i}. {r}" for i, r in enumerate(_recommendations(a), 1))

    thresholds = a.config
    parts.append(
        f"(Decision thresholds: >= {_fmt_pct(thresholds.min_relative_degradation)} "
        f"relative and >= {thresholds.min_absolute_degradation:.3f} absolute "
        f"degradation, alpha = {thresholds.significance_level}.)"
    )
    return "\n".join(parts)
