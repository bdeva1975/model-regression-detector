"""End-to-end pipeline behaviour per scenario, plus drift and segments."""

from src.analysis.explanation import explain
from src.detection.regression_detector import Severity


def test_healthy_no_regression(healthy_run):
    _, _, a = healthy_run
    assert not a.regression_detected
    assert a.severity is Severity.NONE
    assert a.affected_metrics == ()
    assert a.affected_segments == ()


def test_recall_regression_detected_with_recall_named(recall_run):
    _, _, a = recall_run
    assert a.regression_detected
    assert "recall" in a.affected_metrics
    assert a.severity in (Severity.HIGH, Severity.CRITICAL)
    worst = next(m for m in a.metrics if m.metric == "recall")
    assert worst.statistically_significant and worst.practically_significant


def test_segment_regression_localised(segment_run):
    bundle, _, a = segment_run
    col, level = bundle.config.affected_segment
    assert a.regression_detected
    assert f"{col}={level}" in a.affected_segments
    assert len(a.affected_metrics) <= 3  # overall metrics mostly hold


def test_feature_drift_is_not_regression(drift_run):
    _, _, a = drift_run
    assert not a.regression_detected
    drifted = [f.feature for f in a.drift.features if f.drifted]
    assert len(drifted) == 2
    assert not a.drift.prediction_drift


def test_false_alarm_stays_quiet(false_alarm_run):
    _, _, a = false_alarm_run
    assert not a.regression_detected
    assert a.severity is Severity.NONE


def test_thresholds_change_the_verdict(recall_run, fast_config):
    """Same evidence, stricter practical threshold -> fewer affected metrics."""
    from src.detection.regression_detector import assess

    bundle, pair, _ = recall_run
    strict = fast_config.with_overrides(min_relative_degradation=0.80)
    a = assess(bundle, pair, strict)
    assert "recall" not in a.affected_metrics


def test_to_dict_schema(recall_run):
    _, _, a = recall_run
    d = a.to_dict()
    for key in (
        "regression_detected",
        "severity",
        "affected_metrics",
        "largest_degradation",
        "affected_segments",
        "drifted_features",
        "prediction_drift",
    ):
        assert key in d


def test_explanation_mentions_the_evidence(segment_run):
    bundle, _, a = segment_run
    text = explain(a)
    col, level = bundle.config.affected_segment
    assert "REGRESSION DETECTED" in text
    assert f"{col}={level}" in text
    assert "Recommended investigation" in text


def test_explanation_healthy(healthy_run):
    _, _, a = healthy_run
    assert "NO MATERIAL REGRESSION" in explain(a)
