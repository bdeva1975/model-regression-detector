"""Regression-task behaviour: generator, metrics, pipeline, explanation."""

import math

import pytest

from src.analysis.explanation import explain
from src.data.synthetic_generator import (
    CLASSIFICATION_ONLY,
    TARGET,
    GeneratorConfig,
    Scenario,
    TaskType,
    generate,
)
from src.detection.regression_detector import Severity
from src.evaluation.metrics import compute_metrics, infer_task
from src.models.model_factory import Y_PROB


def test_regression_target_is_continuous():
    b = generate(GeneratorConfig(task=TaskType.REGRESSION))
    y = b.eval_baseline[TARGET]
    assert y.dtype.kind == "f"
    assert y.nunique() > 100


def test_regression_is_deterministic():
    a = generate(GeneratorConfig(task=TaskType.REGRESSION, scenario=Scenario.ACCURACY_REGRESSION))
    b = generate(GeneratorConfig(task=TaskType.REGRESSION, scenario=Scenario.ACCURACY_REGRESSION))
    assert a.candidate_train.equals(b.candidate_train)


@pytest.mark.parametrize("scenario", sorted(CLASSIFICATION_ONLY, key=lambda s: s.value))
def test_classification_only_scenarios_rejected(scenario):
    with pytest.raises(ValueError, match="no regression-task analogue"):
        GeneratorConfig(task=TaskType.REGRESSION, scenario=scenario)


def test_scored_frame_has_no_prob_column(reg_healthy_run):
    _, pair, _ = reg_healthy_run
    assert Y_PROB not in pair.candidate_scores.columns
    assert infer_task(pair.candidate_scores) is TaskType.REGRESSION


def test_regression_metrics_computed(reg_healthy_run):
    _, pair, _ = reg_healthy_run
    r = compute_metrics(pair.baseline_scores)
    assert set(r.values) == {"mae", "rmse", "r2"}
    assert r.confusion is None
    assert math.isnan(r.positive_rate)
    assert r.values["rmse"] >= r.values["mae"] > 0
    assert 0.9 < r.values["r2"] <= 1.0


def test_reg_healthy_no_regression(reg_healthy_run):
    _, _, a = reg_healthy_run
    assert not a.regression_detected
    assert a.severity is Severity.NONE


def test_reg_error_regression_detected(reg_error_run):
    _, _, a = reg_error_run
    assert a.regression_detected
    assert {"mae", "rmse"} <= set(a.affected_metrics)
    assert a.severity in (Severity.HIGH, Severity.CRITICAL)


def test_reg_segment_regression_localised(reg_segment_run):
    bundle, _, a = reg_segment_run
    col, level = bundle.config.affected_segment
    assert a.regression_detected
    assert a.affected_segments == (f"{col}={level}",)
    assert a.affected_metrics == ()          # overall metrics hold
    assert a.severity is Severity.MODERATE


def test_reg_segment_explanation_reports_mae_and_bias(reg_segment_run):
    _, _, a = reg_segment_run
    text = explain(a)
    assert "MAE" in text and "bias" in text
    assert "risk_band=high" in text


def test_reg_orientation_error_metrics_degrade_upward(reg_error_run):
    _, _, a = reg_error_run
    mae = next(m for m in a.metrics if m.metric == "mae")
    assert not mae.higher_is_better
    assert mae.absolute_change > 0           # raw change positive
    assert mae.degradation_abs > 0           # counted as degradation
    r2 = next(m for m in a.metrics if m.metric == "r2")
    assert r2.higher_is_better
    assert r2.absolute_change < 0
