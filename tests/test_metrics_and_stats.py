"""Metric computation and bootstrap statistics."""

import math

import numpy as np
import pandas as pd
import pytest

from src.detection.statistical_tests import bootstrap_metric_diff
from src.evaluation.metrics import compute_metrics
from src.models.model_factory import Y_PRED, Y_PROB, Y_TRUE
from src.utils.config import DetectionConfig


def _frame(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    return pd.DataFrame(
        {Y_TRUE: y_true, Y_PROB: y_prob, Y_PRED: (y_prob >= threshold).astype(int)}
    )


def test_perfect_classifier():
    f = _frame([0, 0, 1, 1] * 50, [0.1, 0.2, 0.8, 0.9] * 50)
    r = compute_metrics(f)
    assert r.values["accuracy"] == 1.0
    assert r.values["recall"] == 1.0
    assert r.values["roc_auc"] == 1.0
    assert r.confusion == ((100, 0), (0, 100))


def test_single_class_yields_nan_for_ranking_metrics():
    f = _frame([1] * 120, np.linspace(0.4, 0.9, 120))
    r = compute_metrics(f)
    assert math.isnan(r.values["roc_auc"])
    assert math.isnan(r.values["log_loss"])
    assert r.values["recall"] >= 0.0


def test_empty_frame_rejected():
    with pytest.raises(ValueError):
        compute_metrics(_frame([], []))


def test_bootstrap_detects_real_difference(fast_config):
    rng = np.random.default_rng(0)
    n = 1500
    y = rng.integers(0, 2, n)
    # good: mostly right; bad: noisy scores that misclassify a real fraction
    good = np.clip(0.6 * y + 0.2 + 0.15 * rng.standard_normal(n), 0, 1)
    bad = np.clip(0.25 * y + 0.38 + 0.30 * rng.standard_normal(n), 0, 1)
    r = bootstrap_metric_diff(_frame(y, good), _frame(y, bad), "accuracy", fast_config)
    assert r.diff < 0
    assert r.significant
    assert r.ci_high < 0


def test_bootstrap_no_difference_not_significant(fast_config):
    rng = np.random.default_rng(1)
    n = 1500
    y = rng.integers(0, 2, n)
    probs = np.clip(0.7 * y + 0.3 * rng.random(n), 0, 1)
    r = bootstrap_metric_diff(_frame(y, probs), _frame(y, probs), "f1", fast_config)
    assert not r.significant
    assert r.ci_low <= 0 <= r.ci_high


def test_bootstrap_rejects_small_samples(fast_config):
    y = [0, 1] * 20
    p = [0.3, 0.8] * 20
    with pytest.raises(ValueError, match="min_sample_size"):
        bootstrap_metric_diff(_frame(y, p), _frame(y, p), "accuracy", fast_config)


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        DetectionConfig(significance_level=0.0)
    with pytest.raises(ValueError):
        DetectionConfig(severity_moderate=0.3, severity_high=0.1)
    with pytest.raises(ValueError):
        DetectionConfig(bootstrap_iterations=10)
