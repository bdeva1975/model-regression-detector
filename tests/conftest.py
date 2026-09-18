"""Shared fixtures. Session-scoped: scenario pipelines are expensive."""

from __future__ import annotations

import pytest

from src.data.synthetic_generator import GeneratorConfig, Scenario, TaskType, generate
from src.detection.regression_detector import assess
from src.models.model_factory import train_pair
from src.utils.config import DetectionConfig

FAST = DetectionConfig(bootstrap_iterations=200)


@pytest.fixture(scope="session")
def fast_config() -> DetectionConfig:
    return FAST


def _run(scenario: Scenario):
    bundle = generate(GeneratorConfig(scenario=scenario))
    pair = train_pair(bundle)
    return bundle, pair, assess(bundle, pair, FAST)


@pytest.fixture(scope="session")
def healthy_run():
    return _run(Scenario.HEALTHY)


@pytest.fixture(scope="session")
def recall_run():
    return _run(Scenario.RECALL_REGRESSION)


@pytest.fixture(scope="session")
def segment_run():
    return _run(Scenario.SEGMENT_REGRESSION)


@pytest.fixture(scope="session")
def drift_run():
    return _run(Scenario.FEATURE_DRIFT)


@pytest.fixture(scope="session")
def false_alarm_run():
    return _run(Scenario.FALSE_ALARM)

def _run_reg(scenario: Scenario):
    bundle = generate(GeneratorConfig(task=TaskType.REGRESSION, scenario=scenario))
    pair = train_pair(bundle)
    return bundle, pair, assess(bundle, pair, FAST)


@pytest.fixture(scope="session")
def reg_healthy_run():
    return _run_reg(Scenario.HEALTHY)


@pytest.fixture(scope="session")
def reg_error_run():
    return _run_reg(Scenario.ACCURACY_REGRESSION)


@pytest.fixture(scope="session")
def reg_segment_run():
    return _run_reg(Scenario.SEGMENT_REGRESSION)
