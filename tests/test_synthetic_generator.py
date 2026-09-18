"""Generator behaviour: determinism, scoping, validation."""

import pytest

from src.data.synthetic_generator import (
    TARGET,
    GeneratorConfig,
    Scenario,
    generate,
)


def test_same_config_is_deterministic():
    a = generate(GeneratorConfig(scenario=Scenario.ACCURACY_REGRESSION))
    b = generate(GeneratorConfig(scenario=Scenario.ACCURACY_REGRESSION))
    assert a.candidate_train.equals(b.candidate_train)
    assert a.eval_candidate.equals(b.eval_candidate)


def test_different_seeds_differ():
    a = generate(GeneratorConfig(seed=1))
    b = generate(GeneratorConfig(seed=2))
    assert not a.eval_baseline.equals(b.eval_baseline)


def test_shapes_and_columns():
    cfg = GeneratorConfig(n_train=500, n_eval=300, n_features=6)
    b = generate(cfg)
    assert b.baseline_train.shape[0] == 500
    assert b.eval_candidate.shape[0] == 300
    for col in (*b.feature_columns, *b.segment_columns, TARGET):
        assert col in b.eval_baseline.columns
    assert len(b.feature_columns) == 6


def test_segment_corruption_is_scoped():
    corrupted = generate(GeneratorConfig(scenario=Scenario.SEGMENT_REGRESSION))
    clean = generate(GeneratorConfig())
    col, level = corrupted.config.affected_segment
    mask = (corrupted.candidate_train[col] == level).to_numpy()
    diff = (
        corrupted.candidate_train[TARGET].to_numpy()
        != clean.candidate_train[TARGET].to_numpy()
    )
    assert diff[mask].sum() > 0
    assert diff[~mask].sum() == 0


def test_segment_corruption_is_directional():
    corrupted = generate(GeneratorConfig(scenario=Scenario.SEGMENT_REGRESSION))
    clean = generate(GeneratorConfig())
    flipped = clean.candidate_train[TARGET] != corrupted.candidate_train[TARGET]
    assert (clean.candidate_train.loc[flipped, TARGET] == 1).all()


def test_feature_drift_shifts_only_candidate_window():
    b = generate(GeneratorConfig(scenario=Scenario.FEATURE_DRIFT))
    clean = generate(GeneratorConfig())
    assert b.eval_baseline[list(b.feature_columns)].equals(
        clean.eval_baseline[list(clean.feature_columns)]
    )
    shifted = [
        c
        for c in b.feature_columns
        if abs(b.eval_candidate[c].mean() - clean.eval_candidate[c].mean()) > 0.5
    ]
    assert len(shifted) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_features": 1},
        {"n_informative": 0},
        {"class_balance": 0.99},
        {"label_noise": 1.0},
        {"regression_magnitude": 0.6},
        {"affected_segment": ("region", "Mars")},
        {"n_train": 100},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        GeneratorConfig(**kwargs)


def test_positive_rate_reasonable():
    b = generate(GeneratorConfig())
    rate = b.eval_baseline[TARGET].mean()
    assert 0.25 < rate < 0.55
