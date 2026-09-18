"""Deterministic synthetic-scenario generator.

Produces train/eval datasets for a *baseline* and a *candidate* model under
controlled scenarios (healthy, various regressions, drift, false alarms).

Design principles:

* Regressions are induced by corrupting the CANDIDATE'S TRAINING LABELS, so
  the candidate model genuinely learns worse behaviour. Predictions are never
  tampered with after the fact.
* Drift scenarios shift EVALUATION FEATURES while regenerating labels from
  the same ground-truth function, so drift occurs WITHOUT performance loss:
  - feature_drift shifts non-informative features (data drift, no effect),
  - prediction_drift shifts the most informative feature (score distribution
    moves, quality holds).
* Everything is driven by one ``numpy.random.Generator`` seeded from the
  config: the same config always regenerates identical data.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np
import pandas as pd

TARGET = "target"

SEGMENT_LEVELS: dict[str, tuple[str, ...]] = {
    "region": ("North", "South", "East", "West"),
    "customer_type": ("retail", "corporate"),
    "risk_band": ("low", "medium", "high"),
}
SEGMENT_PROBS: dict[str, tuple[float, ...]] = {
    "region": (0.30, 0.30, 0.20, 0.20),
    "customer_type": (0.65, 0.35),
    "risk_band": (0.50, 0.35, 0.15),
}


class Scenario(StrEnum):
    """Named, reproducible production situations."""

    HEALTHY = "healthy"
    ACCURACY_REGRESSION = "accuracy_regression"
    PRECISION_REGRESSION = "precision_regression"
    RECALL_REGRESSION = "recall_regression"
    SEGMENT_REGRESSION = "segment_regression"
    FEATURE_DRIFT = "feature_drift"
    PREDICTION_DRIFT = "prediction_drift"
    FALSE_ALARM = "false_alarm"
    STAT_SIG_SMALL = "statistically_significant_but_practically_small"


@dataclass(frozen=True)
class GeneratorConfig:
    """Controls one synthetic scenario. Same config -> identical data."""

    scenario: Scenario = Scenario.HEALTHY
    n_train: int = 4000
    n_eval: int = 2000
    n_features: int = 8
    n_informative: int = 5
    class_balance: float = 0.35     # target positive rate
    label_noise: float = 0.15       # 0 = clean labels, 1 = pure noise
    regression_magnitude: float = 0.25   # fraction of candidate train labels corrupted
    drift_magnitude: float = 1.25   # mean shift, in feature std devs
    affected_segment: tuple[str, str] = ("risk_band", "high")
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_features < 2 or not 1 <= self.n_informative <= self.n_features:
            raise ValueError("need n_features >= 2 and 1 <= n_informative <= n_features")
        if not 0.05 <= self.class_balance <= 0.95:
            raise ValueError("class_balance must be in [0.05, 0.95]")
        if not 0.0 <= self.label_noise < 1.0:
            raise ValueError("label_noise must be in [0, 1)")
        if not 0.0 <= self.regression_magnitude <= 0.49:
            raise ValueError("regression_magnitude must be in [0, 0.49]")
        if self.drift_magnitude < 0:
            raise ValueError("drift_magnitude must be >= 0")
        col, level = self.affected_segment
        if col not in SEGMENT_LEVELS or level not in SEGMENT_LEVELS[col]:
            raise ValueError(f"affected_segment {self.affected_segment} is not a known segment")
        if min(self.n_train, self.n_eval) < 200:
            raise ValueError("n_train and n_eval must each be >= 200")


@dataclass(frozen=True)
class ScenarioBundle:
    """Everything downstream stages need for one scenario."""

    scenario: Scenario
    config: GeneratorConfig
    baseline_train: pd.DataFrame
    candidate_train: pd.DataFrame
    eval_baseline: pd.DataFrame     # evaluation window for the baseline model
    eval_candidate: pd.DataFrame    # evaluation window for the candidate model
    feature_columns: tuple[str, ...]
    segment_columns: tuple[str, ...]
    notes: str                      # ground-truth mechanism, for docs and tests


# --------------------------------------------------------------------------
# Ground-truth world
# --------------------------------------------------------------------------

def _make_weights(cfg: GeneratorConfig, rng: np.random.Generator) -> np.ndarray:
    """Informative features get real weights; the rest get exactly zero."""
    w = np.zeros(cfg.n_features)
    w[: cfg.n_informative] = rng.normal(loc=0.0, scale=1.0, size=cfg.n_informative)
    # Guarantee a clearly dominant feature so prediction_drift has a lever.
    w[0] = np.sign(w[0] or 1.0) * max(abs(w[0]), 1.5)
    return w


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _sample_features(cfg: GeneratorConfig, rng: np.random.Generator, n: int) -> pd.DataFrame:
    x = rng.normal(size=(n, cfg.n_features))
    cols = [f"f{i}" for i in range(cfg.n_features)]
    return pd.DataFrame(x, columns=cols)


def _sample_segments(rng: np.random.Generator, n: int) -> pd.DataFrame:
    data = {
        col: rng.choice(SEGMENT_LEVELS[col], size=n, p=SEGMENT_PROBS[col])
        for col in SEGMENT_LEVELS
    }
    return pd.DataFrame(data)


def _true_labels(
    cfg: GeneratorConfig,
    rng: np.random.Generator,
    features: pd.DataFrame,
    weights: np.ndarray,
    intercept: float,
) -> np.ndarray:
    logits = features.to_numpy() @ weights * (1.0 - cfg.label_noise) + intercept
    return (rng.random(len(features)) < _sigmoid(logits)).astype(np.int64)


def _make_split(
    cfg: GeneratorConfig,
    rng: np.random.Generator,
    n: int,
    weights: np.ndarray,
    intercept: float,
) -> pd.DataFrame:
    features = _sample_features(cfg, rng, n)
    segments = _sample_segments(rng, n)
    y = _true_labels(cfg, rng, features, weights, intercept)
    out = pd.concat([features, segments], axis=1)
    out[TARGET] = y
    return out


# --------------------------------------------------------------------------
# Corruption and drift mechanisms
# --------------------------------------------------------------------------

def _flip_labels(
    df: pd.DataFrame,
    rng: np.random.Generator,
    fraction: float,
    direction: str,
    mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return a copy with a fraction of eligible labels flipped.

    direction: "both" | "pos_to_neg" | "neg_to_pos".
    mask: optional boolean row filter (e.g. one segment).
    """
    out = df.copy()
    y = out[TARGET].to_numpy().copy()
    eligible = np.ones(len(out), dtype=bool) if mask is None else mask.copy()
    if direction == "pos_to_neg":
        eligible &= y == 1
    elif direction == "neg_to_pos":
        eligible &= y == 0
    elif direction != "both":
        raise ValueError(f"unknown flip direction: {direction}")
    idx = np.flatnonzero(eligible)
    n_flip = int(round(fraction * len(idx)))
    if n_flip > 0:
        chosen = rng.choice(idx, size=n_flip, replace=False)
        y[chosen] = 1 - y[chosen]
    out[TARGET] = y
    return out


def _shift_features(
    df: pd.DataFrame,
    cfg: GeneratorConfig,
    rng: np.random.Generator,
    columns: list[str],
    weights: np.ndarray,
    intercept: float,
    relabel: bool,
) -> pd.DataFrame:
    """Return a copy with the given feature columns mean-shifted.

    If relabel is True, targets are regenerated from the ground-truth
    function on the shifted features - drift WITHOUT regression.
    """
    out = df.copy()
    for col in columns:
        out[col] = out[col] + cfg.drift_magnitude
    if relabel:
        feature_cols = [f"f{i}" for i in range(cfg.n_features)]
        out[TARGET] = _true_labels(cfg, rng, out[feature_cols], weights, intercept)
    return out


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def generate(cfg: GeneratorConfig) -> ScenarioBundle:
    """Build the full bundle for one scenario, deterministically."""
    eff = cfg
    if cfg.scenario is Scenario.STAT_SIG_SMALL:
        # Large EVAL windows only: power to detect a tiny, real effect.
        # (Scaling n_train would let the model average the noise away.)
        eff = replace(cfg, n_eval=max(cfg.n_eval, 20_000))

    rng = np.random.default_rng(eff.seed)
    weights = _make_weights(eff, rng)
    intercept = float(np.log(eff.class_balance / (1.0 - eff.class_balance)))

    baseline_train = _make_split(eff, rng, eff.n_train, weights, intercept)
    candidate_train = _make_split(eff, rng, eff.n_train, weights, intercept)
    eval_baseline = _make_split(eff, rng, eff.n_eval, weights, intercept)
    eval_candidate = _make_split(eff, rng, eff.n_eval, weights, intercept)

    feature_cols = [f"f{i}" for i in range(eff.n_features)]
    informative = feature_cols[: eff.n_informative]
    non_informative = feature_cols[eff.n_informative :] or feature_cols[-2:]

    s = eff.scenario
    if s is Scenario.HEALTHY:
        notes = "Candidate trained on clean data from the same distribution. Expect: no regression."
    elif s is Scenario.ACCURACY_REGRESSION:
        candidate_train = _flip_labels(candidate_train, rng, eff.regression_magnitude, "both")
        notes = (
            f"{eff.regression_magnitude:.0%} of candidate training labels flipped in both "
            "directions. Expect: broad metric regression (accuracy, F1, AUC, log loss)."
        )
    elif s is Scenario.RECALL_REGRESSION:
        frac = min(1.8 * eff.regression_magnitude, 0.49)
        candidate_train = _flip_labels(candidate_train, rng, frac, "pos_to_neg")
        notes = (
            f"{frac:.0%} of candidate training POSITIVES relabelled negative. "
            "Expect: recall drops sharply; precision holds or rises; accuracy may look fine."
        )
    elif s is Scenario.PRECISION_REGRESSION:
        frac = min(1.2 * eff.regression_magnitude, 0.49)
        candidate_train = _flip_labels(candidate_train, rng, frac, "neg_to_pos")
        notes = (
            f"{frac:.0%} of candidate training NEGATIVES relabelled positive. "
            "Expect: precision drops; recall holds or rises."
        )
    elif s is Scenario.SEGMENT_REGRESSION:
        col, level = eff.affected_segment
        mask = (candidate_train[col] == level).to_numpy()
        frac = min(3.0 * eff.regression_magnitude, 0.85)
        candidate_train = _flip_labels(candidate_train, rng, frac, "pos_to_neg", mask=mask)
        notes = (
            f"{frac:.0%} of candidate training POSITIVES relabelled negative ONLY where "
            f"{col} == '{level}'. Expect: overall metrics near-healthy; that segment's "
            "recall and accuracy degrade materially."
        )
    elif s is Scenario.FEATURE_DRIFT:
        eval_candidate = _shift_features(
            eval_candidate, eff, rng, non_informative[:2], weights, intercept, relabel=True
        )
        notes = (
            f"Non-informative features {non_informative[:2]} mean-shifted by "
            f"{eff.drift_magnitude} std in the candidate window; labels regenerated from "
            "ground truth. Expect: feature drift flagged, NO performance regression."
        )
    elif s is Scenario.PREDICTION_DRIFT:
        eval_candidate = _shift_features(
            eval_candidate, eff, rng, [informative[0]], weights, intercept, relabel=True
        )
        notes = (
            f"Most informative feature {informative[0]} mean-shifted by "
            f"{eff.drift_magnitude} std with labels regenerated. Expect: prediction "
            "distribution shifts, quality roughly holds - a warning, not a regression."
        )
    elif s is Scenario.FALSE_ALARM:
        candidate_train = _flip_labels(candidate_train, rng, 0.02, "both")
        notes = (
            "2% label noise in candidate training. Expect: tiny metric wobble below both "
            "practical and statistical thresholds - NO material regression."
        )
    elif s is Scenario.STAT_SIG_SMALL:
        candidate_train = _flip_labels(candidate_train, rng, 0.04, "pos_to_neg")
        notes = (
            "4% of candidate training positives relabelled negative, evaluated on "
            "~20k-sample windows. Expect: a small, real recall dip - statistically "
            "significant at this sample size, but below the practical threshold. "
            "The system must report it as negligible, not as a regression."
        )
    else:  # pragma: no cover - enum is exhaustive
        raise ValueError(f"unhandled scenario: {s}")

    return ScenarioBundle(
        scenario=s,
        config=eff,
        baseline_train=baseline_train,
        candidate_train=candidate_train,
        eval_baseline=eval_baseline,
        eval_candidate=eval_candidate,
        feature_columns=tuple(feature_cols),
        segment_columns=tuple(SEGMENT_LEVELS),
        notes=notes,
    )
