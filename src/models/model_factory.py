"""Model construction, training, and scoring.

The factory trains a *baseline* model on clean training data and a
*candidate* model on the (possibly corrupted) candidate training data, then
scores each on its own evaluation window. Downstream stages never touch
estimators - they consume the scored frames produced here.

Feature handling: numeric features are standardised; segment columns
(region, customer_type, risk_band) are one-hot encoded and fed to the model
as features. This mirrors real production models and is what makes
segment-level regression mechanically possible: a segment-blind model
cannot learn segment-specific (mis)behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.synthetic_generator import TARGET, ScenarioBundle

Y_TRUE = "y_true"
Y_PROB = "y_prob"
Y_PRED = "y_pred"

MODEL_TYPES = ("logistic_regression", "random_forest", "gradient_boosting")


@dataclass(frozen=True)
class TrainedModel:
    """A fitted classifier plus the metadata the dashboard displays."""

    name: str            # "baseline" | "candidate"
    version: str         # display label, e.g. "v1.4"
    model_type: str
    estimator: BaseEstimator
    numeric_columns: tuple[str, ...]
    segment_columns: tuple[str, ...]

    @property
    def input_columns(self) -> list[str]:
        return [*self.numeric_columns, *self.segment_columns]

    def score_frame(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Score a dataset, returning truth, probabilities, predictions, segments."""
        prob = self.estimator.predict_proba(df.loc[:, self.input_columns])[:, 1]
        out = df.drop(columns=list(self.numeric_columns)).copy()
        out = out.rename(columns={TARGET: Y_TRUE})
        out[Y_PROB] = prob
        out[Y_PRED] = (prob >= threshold).astype(int)
        return out


def build_estimator(
    model_type: str,
    numeric_columns: tuple[str, ...],
    segment_columns: tuple[str, ...],
    seed: int,
) -> Pipeline:
    """Construct an unfitted preprocessing + classifier pipeline."""
    if model_type == "logistic_regression":
        clf: BaseEstimator = LogisticRegression(max_iter=2000, random_state=seed)
    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=8, n_jobs=-1, random_state=seed
        )
    elif model_type == "gradient_boosting":
        clf = GradientBoostingClassifier(random_state=seed)
    else:
        raise ValueError(f"unknown model_type {model_type!r}; choose from {MODEL_TYPES}")
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), list(numeric_columns)),
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(segment_columns)),
        ]
    )
    return Pipeline([("preprocess", pre), ("clf", clf)])


def _fit(
    name: str,
    version: str,
    model_type: str,
    train_df: pd.DataFrame,
    numeric_columns: tuple[str, ...],
    segment_columns: tuple[str, ...],
    seed: int,
) -> TrainedModel:
    est = build_estimator(model_type, numeric_columns, segment_columns, seed)
    est.fit(train_df.loc[:, [*numeric_columns, *segment_columns]], train_df[TARGET])
    return TrainedModel(
        name=name,
        version=version,
        model_type=model_type,
        estimator=est,
        numeric_columns=numeric_columns,
        segment_columns=segment_columns,
    )


@dataclass(frozen=True)
class TrainedPair:
    """Both fitted models plus their scored evaluation windows."""

    baseline: TrainedModel
    candidate: TrainedModel
    baseline_scores: pd.DataFrame    # scored eval_baseline
    candidate_scores: pd.DataFrame   # scored eval_candidate


def train_pair(
    bundle: ScenarioBundle,
    model_type: str = "logistic_regression",
    baseline_version: str = "v1.4",
    candidate_version: str = "v1.5",
) -> TrainedPair:
    """Train baseline and candidate on their own data; score their own windows."""
    seed = bundle.config.seed
    baseline = _fit(
        "baseline", baseline_version, model_type,
        bundle.baseline_train, bundle.feature_columns, bundle.segment_columns, seed,
    )
    candidate = _fit(
        "candidate", candidate_version, model_type,
        bundle.candidate_train, bundle.feature_columns, bundle.segment_columns, seed,
    )
    return TrainedPair(
        baseline=baseline,
        candidate=candidate,
        baseline_scores=baseline.score_frame(bundle.eval_baseline),
        candidate_scores=candidate.score_frame(bundle.eval_candidate),
    )
