"""Model Regression Detection System - Streamlit dashboard.

Run with: streamlit run app.py
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analysis.explanation import explain
from src.data.synthetic_generator import (
    CLASSIFICATION_ONLY,
    GeneratorConfig,
    Scenario,
    ScenarioBundle,
    TaskType,
    generate,
)
from src.detection.regression_detector import (
    RegressionAssessment,
    assess,
)
from src.models.model_factory import MODEL_TYPES, Y_PROB, TrainedPair, train_pair
from src.utils.config import DetectionConfig

st.set_page_config(page_title="Model Regression Detector", page_icon="📉", layout="wide")

SCENARIO_LABELS: dict[Scenario, str] = {
    Scenario.HEALTHY: "Healthy deployment",
    Scenario.ACCURACY_REGRESSION: "Overall performance regression",
    Scenario.PRECISION_REGRESSION: "Precision regression",
    Scenario.RECALL_REGRESSION: "Recall regression",
    Scenario.SEGMENT_REGRESSION: "Segment regression",
    Scenario.FEATURE_DRIFT: "Feature drift (no regression)",
    Scenario.PREDICTION_DRIFT: "Prediction drift (no regression)",
    Scenario.FALSE_ALARM: "False alarm (tiny wobble)",
    Scenario.STAT_SIG_SMALL: "Statistically significant, practically small (slow)",
}


@st.cache_resource(show_spinner=False)
def run_pipeline(
    task_value: str,
    scenario_value: str,
    model_type: str,
    seed: int,
    min_abs: float,
    min_rel: float,
    alpha: float,
) -> tuple[ScenarioBundle, TrainedPair, RegressionAssessment, DetectionConfig]:
    config = DetectionConfig(
        random_seed=seed,
        min_absolute_degradation=min_abs,
        min_relative_degradation=min_rel,
        significance_level=alpha,
    )
    bundle = generate(
        GeneratorConfig(
            task=TaskType(task_value), scenario=Scenario(scenario_value), seed=seed
        )
    )
    pair = train_pair(bundle, model_type=model_type)
    assessment = assess(bundle, pair, config)
    return bundle, pair, assessment, config


# ---------------------------------------------------------------- sidebar --
st.sidebar.title("📉 Regression Detector")
task = st.sidebar.selectbox(
    "Task", list(TaskType), format_func=lambda t: t.value.capitalize()
)
scenarios = [
    s for s in SCENARIO_LABELS
    if task is TaskType.CLASSIFICATION or s not in CLASSIFICATION_ONLY
]
scenario = st.sidebar.selectbox(
    "Scenario", scenarios, format_func=lambda s: SCENARIO_LABELS[s]
)
model_type = st.sidebar.selectbox("Model type", MODEL_TYPES[task])
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=99_999, value=42)

st.sidebar.subheader("Decision thresholds")
min_rel = st.sidebar.slider("Min relative degradation", 0.01, 0.30, 0.05, 0.01)
min_abs = st.sidebar.slider("Min absolute degradation", 0.0, 0.10, 0.01, 0.005)
alpha = st.sidebar.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01)

page = st.sidebar.radio(
    "View",
    (
        "Dashboard",
        "Model comparison",
        "Drift analysis",
        "Segment analysis",
        "Explanation",
        "Scenario notes",
    ),
)

with st.spinner("Training models and running detection (cached per configuration)..."):
    bundle, pair, a, config = run_pipeline(
        task.value,
        scenario.value,
        model_type,
        int(seed),
        float(min_abs),
        float(min_rel),
        float(alpha),
    )


# --------------------------------------------------------------- helpers --
def status_badge(assessment: RegressionAssessment) -> None:
    if assessment.regression_detected:
        st.error(f"⚠ REGRESSION DETECTED - severity {assessment.severity.value}")
    else:
        st.success("✔ NO MATERIAL REGRESSION")


def metric_table(assessment: RegressionAssessment) -> pd.DataFrame:
    rows = []
    for m in assessment.metrics:
        rows.append(
            {
                "metric": m.metric,
                "baseline": round(m.baseline_value, 4),
                "candidate": round(m.candidate_value, 4),
                "change": round(m.absolute_change, 4),
                "relative degradation": f"{100 * m.degradation_rel:.1f}%",
                "p-value": round(m.p_value, 4),
                "95% CI low": round(m.ci_low, 4),
                "95% CI high": round(m.ci_high, 4),
                "status": m.status.value,
            }
        )
    return pd.DataFrame(rows)


def segment_rows(assessment: RegressionAssessment) -> list[dict]:
    rows = []
    for r in assessment.segments.results:
        if r.skipped:
            continue
        row = {
            "segment": f"{r.column}={r.level}",
            "n (base/cand)": f"{r.n_baseline}/{r.n_candidate}",
        }
        for k in r.baseline:
            row[f"{k} base"] = round(r.baseline[k], 3)
            row[f"{k} cand"] = round(r.candidate[k], 3)
        row["adj. p"] = round(r.p_value_adjusted, 4)
        row["degraded"] = r.degraded
        rows.append(row)
    return rows


# ------------------------------------------------------------------ pages --
if page == "Dashboard":
    st.title("Model health")
    status_badge(a)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Baseline model", pair.baseline.version, pair.baseline.model_type)
    c2.metric("Candidate model", pair.candidate.version, pair.candidate.model_type)
    c3.metric("Metrics degraded", f"{len(a.affected_metrics)} / {len(a.metrics)}")
    c4.metric("Drifted features", a.drift.n_drifted_features)
    c5.metric("Affected segments", len(a.affected_segments))

    st.caption(f"Scenario mechanism: {bundle.notes}")

    fig = go.Figure()
    names = [m.metric for m in a.metrics]
    fig.add_bar(name="baseline", x=names, y=[m.baseline_value for m in a.metrics])
    fig.add_bar(name="candidate", x=names, y=[m.candidate_value for m in a.metrics])
    fig.update_layout(barmode="group", title="Baseline vs candidate metrics", height=420)
    st.plotly_chart(fig, use_container_width=True)
    if bundle.task is TaskType.REGRESSION:
        st.caption(
            "Orientation differs by metric: MAE and RMSE degrade upward, "
            "R-squared downward."
        )

elif page == "Model comparison":
    st.title("Metric comparison")
    status_badge(a)
    st.dataframe(metric_table(a), use_container_width=True, hide_index=True)

    fig = go.Figure()
    for m in a.metrics:
        color = "crimson" if m.status.value == "regressed" else "steelblue"
        fig.add_trace(
            go.Scatter(
                x=[m.metric],
                y=[m.absolute_change],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [m.ci_high - m.absolute_change],
                    "arrayminus": [m.absolute_change - m.ci_low],
                },
                mode="markers",
                marker={"size": 12, "color": color},
                name=m.metric,
                showlegend=False,
            )
        )
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(
        title="Change (candidate - baseline) with 95% bootstrap CI",
        yaxis_title="change",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "A metric is 'regressed' only when the change is practically large "
        "(both thresholds) AND the CI excludes zero at the chosen alpha - "
        "in the metric's own degrading direction."
    )

elif page == "Drift analysis":
    st.title("Distribution drift")
    rows = [
        {
            "feature": f.feature,
            "kind": f.kind,
            "test": f.test,
            "statistic": round(f.statistic, 4),
            "p-value": f"{f.p_value:.2e}",
            "PSI": round(f.psi, 4),
            "band": f.psi_band,
            "drifted": f.drifted,
        }
        for f in [*a.drift.features, a.drift.prediction]
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    pred_label = a.drift.prediction.feature
    chosen = st.selectbox(
        "Inspect distribution", [*bundle.feature_columns, pred_label]
    )
    fig = go.Figure()
    if chosen == pred_label:
        col = Y_PROB if Y_PROB in pair.baseline_scores.columns else "y_pred"
        base_vals = pair.baseline_scores[col]
        cand_vals = pair.candidate_scores[col]
    else:
        base_vals = bundle.eval_baseline[chosen]
        cand_vals = bundle.eval_candidate[chosen]
    fig.add_histogram(x=base_vals, name="baseline window", opacity=0.6, nbinsx=50)
    fig.add_histogram(x=cand_vals, name="candidate window", opacity=0.6, nbinsx=50)
    fig.update_layout(barmode="overlay", title=f"{chosen}: baseline vs candidate", height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Drift is context, not verdict: a feature can drift without the model "
        "degrading, and a model can regress with no visible drift."
    )

elif page == "Segment analysis":
    st.title("Segment-level performance")
    status_badge(a)
    active = [r for r in a.segments.results if not r.skipped]
    st.dataframe(pd.DataFrame(segment_rows(a)), use_container_width=True, hide_index=True)

    tested = active[0].tested_metric if active else "accuracy"
    fig = go.Figure()
    labels = [f"{r.column}={r.level}" for r in active]
    fig.add_bar(name="baseline", x=labels, y=[r.baseline[tested] for r in active])
    fig.add_bar(name="candidate", x=labels, y=[r.candidate[tested] for r in active])
    fig.update_layout(
        barmode="group",
        title=f"{tested} by segment"
        + (" (lower is better)" if tested == "mae" else ""),
        height=420,
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig, use_container_width=True)

    skipped = [r for r in a.segments.results if r.skipped]
    if skipped:
        st.caption(
            "Skipped (too small): "
            + "; ".join(f"{r.column}={r.level} ({r.skip_reason})" for r in skipped)
        )

elif page == "Explanation":
    st.title("Why this verdict")
    status_badge(a)
    st.text(explain(a))
    st.subheader("Machine-readable summary")
    st.code(json.dumps(a.to_dict(), indent=2), language="json")

else:  # Scenario notes
    st.title("Scenario mechanism (ground truth)")
    st.write(f"**{SCENARIO_LABELS[scenario]}** — task: **{task.value}**")
    st.info(bundle.notes)
    st.write(
        "Every scenario is generated deterministically from the seed shown in the "
        "sidebar; the same configuration always reproduces identical data, models "
        "and verdicts."
    )
    st.subheader("Generator configuration")
    st.code(str(bundle.config), language="text")
