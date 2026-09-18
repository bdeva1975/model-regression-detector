# Architecture

## Design goals

1. **UI-independent core.** Everything under `src/` runs without Streamlit;
   `app.py` is a thin presentation layer over `assess()`. The core is usable
   as a library today and as a CLI/CI quality gate later without refactoring.
2. **Determinism end to end.** One seed drives data generation, model
   training, and bootstrap resampling. Same configuration → identical data,
   models, verdicts, and explanation text.
3. **Evidence flows one way.** Each stage consumes the previous stage's
   structured output and never reaches backwards (e.g. the explanation
   engine sees only the assessment, never raw data).

## Pipeline

```text
GeneratorConfig ──▶ generate() ──▶ ScenarioBundle
                                     │  (train/eval frames, ground-truth notes)
                                     ▼
                    train_pair() ──▶ TrainedPair
                                     │  (scored frames: y_true, y_prob, y_pred, segments)
                                     ▼
        ┌──────────────────┬─────────┴────────┬───────────────────┐
        ▼                  ▼                  ▼                   ▼
 compute_metrics()  bootstrap_metric_diff()  detect_drift()  analyze_segments()
        └──────────────────┴─────────┬────────┴───────────────────┘
                                     ▼
                       assess() ──▶ RegressionAssessment
                                     │
                                     ▼
                       explain() ──▶ deterministic prose + to_dict() JSON
```

## Component responsibilities

| Module | Responsibility | Key output |
|---|---|---|
| `data/synthetic_generator.py` | Nine seeded scenarios; corrupts **training labels**, never predictions | `ScenarioBundle` |
| `models/model_factory.py` | sklearn pipelines (scaler + one-hot segments + classifier); scoring | `TrainedPair` |
| `evaluation/metrics.py` | Seven classification metrics with orientation metadata | `MetricReport` |
| `detection/statistical_tests.py` | Two-sample bootstrap: CIs, p-values | `BootstrapResult` |
| `detection/drift_detector.py` | KS / chi-square / PSI on features and prediction scores | `DriftReport` |
| `analysis/segment_analysis.py` | Per-segment comparison, Bonferroni adjustment | `SegmentReport` |
| `detection/regression_detector.py` | Combines all evidence into one verdict | `RegressionAssessment` |
| `analysis/explanation.py` | Deterministic prose from the assessment | `str` |
| `utils/config.py` | Every threshold, validated, no magic numbers downstream | `DetectionConfig` |

## Two design decisions worth explaining

**Regressions are learned, not staged.** Scenarios corrupt the candidate's
*training labels* so the candidate model genuinely learns worse behaviour;
its predictions are honest outputs of a bad training process. Tampering
with predictions post-hoc would demonstrate nothing about detection - any
threshold check can catch fabricated numbers.

**Segments are model features.** Segment columns are one-hot encoded into
the model input. This mirrors production practice and is what makes
segment-level regression mechanically possible: during development we
verified that a segment-blind model cannot learn segment-specific
misbehaviour - symmetric label noise in one region simply diluted into
global noise and no localized damage appeared. Directional corruption of a
segment the model can see produces the real failure mode: the candidate
learns "this segment doesn't convert" and its recall collapses there.

## Extension points

- **New model types**: add a branch in `build_estimator()`.
- **New metrics**: add to `CLASSIFICATION_METRICS` + `METRIC_ORIENTATION`
  + `_metric_value()`; the decision engine picks them up unchanged.
- **New drift detectors**: return `FeatureDriftResult` from a new function
  and append in `detect_drift()`.
- **Real data**: replace `ScenarioBundle`'s frames with your own scored
  windows; nothing downstream knows the data is synthetic.