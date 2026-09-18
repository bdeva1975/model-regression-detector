# Detection methodology

## Why "did the model get worse?" is a hard question

- **Metric variability.** Any metric computed on a finite window is a noisy
  estimate. Two evaluations of the *same* model differ; a naive
  before/after comparison alarms on noise.
- **Aggregation hides damage.** A model can hold its overall accuracy while
  one segment collapses (our segment scenario: overall accuracy -4.5 points,
  affected segment -17 points, segment recall 0.73 → 0.12).
- **The wrong metric looks fine.** A shifted decision boundary can raise
  precision while recall collapses and ranking metrics (AUC) stay intact -
  the model still *ranks* well but decides badly.
- **Drift is not regression.** Features can drift with zero performance
  impact (non-informative features shifting), and models can regress with
  zero drift (bad training labels). Conflating the two produces alert
  fatigue in one direction and false confidence in the other.
- **Statistical vs practical significance.** With 20,000 evaluation samples,
  a 0.3-point recall dip is statistically detectable and operationally
  irrelevant. With 500 samples, a 5-point dip may be real and undetectable.
  A system that ignores either half of this is not trustworthy.

## The decision rule

A metric is **REGRESSED** only if all three hold, in its own orientation
(log loss degrades upward, all others downward):

1. absolute degradation ≥ `min_absolute_degradation` (default 0.01),
2. relative degradation ≥ `min_relative_degradation` (default 5%),
3. bootstrap p-value < `significance_level` (default 0.05).

Requiring *both* absolute and relative thresholds prevents two failure
modes: tiny baselines inflating relative change (0.02 → 0.01 is "-50%"),
and large baselines hiding absolute change.

Partial evidence gets its own named status instead of a forced yes/no:

| Practically large | Statistically confirmed | Status |
|---|---|---|
| yes | yes | **regressed** |
| no | yes | **negligible** - reported, not alerted |
| yes | no | **inconclusive** - more data would settle it |
| no | no | ok |
| improved & confirmed | - | improved |

## Overall verdict and severity

`REGRESSION` if any metric regressed **or** any segment degraded. Severity
grades on the worst relative metric degradation (`severity_moderate` /
`high` / `critical` config thresholds); a segment-only regression is at
least MODERATE. **Drift never triggers a regression verdict by itself** -
it is reported as supporting context, because drift is a property of the
data, not of model quality.

## Segment analysis

Per level of each segment column: threshold metrics for both sides, a
within-segment bootstrap on the accuracy difference, and Bonferroni
adjustment across all levels actually tested. Segments below
`min_segment_size` on either side are reported as skipped, never silently
dropped. Note that segmentations are correlated views of the same rows:
damage concentrated in one region also surfaces, diluted, in any customer
type that overlaps it. The explanation names the largest drop as primary.

## What the scenario lab demonstrates

Each scenario is engineered so the intended signature - and *only* that
signature - appears: recall regression leaves AUC intact; feature drift
fires the drift detectors but not the verdict; the significant-but-small
scenario produces a real, confirmed, sub-threshold dip that the system
explicitly labels negligible. The test suite pins these behaviours.