# Statistical methods

Every test below is documented with purpose, hypotheses, assumptions, and
limitations. Nothing is applied merely because a library offers it.

## Two-sample bootstrap (metric differences)

- **Purpose**: quantify uncertainty in `candidate_metric - baseline_metric`
  when the two models are evaluated on *independent* windows.
- **Procedure**: resample each window with replacement (default 1000
  iterations), recompute the metric on each resample pair, collect the
  distribution of differences. Percentile CI at `1 - alpha`; achieved
  significance level `p = 2 * min(P(diff <= 0), P(diff >= 0))`, clipped to
  `[1/B, 1]`.
- **Hypotheses**: H0 - the metric difference is 0; H1 - it is not (two-sided).
- **Why bootstrap**: one method covers all seven metrics, including those
  with no convenient analytic variance (F1, PR-AUC, log loss), and makes
  no normality assumption.
- **Assumptions**: i.i.d. observations within each window; windows
  independent of each other.
- **Limitations**: production windows with temporal correlation violate
  i.i.d. and make true intervals wider than reported; percentile CIs can be
  biased for skewed statistics at small n (a minimum sample size is
  enforced, which mitigates but does not cure); p-value resolution is
  bounded by 1/iterations. Resamples that collapse to a single class are
  skipped for metrics undefined there, and the run fails loudly if too few
  effective iterations remain.

## Kolmogorov-Smirnov two-sample test (numeric features, prediction scores)

- **Purpose**: detect any distributional difference between the baseline and
  candidate windows for a numeric variable.
- **Hypotheses**: H0 - both samples come from the same distribution;
  H1 - they do not.
- **Assumptions**: continuous distributions, independent samples. Ties (as
  in near-constant features) make it conservative.
- **Limitations**: at large n it flags trivially small shifts - which is
  precisely why a magnitude measure (PSI) accompanies every p-value, and a
  feature is flagged as drifted only when **both** the test is significant
  **and** PSI ≥ `psi_warning`.

## Chi-square test of homogeneity (categorical features)

- **Purpose**: compare level frequencies of a categorical feature across
  the two windows.
- **Hypotheses**: H0 - level proportions are equal; H1 - they differ.
- **Assumptions**: expected counts ≥ 5 per cell; sparse levels are pooled
  into a synthetic "other" bucket before testing.
- **Limitations**: same large-n sensitivity as KS; paired with PSI for the
  same reason.

## Population Stability Index (PSI)

- **Purpose**: a *magnitude* measure of distributional shift, complementing
  the p-value tests. `PSI = Σ (q_i - p_i) · ln(q_i / p_i)` over bins.
- **Binning**: numeric features use quantile bins defined on the
  **baseline** distribution (outer edges extended to ±∞); categoricals use
  level frequencies. Proportions are clipped away from zero before the log.
- **Conventional bands**: < 0.10 stable, 0.10-0.25 moderate, ≥ 0.25 major.
  These are industry rules of thumb, not derived quantities; they are
  configurable and should be read as such.
- **Limitations**: no sampling distribution, hence no p-value; sensitive to
  bin count; the bands have no universal statistical meaning.

## Bonferroni adjustment (segment testing)

- **Purpose**: control the family-wise false-alarm rate when ~9 segment
  levels are each bootstrap-tested.
- **Procedure**: multiply each within-segment p-value by the number of
  segments actually tested, cap at 1.
- **Limitations**: conservative - power drops as segment count grows, and
  small segments may hide real damage. The trade-off is deliberate: for an
  alerting gate, a missed marginal segment costs less than routine false
  alarms. Skipped (undersized) segments are listed explicitly.

## What is deliberately absent

- **No composite health score.** A 0-100 score would require arbitrary
  weights across incommensurable evidence (metrics, drift, segments) and
  would invite false precision. The structured verdict + severity grade +
  explanation carries the same information honestly.
- **No permutation tests.** They answer the same question as the bootstrap
  here at similar cost; adding both would be statistical theater.