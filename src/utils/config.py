"""Central configuration for the Model Regression Detection System.

Every threshold that influences a regression decision lives here, so the
decision logic contains no magic numbers. All defaults are documented and
overridable, both programmatically and from the Streamlit UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields


@dataclass
class DetectionConfig:
    """Thresholds and knobs governing regression and drift decisions.

    Practical significance:
        A metric counts as *practically* degraded only if it worsens by at
        least ``min_absolute_degradation`` (absolute points) AND at least
        ``min_relative_degradation`` (fraction of the baseline value).
        Requiring both prevents tiny baselines from inflating relative
        change and large baselines from hiding absolute change.

    Statistical significance:
        Assessed separately (bootstrap / analytic tests) at
        ``significance_level``. A finding must be BOTH practically and
        statistically significant to trigger a regression alert.
    """

    # --- Reproducibility -------------------------------------------------
    random_seed: int = 42

    # --- Practical significance ------------------------------------------
    min_absolute_degradation: float = 0.01
    min_relative_degradation: float = 0.05

    # --- Statistical significance ----------------------------------------
    significance_level: float = 0.05
    min_sample_size: int = 100
    bootstrap_iterations: int = 2000

    # --- Severity grading (relative degradation of the worst metric) -----
    severity_moderate: float = 0.05   # >= 5%  relative -> MODERATE
    severity_high: float = 0.10       # >= 10% relative -> HIGH
    severity_critical: float = 0.20   # >= 20% relative -> CRITICAL

    # --- Drift detection --------------------------------------------------
    psi_warning: float = 0.10         # PSI >= 0.10 -> moderate shift
    psi_alert: float = 0.25           # PSI >= 0.25 -> major shift
    drift_significance_level: float = 0.05
    psi_bins: int = 10

    # --- Segment analysis -------------------------------------------------
    min_segment_size: int = 50
    segment_columns: tuple[str, ...] = field(
        default_factory=lambda: ("region", "customer_type", "risk_band")
    )

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` on any inconsistent or out-of-range setting."""
        if not 0 < self.significance_level < 1:
            raise ValueError(
                f"significance_level must be in (0, 1), got {self.significance_level}"
            )
        if not 0 < self.drift_significance_level < 1:
            raise ValueError(
                f"drift_significance_level must be in (0, 1), got {self.drift_significance_level}"
            )
        if self.min_absolute_degradation < 0:
            raise ValueError("min_absolute_degradation must be >= 0")
        if self.min_relative_degradation < 0:
            raise ValueError("min_relative_degradation must be >= 0")
        if not self.severity_moderate <= self.severity_high <= self.severity_critical:
            raise ValueError(
                "severity thresholds must be ordered: moderate <= high <= critical "
                f"(got {self.severity_moderate}, {self.severity_high}, {self.severity_critical})"
            )
        if not 0 <= self.psi_warning <= self.psi_alert:
            raise ValueError(
                f"PSI thresholds must satisfy 0 <= warning <= alert "
                f"(got {self.psi_warning}, {self.psi_alert})"
            )
        if self.min_sample_size < 2:
            raise ValueError("min_sample_size must be >= 2")
        if self.min_segment_size < 2:
            raise ValueError("min_segment_size must be >= 2")
        if self.bootstrap_iterations < 100:
            raise ValueError("bootstrap_iterations must be >= 100 for stable intervals")
        if self.psi_bins < 2:
            raise ValueError("psi_bins must be >= 2")

    def with_overrides(self, **overrides: object) -> DetectionConfig:
        """Return a new validated config with the given fields replaced."""
        current = {f.name: getattr(self, f.name) for f in fields(self)}
        unknown = set(overrides) - set(current)
        if unknown:
            raise ValueError(f"Unknown config fields: {sorted(unknown)}")
        current.update(overrides)
        return DetectionConfig(**current)  # type: ignore[arg-type]


DEFAULT_CONFIG = DetectionConfig()
