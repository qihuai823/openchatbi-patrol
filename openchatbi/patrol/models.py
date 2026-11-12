"""Data models for the proactive patrol subsystem.

The patrol subsystem turns the passive "user asks, agent answers" flow into an
active one: a schedule evaluates a set of business metrics, flags the ones that
deviate from their own history, and pushes only the alerts that are new or still
unresolved.

Design notes
------------
* ``AlertFingerprint`` deliberately excludes the time bucket. A metric that is
  anomalous on three consecutive days is the *same* ongoing problem, so it must
  not page three times. The bucket lives on the ``Alert`` for reporting while the
  fingerprint drives de-duplication.
* Anomaly scoring is statistical (see :mod:`openchatbi.patrol.baseline`) rather
  than forecast-model based. The existing ``openchatbi.analysis.anomaly_detection``
  module requires an external forecasting service over HTTP; a patrol job must
  run on a bare cron box with no model server, so the cheap baseline wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class Direction(StrEnum):
    """Which way the metric moved away from its baseline."""

    RISE = "rise"
    DROP = "drop"


class Severity(StrEnum):
    """Coarse severity bucket derived from the continuous anomaly score."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_score(cls, score: float) -> Severity:
        """Map a score in [0, 1] onto a severity bucket.

        Thresholds match ``openchatbi.analysis.anomaly_detection._severity_label``
        so that patrol alerts and interactive analysis report the same words.
        """
        if score >= 0.8:
            return cls.CRITICAL
        if score >= 0.6:
            return cls.HIGH
        if score >= 0.4:
            return cls.MEDIUM
        return cls.LOW

    @property
    def rank(self) -> int:
        """Numeric rank, useful for sorting and for ``min_severity`` filters."""
        return {"low": 0, "medium": 1, "high": 2, "critical": 3}[self.value]


@dataclass(frozen=True)
class AlertFingerprint:
    """Stable identity of an alert.

    Two alerts with the same fingerprint describe the same underlying problem,
    which is what de-duplication keys on.

    Attributes:
        metric: Metric name from the patrol plan, e.g. ``gmv``.
        dimension: Optional dimension name, e.g. ``category``.
        dimension_value: Optional dimension member, e.g. ``electronics``.
        direction: Expected direction of the problem. ``None`` means "any
            direction", which lets a rise and a drop on the same series be
            treated as one problem only when the plan says so.
    """

    metric: str
    dimension: str | None = None
    dimension_value: str | None = None
    direction: Direction | None = None

    @property
    def key(self) -> str:
        """Serialize the fingerprint into a stable, human-readable key."""
        parts = [self.metric]
        if self.dimension:
            parts.append(f"{self.dimension}={self.dimension_value or '*'}")
        if self.direction is not None:
            parts.append(self.direction.value)
        return "|".join(parts)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.key


@dataclass
class Observation:
    """A single point of a metric time series."""

    bucket: str
    value: float


@dataclass
class Deviation:
    """Result of comparing one observation against its own baseline."""

    bucket: str
    actual: float
    baseline: float
    scale: float
    z: float
    score: float
    direction: Direction

    @property
    def severity(self) -> Severity:
        return Severity.from_score(self.score)

    @property
    def delta_pct(self) -> float | None:
        """Relative change vs baseline, or ``None`` when the baseline is zero."""
        if self.baseline == 0:
            return None
        return (self.actual - self.baseline) / abs(self.baseline) * 100.0


@dataclass
class Alert:
    """One alert produced by a patrol run."""

    fingerprint: AlertFingerprint
    deviation: Deviation
    metric_label: str
    dimension_label: str | None = None
    context: dict[str, object] = field(default_factory=dict)

    @property
    def severity(self) -> Severity:
        return self.deviation.severity

    @property
    def score(self) -> float:
        return self.deviation.score

    def summary(self) -> str:
        """One-line human-readable summary, used by notifiers and the CLI."""
        who = self.metric_label
        if self.dimension_label:
            who = f"{who} · {self.dimension_label}"
        change = self.deviation.delta_pct
        change_txt = f"{change:+.1f}%" if change is not None else "n/a"
        arrow = "▲" if self.deviation.direction is Direction.RISE else "▼"
        return (
            f"[{self.severity.value.upper()}] {who} {arrow} {change_txt} "
            f"on {self.deviation.bucket} (actual={self.deviation.actual:,.2f}, "
            f"baseline={self.deviation.baseline:,.2f}, score={self.score:.2f})"
        )


@dataclass
class AlertState:
    """Persisted lifecycle state of one fingerprint.

    This is what makes patrol safe to run on a schedule: the state remembers when
    a problem was first seen and when it was last pushed, so a persistent anomaly
    does not re-page on every run.
    """

    fingerprint_key: str
    first_seen: datetime
    last_seen: datetime
    last_pushed: datetime | None
    push_count: int
    last_score: float
    last_bucket: str
    status: str = "open"

    def is_open(self) -> bool:
        return self.status == "open"


@dataclass
class PatrolResult:
    """Outcome of one patrol run, suitable for logging and for the CLI.

    ``series_evaluated`` counts how many metric/dimension series were judged, while
    ``alerts`` holds the ones that actually deviated. Keeping them separate matters
    in run logs: "evaluated 120 series, detected 3, pushed 1" is diagnosable,
    whereas a single counter is not.
    """

    run_at: datetime
    series_evaluated: int = 0
    alerts: list[Alert] = field(default_factory=list)
    pushed: list[Alert] = field(default_factory=list)
    suppressed: list[tuple[Alert, str]] = field(default_factory=list)
    resolved: list[str] = field(default_factory=list)
    skipped_metrics: int = 0
    failed_metrics: int = 0
    run_id: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def detected(self) -> int:
        """Number of deviations detected, pushed or not."""
        return len(self.alerts)

    def add_suppressed(self, alert: Alert, reason: str) -> None:
        self.suppressed.append((alert, reason))

    def summary_line(self) -> str:
        """One-line run summary for cron logs."""
        return (
            f"series={self.series_evaluated} detected={self.detected} pushed={len(self.pushed)} "
            f"suppressed={len(self.suppressed)} resolved={len(self.resolved)} errors={len(self.errors)}"
        )
