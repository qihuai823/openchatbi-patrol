"""Statistical baseline and deviation scoring for patrol metrics.

The scorer answers one question: *is this value surprising given this metric's own
recent history?* It uses a robust median/MAD baseline rather than a forecast
model, for three reasons:

1. A patrol job must run unattended on a bare box. Requiring a model server over
   HTTP (as ``openchatbi.analysis.anomaly_detection`` does) makes the job fragile.
2. Median + MAD are insensitive to the very outliers we are trying to detect, so
   one bad day does not poison the baseline.
3. It is a pure function, so the whole scoring path is unit-testable without any
   LLM, database or network access.

Weekday awareness matters for business metrics: GMV on a Saturday is not
comparable to GMV on a Tuesday. When enough same-weekday history exists, the
baseline is computed per weekday; otherwise it falls back to the whole window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from openchatbi.patrol.models import Deviation, Direction, Observation

# Score ramp breakpoints, expressed in robust sigma units. The thresholds line up
# with the severity buckets in ``models.Severity`` (0.4 medium / 0.6 high /
# 0.8 critical), so a 3-sigma move is "high" and a 4-sigma move is "critical".
_Z_MEDIUM = 2.0
_Z_HIGH = 3.0
_Z_CRITICAL = 4.0
_Z_SATURATE = 6.0

# Minimum same-weekday samples before the weekday-specific baseline is trusted.
MIN_WEEKDAY_SAMPLES = 3

# Floor for the noise scale, as a fraction of the baseline level. Prevents a
# near-constant series from producing an infinite z-score on a 1-unit move.
_SCALE_FLOOR_RATIO = 0.01


def parse_bucket(bucket: str) -> date | None:
    """Parse a bucket label into a date, or ``None`` when it is not date-like.

    Supports ``YYYY-MM-DD`` (daily) and ``YYYY-MM`` (monthly, anchored to the
    first of the month). Anything else is treated as an opaque label and simply
    disables weekday awareness for that point.
    """
    text = (bucket or "").strip()
    try:
        if len(text) >= 10:
            return date.fromisoformat(text[:10])
        if len(text) == 7:
            return date.fromisoformat(f"{text}-01")
    except ValueError:
        return None
    return None


@dataclass
class Baseline:
    """Robust centre and scale of a metric's history."""

    center: float
    scale: float
    samples: int
    weekday: int | None = None

    def is_usable(self) -> bool:
        """A baseline is only usable when it has enough points and a real scale."""
        return self.samples > 0 and self.scale > 0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile, ``pct`` in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _robust_scale(values: list[float], center: float) -> float:
    """Estimate the noise scale with a chain of robust estimators.

    The order matters. MAD is the primary estimator because it has a 50% breakdown
    point. It degenerates to zero whenever more than half the values are identical
    - which is common for low-volume daily metrics and for series where most days
    sit on a round number. The naive fallback of "mean absolute deviation" is a
    trap: a single 50x spike drags it up by two orders of magnitude and the metric
    then never alerts again. So the fallback is the IQR, which keeps the 25%
    breakdown point.

    Only if both estimators collapse (a perfectly constant series) do we fall back
    to a fraction of the level, which merely prevents a division by zero.
    """
    deviations = [abs(v - center) for v in values]
    mad = _median(deviations)
    if mad > 0:
        return 1.4826 * mad

    iqr = _percentile(values, 75) - _percentile(values, 25)
    if iqr > 0:
        # 1.349 is the IQR of a standard normal, so IQR/1.349 estimates sigma.
        return iqr / 1.349

    return abs(center) * _SCALE_FLOOR_RATIO


def compute_baseline(
    history: list[Observation],
    weekday_aware: bool = True,
    min_samples: int = 3,
) -> Baseline:
    """Compute the robust baseline of a metric's history.

    Args:
        history: Prior observations, oldest first. The caller is responsible for
            excluding the point being evaluated.
        weekday_aware: When True and the buckets are daily, try to use only the
            observations that fall on the same weekday as the latest point.
        min_samples: Minimum number of observations required to trust a baseline.

    Returns:
        A :class:`Baseline`. When the history is too short, ``samples`` is below
        ``min_samples`` and ``is_usable()`` returns False.
    """
    if not history:
        return Baseline(center=0.0, scale=0.0, samples=0)

    # The global sample check comes first: ``min_samples`` describes how much
    # history a metric needs overall. Weekday filtering is an optional refinement
    # applied afterwards, and it is allowed to leave fewer points than
    # ``min_samples`` (a 35-day window only yields 5 same-weekday points).
    if len(history) < min_samples:
        return Baseline(center=_median([obs.value for obs in history]), scale=0.0, samples=len(history))

    values = [obs.value for obs in history]
    target_weekday: int | None = None

    if weekday_aware:
        parsed = [parse_bucket(obs.bucket) for obs in history]
        if all(p is not None for p in parsed):
            latest = parsed[-1]
            assert latest is not None
            same = [
                obs.value
                for obs, p in zip(history, parsed, strict=True)
                if p is not None and p.weekday() == latest.weekday()
            ]
            if len(same) >= MIN_WEEKDAY_SAMPLES:
                values = same
                target_weekday = latest.weekday()

    center = _median(values)
    scale = max(_robust_scale(values, center), abs(center) * _SCALE_FLOOR_RATIO)

    return Baseline(center=center, scale=scale, samples=len(values), weekday=target_weekday)


def score_from_z(z: float) -> float:
    """Map a signed z-score onto a continuous anomaly score in [0, 1].

    The ramp is piecewise linear and deliberately asymmetric in resolution: the
    2-4 sigma band is where most real business anomalies live, so it gets the
    finest granularity, while anything past 6 sigma saturates at 1.0.
    """
    abs_z = abs(z)
    if abs_z <= _Z_MEDIUM:
        return 0.3 * abs_z / _Z_MEDIUM
    if abs_z <= _Z_HIGH:
        return 0.3 + 0.3 * (abs_z - _Z_MEDIUM) / (_Z_HIGH - _Z_MEDIUM)
    if abs_z <= _Z_CRITICAL:
        return 0.6 + 0.2 * (abs_z - _Z_HIGH) / (_Z_CRITICAL - _Z_HIGH)
    span = _Z_SATURATE - _Z_CRITICAL
    return min(0.8 + 0.2 * (abs_z - _Z_CRITICAL) / span, 1.0)


def detect(
    latest: Observation,
    history: list[Observation],
    weekday_aware: bool = True,
    min_samples: int = 3,
    min_relative_change: float = 0.0,
) -> Deviation | None:
    """Compare the latest observation against the baseline of its history.

    Args:
        latest: The point to judge.
        history: Prior observations, oldest first, excluding ``latest``.
        weekday_aware: Use same-weekday history when possible.
        min_samples: Minimum history size before a judgement is made.
        min_relative_change: Absolute-change floor expressed as a fraction of the
            baseline. A deviation smaller than this is treated as noise and
            reported with a zero score. Use it to silence low-amplitude chatter
            on stable metrics.

    Returns:
        A :class:`Deviation`, or ``None`` when there is not enough history to
        form a baseline.
    """
    baseline = compute_baseline(history, weekday_aware=weekday_aware, min_samples=min_samples)
    if not baseline.is_usable():
        return None

    delta = latest.value - baseline.center
    z = delta / baseline.scale
    score = score_from_z(z)

    if min_relative_change > 0 and baseline.center != 0:
        relative = abs(delta) / abs(baseline.center)
        if relative < min_relative_change:
            score = 0.0

    direction = Direction.RISE if delta >= 0 else Direction.DROP
    return Deviation(
        bucket=latest.bucket,
        actual=latest.value,
        baseline=baseline.center,
        scale=baseline.scale,
        z=z,
        score=score,
        direction=direction,
    )


def detect_series(
    observations: list[Observation],
    weekday_aware: bool = True,
    min_samples: int = 3,
    min_relative_change: float = 0.0,
    max_alerts: int = 1,
) -> list[Deviation]:
    """Score the tail of a series and return the most severe deviations.

    Args:
        observations: The full series, oldest first.
        weekday_aware: Use same-weekday history when possible.
        min_samples: Minimum history before the first point can be judged.
        min_relative_change: Passed through to :func:`detect`.
        max_alerts: How many of the most recent deviations to return. The patrol
            use case only cares about the freshest point, but backtesting wants
            more, so this is a knob rather than a constant.

    Returns:
        Deviations ordered oldest first. Empty when the series is too short.
    """
    results: list[Deviation] = []
    for index in range(min_samples, len(observations)):
        deviation = detect(
            observations[index],
            observations[:index],
            weekday_aware=weekday_aware,
            min_samples=min_samples,
            min_relative_change=min_relative_change,
        )
        if deviation is not None:
            results.append(deviation)
    return results[-max_alerts:] if max_alerts > 0 else []


def z_to_confidence(z: float) -> float:
    """Two-sided normal confidence that the deviation is not noise.

    Used only for logging and for the CLI display; scoring does not depend on it.
    """
    return math.erf(abs(z) / math.sqrt(2.0))
