"""Tests for patrol baseline scoring."""

from __future__ import annotations

import pytest

from openchatbi.patrol.baseline import (
    compute_baseline,
    detect,
    detect_series,
    parse_bucket,
    score_from_z,
)
from openchatbi.patrol.models import Direction, Observation, Severity


def series(values: list[float], start_day: int = 1) -> list[Observation]:
    """Build a daily series starting at 2026-01-<start_day>."""
    return [Observation(bucket=f"2026-01-{start_day + i:02d}", value=v) for i, v in enumerate(values)]


def test_parse_bucket_handles_daily_and_monthly() -> None:
    assert parse_bucket("2026-05-31") is not None
    assert parse_bucket("2026-05-31 12:00:00") is not None
    assert parse_bucket("2026-05") is not None
    assert parse_bucket("not-a-date") is None


def test_parse_bucket_returns_none_for_month_like_garbage() -> None:
    # 7 characters but not a real month: must not raise, must not guess.
    assert parse_bucket("2026-13") is None


def test_baseline_uses_median_and_mad() -> None:
    history = series([100, 102, 98, 101, 99, 103, 97, 100, 102, 99])
    baseline = compute_baseline(history, weekday_aware=False, min_samples=5)

    assert baseline.is_usable()
    assert baseline.center == pytest.approx(100.0, abs=1.0)
    # MAD of a series spread by ~+-3 must stay small; a std-dev baseline would be
    # inflated by the same amount, so this pins the robust estimator.
    assert baseline.scale < 5.0


def test_baseline_is_robust_to_a_single_historical_spike() -> None:
    clean = series([100, 101, 99, 100, 100, 101, 99, 100, 100, 101])
    spiked = series([100, 101, 99, 100, 100, 101, 99, 100, 100, 5000])

    clean_baseline = compute_baseline(clean, weekday_aware=False, min_samples=5)
    spiked_baseline = compute_baseline(spiked, weekday_aware=False, min_samples=5)

    # The median ignores the spike entirely, so the centre must not move.
    assert spiked_baseline.center == pytest.approx(clean_baseline.center, abs=1.0)
    # The scale may widen slightly, but nowhere near the 4900-point outlier.
    assert spiked_baseline.scale < 100


def test_baseline_requires_min_samples() -> None:
    baseline = compute_baseline(series([100, 101, 99]), weekday_aware=False, min_samples=10)
    assert not baseline.is_usable()


def test_weekday_awareness_narrows_the_baseline_on_a_weekly_pattern() -> None:
    # Weekends are 3x the weekday level. A naive baseline would treat every
    # weekend as an anomaly and every weekday as a drop.
    values: list[float] = []
    for day_index in range(28):
        weekday = (day_index + 6) % 7  # 2026-03-01 is a Sunday
        values.append(300.0 if weekday >= 5 else 100.0)
    history = [Observation(bucket=f"2026-03-{i + 1:02d}", value=v) for i, v in enumerate(values)]

    naive = compute_baseline(history, weekday_aware=False, min_samples=14)
    aware = compute_baseline(history, weekday_aware=True, min_samples=14)

    assert aware.weekday is not None
    assert aware.scale < naive.scale


def test_score_ramp_hits_severity_buckets_at_expected_sigmas() -> None:
    assert score_from_z(0.0) == pytest.approx(0.0)
    assert score_from_z(2.0) == pytest.approx(0.3)
    assert score_from_z(3.0) == pytest.approx(0.6)
    assert score_from_z(4.0) == pytest.approx(0.8)
    assert score_from_z(6.0) == pytest.approx(1.0)
    # Monotonic and symmetric.
    assert score_from_z(-3.0) == pytest.approx(score_from_z(3.0))
    assert score_from_z(10.0) == pytest.approx(1.0)


def test_score_ramp_crosses_medium_and_high_where_severity_says_it_should() -> None:
    assert Severity.from_score(score_from_z(2.5)) is Severity.MEDIUM
    assert Severity.from_score(score_from_z(3.5)) is Severity.HIGH
    assert Severity.from_score(score_from_z(5.0)) is Severity.CRITICAL


def test_detect_flags_a_rise() -> None:
    history = series([100, 102, 98, 101, 99, 103, 97, 100, 102, 99, 101, 100])
    latest = Observation(bucket="2026-01-13", value=300)

    deviation = detect(latest, history, weekday_aware=False, min_samples=5)

    assert deviation is not None
    assert deviation.direction is Direction.RISE
    assert deviation.score > 0.8
    assert deviation.severity is Severity.CRITICAL
    assert deviation.delta_pct == pytest.approx(200.0, abs=5)


def test_detect_flags_a_drop() -> None:
    history = series([100, 102, 98, 101, 99, 103, 97, 100, 102, 99, 101, 100])
    latest = Observation(bucket="2026-01-13", value=5)

    deviation = detect(latest, history, weekday_aware=False, min_samples=5)

    assert deviation is not None
    assert deviation.direction is Direction.DROP
    assert deviation.score > 0.8


def test_detect_returns_none_without_enough_history() -> None:
    assert detect(Observation(bucket="2026-01-04", value=100), series([100, 100]), min_samples=10) is None


def test_min_relative_change_silences_small_moves() -> None:
    history = series([100, 100, 100, 100, 100, 100, 100, 100])
    # A move of 1% on a very tight series produces a large z but is not news.
    latest = Observation(bucket="2026-01-09", value=101)

    loud = detect(latest, history, weekday_aware=False, min_samples=5)
    quiet = detect(latest, history, weekday_aware=False, min_samples=5, min_relative_change=0.05)

    assert loud is not None and loud.score > 0
    assert quiet is not None and quiet.score == 0.0


def test_flat_history_does_not_divide_by_zero() -> None:
    history = series([50, 50, 50, 50, 50, 50, 50, 50])
    deviation = detect(Observation(bucket="2026-01-09", value=80), history, weekday_aware=False, min_samples=5)

    assert deviation is not None
    # Scale collapses to the level floor (1% of 50 = 0.5), so the z-score stays
    # finite and the score saturates instead of becoming infinite or NaN.
    assert deviation.scale == pytest.approx(0.5)
    assert deviation.z == pytest.approx(60.0)
    assert deviation.score == pytest.approx(1.0)


def test_detect_series_returns_only_the_requested_tail() -> None:
    values = [100] * 15 + [500]
    deviations = detect_series(series(values), weekday_aware=False, min_samples=5, max_alerts=1)

    assert len(deviations) == 1
    assert deviations[0].bucket == "2026-01-16"
