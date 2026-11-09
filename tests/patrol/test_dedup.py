"""Tests for patrol de-duplication and suppression."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from openchatbi.patrol.dedup import SuppressionRule, evaluate, find_suppression, resolve, summarize
from openchatbi.patrol.models import Alert, AlertFingerprint, AlertState, Deviation, Direction, Severity

NOW = datetime(2026, 5, 31, 9, 0, 0)
COOLDOWN = timedelta(hours=24)


def make_alert(
    score: float = 0.9,
    bucket: str = "2026-05-31",
    metric: str = "gmv",
    direction: Direction = Direction.DROP,
    dimension: str | None = None,
    dimension_value: str | None = None,
) -> Alert:
    deviation = Deviation(
        bucket=bucket,
        actual=10.0,
        baseline=100.0,
        scale=5.0,
        z=-18.0,
        score=score,
        direction=direction,
    )
    return Alert(
        fingerprint=AlertFingerprint(
            metric=metric, dimension=dimension, dimension_value=dimension_value, direction=None
        ),
        deviation=deviation,
        metric_label="成交总额 GMV",
        dimension_label=dimension_value,
    )


def prior_state(score: float = 0.9, last_pushed: datetime | None = None, push_count: int = 1) -> AlertState:
    return AlertState(
        fingerprint_key="gmv",
        first_seen=NOW - timedelta(days=2),
        last_seen=NOW - timedelta(days=1),
        last_pushed=last_pushed,
        push_count=push_count,
        last_score=score,
        last_bucket="2026-05-30",
        status="open",
    )


def test_first_sighting_pushes() -> None:
    decision = evaluate(make_alert(), None, now=NOW, cooldown=COOLDOWN)

    assert decision.push is True
    assert decision.reason == "new"
    assert decision.state.push_count == 1
    assert decision.state.last_pushed == NOW


def test_second_sighting_inside_cooldown_is_suppressed() -> None:
    previous = prior_state(last_pushed=NOW - timedelta(hours=2))

    decision = evaluate(make_alert(), previous, now=NOW, cooldown=COOLDOWN)

    assert decision.push is False
    assert decision.reason.startswith("cooldown:")
    # The state still records the sighting, so the audit trail shows we looked.
    assert decision.state.last_seen == NOW
    assert decision.state.push_count == 1


def test_escalation_overrides_the_cooldown() -> None:
    previous = prior_state(score=0.5, last_pushed=NOW - timedelta(hours=1))

    decision = evaluate(make_alert(score=0.95), previous, now=NOW, cooldown=COOLDOWN, escalate=True)

    assert decision.push is True
    assert decision.reason == "escalated:medium->critical"
    assert decision.state.push_count == 2


def test_escalation_can_be_disabled() -> None:
    previous = prior_state(score=0.5, last_pushed=NOW - timedelta(hours=1))

    decision = evaluate(make_alert(score=0.95), previous, now=NOW, cooldown=COOLDOWN, escalate=False)

    assert decision.push is False
    assert decision.reason.startswith("cooldown:")


def test_same_severity_is_not_an_escalation() -> None:
    # 0.85 and 0.95 are both critical; jitter inside a bucket must not re-page.
    previous = prior_state(score=0.85, last_pushed=NOW - timedelta(hours=1))

    decision = evaluate(make_alert(score=0.95), previous, now=NOW, cooldown=COOLDOWN)

    assert decision.push is False


def test_reminder_after_cooldown_elapses() -> None:
    previous = prior_state(score=0.9, last_pushed=NOW - timedelta(hours=25))

    decision = evaluate(make_alert(score=0.9), previous, now=NOW, cooldown=COOLDOWN)

    assert decision.push is True
    assert decision.reason == "reminder"
    assert decision.state.push_count == 2


def test_low_severity_is_recorded_but_never_pushed() -> None:
    decision = evaluate(make_alert(score=0.2), None, now=NOW, cooldown=COOLDOWN, min_severity=Severity.MEDIUM)

    assert decision.push is False
    assert decision.reason.startswith("below_min_severity:")
    assert decision.state.push_count == 0
    assert decision.state.last_pushed is None


def test_severity_floor_is_configurable() -> None:
    decision = evaluate(make_alert(score=0.5), None, now=NOW, cooldown=COOLDOWN, min_severity=Severity.HIGH)

    assert decision.push is False
    assert decision.reason == "below_min_severity:medium<high"


def test_suppression_rule_matches_on_bucket_glob() -> None:
    rule = SuppressionRule(reason="partial month", bucket_glob="2025-11*")
    alert = make_alert(bucket="2025-11-30")

    assert find_suppression(alert, [rule]) == "partial month"
    assert find_suppression(make_alert(bucket="2026-05-31"), [rule]) is None


def test_suppression_rule_requires_every_set_field_to_match() -> None:
    rule = SuppressionRule(reason="electronics only", metric="gmv", dimension="category", dimension_value="electronics")

    assert find_suppression(make_alert(dimension="category", dimension_value="electronics"), [rule]) is not None
    assert find_suppression(make_alert(dimension="category", dimension_value="kitchen"), [rule]) is None
    assert find_suppression(make_alert(), [rule]) is None


def test_first_matching_suppression_rule_wins() -> None:
    rules = [
        SuppressionRule(reason="first", metric="gmv"),
        SuppressionRule(reason="second", metric="gmv"),
    ]

    assert find_suppression(make_alert(), rules) == "first"


def test_resolve_marks_the_fingerprint_resolved() -> None:
    previous = prior_state(last_pushed=NOW - timedelta(hours=3))

    resolved = resolve("gmv", previous, NOW)

    assert resolved.status == "resolved"
    assert resolved.last_score == 0.0
    assert resolved.push_count == previous.push_count
    assert not resolved.is_open()


def test_summarize_counts_reason_prefixes() -> None:
    decisions = [
        evaluate(make_alert(), None, now=NOW, cooldown=COOLDOWN),
        evaluate(make_alert(), prior_state(last_pushed=NOW - timedelta(hours=1)), now=NOW, cooldown=COOLDOWN),
        evaluate(make_alert(), prior_state(last_pushed=NOW - timedelta(hours=30)), now=NOW, cooldown=COOLDOWN),
    ]

    counts = summarize(decisions)

    assert counts["new"] == 1
    assert counts["cooldown"] == 1
    assert counts["reminder"] == 1


def test_cooldown_reason_reports_remaining_minutes() -> None:
    previous = prior_state(last_pushed=NOW - timedelta(hours=23))

    decision = evaluate(make_alert(), previous, now=NOW, cooldown=COOLDOWN)

    assert decision.push is False
    assert "60m_left" in decision.reason


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.2, Severity.LOW), (0.5, Severity.MEDIUM), (0.7, Severity.HIGH), (0.9, Severity.CRITICAL)],
)
def test_severity_buckets(score: float, expected: Severity) -> None:
    assert make_alert(score=score).severity is expected
