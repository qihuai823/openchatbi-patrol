"""De-duplication and suppression for patrol alerts.

A patrol job that pages on every run is worse than no patrol job: it trains people
to ignore the channel. This module is the gate between "we detected a deviation"
and "we woke someone up".

Four rules, applied in order:

1. **Severity floor** - low-severity deviations are recorded but never pushed.
2. **Cooldown** - a fingerprint that was already pushed inside the cooldown window
   stays quiet. This is what makes a persistent problem page once, not nightly.
3. **Escalation override** - if the same fingerprint got worse (severity rank went
   up), it pushes immediately even inside the cooldown. A worsening problem must
   not wait for a timer.
4. **Reminder** - an unresolved fingerprint whose cooldown has elapsed pushes
   again, so a long-running problem does not silently disappear.

Everything here is a pure function over the previous state, which is what makes
the behaviour testable without a scheduler, a database or a clock.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from datetime import datetime, timedelta

from openchatbi.patrol.models import Alert, AlertState, Severity


@dataclass(frozen=True)
class SuppressionRule:
    """A standing reason to ignore certain alerts.

    Used for known non-issues such as an incomplete data month or a planned
    migration window. Matching is conjunctive: every non-``None`` field must
    match for the rule to fire.
    """

    reason: str
    metric: str | None = None
    dimension: str | None = None
    dimension_value: str | None = None
    bucket_glob: str | None = None

    def matches(self, alert: Alert) -> bool:
        fp = alert.fingerprint
        if self.metric is not None and fp.metric != self.metric:
            return False
        if self.dimension is not None and fp.dimension != self.dimension:
            return False
        if self.dimension_value is not None and fp.dimension_value != self.dimension_value:
            return False
        if self.bucket_glob is not None and not fnmatch.fnmatch(alert.deviation.bucket, self.bucket_glob):
            return False
        return True


@dataclass
class Decision:
    """Outcome of the de-duplication gate for a single alert."""

    push: bool
    reason: str
    state: AlertState


def find_suppression(alert: Alert, rules: list[SuppressionRule]) -> str | None:
    """Return the reason string of the first matching rule, or ``None``."""
    for rule in rules:
        if rule.matches(alert):
            return rule.reason
    return None


def evaluate(
    alert: Alert,
    previous: AlertState | None,
    now: datetime,
    cooldown: timedelta,
    min_severity: Severity = Severity.MEDIUM,
    escalate: bool = True,
) -> Decision:
    """Decide whether an alert should be pushed, and compute the next state.

    Args:
        alert: The alert produced by this run.
        previous: Persisted state for this fingerprint, or ``None`` on first sight.
        now: Timestamp of this run. Injected so tests are deterministic.
        cooldown: Minimum quiet period between two pushes of one fingerprint.
        min_severity: Alerts below this severity are recorded but never pushed.
        escalate: When True, a severity increase overrides the cooldown.

    Returns:
        A :class:`Decision` carrying the push verdict, a machine-readable reason
        and the state that should be persisted.
    """
    severity = alert.severity
    is_new = previous is None

    first_seen = previous.first_seen if previous is not None else now
    push_count = previous.push_count if previous is not None else 0
    last_pushed = previous.last_pushed if previous is not None else None
    previous_score = previous.last_score if previous is not None else 0.0

    state = AlertState(
        fingerprint_key=alert.fingerprint.key,
        first_seen=first_seen,
        last_seen=now,
        last_pushed=last_pushed,
        push_count=push_count,
        last_score=alert.score,
        last_bucket=alert.deviation.bucket,
        status="open",
    )

    # Rule 1: severity floor. Record the sighting, but stay quiet.
    if severity.rank < min_severity.rank:
        return Decision(
            push=False,
            reason=f"below_min_severity:{severity.value}<{min_severity.value}",
            state=state,
        )

    if is_new:
        state.last_pushed = now
        state.push_count = 1
        return Decision(push=True, reason="new", state=state)

    assert previous is not None  # narrowed by is_new above

    # Rule 3: escalation overrides the cooldown. Compared on severity rank rather
    # than raw score so that jitter inside one bucket does not count as escalation.
    if escalate and severity.rank > Severity.from_score(previous_score).rank:
        state.last_pushed = now
        state.push_count += 1
        return Decision(
            push=True,
            reason=f"escalated:{Severity.from_score(previous_score).value}->{severity.value}",
            state=state,
        )

    # Rule 2: cooldown.
    if last_pushed is not None and now - last_pushed < cooldown:
        remaining = cooldown - (now - last_pushed)
        return Decision(
            push=False,
            reason=f"cooldown:{int(remaining.total_seconds() // 60)}m_left",
            state=state,
        )

    # Rule 4: reminder for a still-unresolved problem.
    state.last_pushed = now
    state.push_count += 1
    return Decision(push=True, reason="reminder", state=state)


def resolve(fingerprint_key: str, previous: AlertState, now: datetime) -> AlertState:
    """Mark a fingerprint as resolved.

    Called when a metric returns inside its normal band. The next time the same
    fingerprint appears it will be treated as a fresh problem, because ``evaluate``
    reads ``status`` via the caller deciding whether to pass state at all.
    """
    return AlertState(
        fingerprint_key=fingerprint_key,
        first_seen=previous.first_seen,
        last_seen=now,
        last_pushed=previous.last_pushed,
        push_count=previous.push_count,
        last_score=0.0,
        last_bucket=previous.last_bucket,
        status="resolved",
    )


def summarize(decisions: list[Decision]) -> dict[str, int]:
    """Aggregate decisions by reason prefix, for run logs."""
    counts: dict[str, int] = {}
    for decision in decisions:
        key = decision.reason.split(":", 1)[0]
        counts[key] = counts.get(key, 0) + 1
    return counts
