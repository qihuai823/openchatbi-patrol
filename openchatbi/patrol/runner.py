"""Patrol runner: evaluate a plan against a warehouse and emit alerts.

This is the piece that turns the passive agent into an active one. It runs on a
schedule with no user in the loop, which imposes two constraints the interactive
path does not have:

* **It must be idempotent per bucket.** Running twice for the same day must not
  page twice. That is handled by the cooldown state, not by the runner.
* **It must not need an LLM.** Detection is statistical and data access is plain
  SQL, so a patrol run costs nothing and cannot fail on a model outage. The agent
  is only involved when a human asks a follow-up question about an alert.

The runner is deliberately synchronous and dependency-light so it can be dropped
into cron, a Celery beat, or a GitHub Action without extra machinery.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from openchatbi.patrol.baseline import detect, parse_bucket
from openchatbi.patrol.dedup import evaluate, find_suppression, resolve
from openchatbi.patrol.models import (
    Alert,
    AlertFingerprint,
    Direction,
    Observation,
    PatrolResult,
)
from openchatbi.patrol.plan import MetricSpec, PatrolPlan
from openchatbi.patrol.store import AlertStateStore

BUCKET_COLUMN = "bucket"
VALUE_COLUMN = "value"


class PatrolError(RuntimeError):
    """Raised when a metric query cannot be turned into a usable series."""


def _fetch_series(
    connection: sqlite3.Connection,
    metric: MetricSpec,
) -> dict[str | None, list[Observation]]:
    """Run a metric query and split the result into one series per dimension value.

    Returns a mapping of ``dimension_value -> observations`` ordered by bucket.
    For a metric without a dimension the mapping has a single ``None`` key.
    """
    connection.row_factory = sqlite3.Row
    try:
        cursor = connection.execute(metric.sql)
    except sqlite3.Error as exc:
        raise PatrolError(f"metric {metric.name!r}: SQL failed: {exc}") from exc

    columns = [description[0] for description in cursor.description or []]
    for required in (BUCKET_COLUMN, VALUE_COLUMN):
        if required not in columns:
            raise PatrolError(
                f"metric {metric.name!r}: query must return a `{required}` column, got {columns}"
            )
    if metric.dimension and metric.dimension not in columns:
        raise PatrolError(
            f"metric {metric.name!r}: dimension `{metric.dimension}` declared but not returned by the query "
            f"(columns: {columns})"
        )

    grouped: dict[str | None, list[Observation]] = defaultdict(list)
    for row in cursor.fetchall():
        raw_value = row[VALUE_COLUMN]
        if raw_value is None:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        dimension_value = str(row[metric.dimension]) if metric.dimension else None
        grouped[dimension_value].append(Observation(bucket=str(row[BUCKET_COLUMN]), value=value))

    for observations in grouped.values():
        observations.sort(key=lambda obs: obs.bucket)

    return dict(grouped)


def _drop_incomplete_tail(series: list[Observation], run_date: date) -> list[Observation]:
    """Drop a trailing bucket that is still accumulating data.

    This is the single most common source of false alarms in a monitoring system:
    a daily job that runs mid-day compares half a day of data against complete
    days and reports a collapse. Every metric then "drops", the on-call engineer
    learns to ignore the channel, and the alerting is worthless.

    A trailing bucket is dropped when its date is on or after the run date, i.e.
    the period has not finished yet. Historical buckets are never touched, so a
    backfill or a re-run over closed periods is unaffected.
    """
    if len(series) < 2:
        return series
    last_date = parse_bucket(series[-1].bucket)
    if last_date is not None and last_date >= run_date:
        return series[:-1]
    return series


def _build_alerts(
    metric: MetricSpec,
    series: list[Observation],
    dimension_value: str | None,
    run_date: date,
) -> list[Alert]:
    """Score the tail of one series and return in-scope alerts.

    Only the most recent point is judged: a patrol run asks "is right now
    abnormal?", and re-reporting older points would produce duplicate pages for
    the same incident.
    """
    if metric.drop_incomplete_tail:
        series = _drop_incomplete_tail(series, run_date)

    if len(series) < metric.min_samples + 1:
        return []

    window = series[-metric.history_points :]
    latest = window[-1]
    history = window[:-1]

    deviation = detect(
        latest,
        history,
        weekday_aware=metric.weekday_aware,
        min_samples=metric.min_samples,
        min_relative_change=metric.min_relative_change,
    )
    if deviation is None or deviation.score <= 0:
        return []

    if metric.min_absolute_change is not None:
        if abs(deviation.actual - deviation.baseline) < metric.min_absolute_change:
            return []

    if not metric.allows(deviation.direction.value):
        return []

    fingerprint = AlertFingerprint(
        metric=metric.name,
        dimension=metric.dimension,
        dimension_value=dimension_value,
        direction=deviation.direction if metric.direction != "both" else None,
    )
    return [
        Alert(
            fingerprint=fingerprint,
            deviation=deviation,
            metric_label=metric.display(dimension_value),
            dimension_label=dimension_value,
        )
    ]


class PatrolRunner:
    """Execute a patrol plan once and return the outcome."""

    def __init__(
        self,
        plan: PatrolPlan,
        store: AlertStateStore,
        notifier: Any,
        database: str | Path | None = None,
    ) -> None:
        self.plan = plan
        self.store = store
        self.notifier = notifier
        self.database = database or plan.database
        if not self.database:
            raise PatrolError("no database configured: set `database` in the plan or pass it to the runner")

    def run(
        self,
        now: datetime | None = None,
        dry_run: bool = False,
        run_id: str | None = None,
        resume: bool = False,
        limit: int | None = None,
    ) -> PatrolResult:
        """Evaluate the plan, recording per-metric progress so it can resume.

        Args:
            now: Timestamp of the run. Injected for deterministic tests.
            dry_run: When True, nothing is pushed and no state is persisted.
                Resumption is unavailable in dry-run mode by design: there is no
                ledger to resume from.
            run_id: Identifier of this execution. Resuming needs a stable value,
                so callers should pass one (the CLI derives it from the plan
                name and the run date).
            resume: Skip metrics already recorded as ``done`` under ``run_id``.
                Metrics left ``running`` by a killed process are re-evaluated,
                which is what makes an interrupted run safe to restart.
            limit: Evaluate at most this many remaining metrics. Partial runs are
                how a wide plan stays inside a scheduler window, and they are
                also the easiest way to demonstrate resumption.
        """
        run_at = now or datetime.now()
        run_id = run_id or f"patrol:{run_at.date().isoformat()}"
        result = PatrolResult(run_at=run_at, run_id=run_id)
        seen_fingerprints: set[str] = set()
        track = not dry_run

        path = Path(self.database)
        if not path.exists():
            raise PatrolError(f"database not found: {path}")

        done: set[str] = self.store.run_done(run_id) if (resume and track) else set()
        pending: list[MetricSpec] = []
        for metric in self.plan.metrics:
            if metric.name in done:
                result.skipped_metrics += 1
                continue
            pending.append(metric)
        if limit is not None and limit >= 0:
            pending = pending[:limit]

        connection = sqlite3.connect(str(path))
        try:
            for metric in pending:
                if track:
                    self.store.run_begin(run_id, metric.name, run_at)
                try:
                    grouped = _fetch_series(connection, metric)
                    for dimension_value, series in grouped.items():
                        result.series_evaluated += 1
                        for alert in _build_alerts(metric, series, dimension_value, run_at.date()):
                            seen_fingerprints.add(alert.fingerprint.key)
                            self._handle_alert(alert, run_at, dry_run, result)
                except PatrolError as exc:
                    result.errors.append(str(exc))
                    result.failed_metrics += 1
                    if track:
                        self.store.run_finish(run_id, metric.name, run_at, str(exc))
                    continue
                except Exception as exc:  # noqa: BLE001 - one bad metric must not sink the run
                    message = f"{metric.name}: {type(exc).__name__}: {exc}"
                    result.errors.append(message)
                    result.failed_metrics += 1
                    if track:
                        self.store.run_finish(run_id, metric.name, run_at, message)
                    continue
                if track:
                    self.store.run_finish(run_id, metric.name, run_at)
        finally:
            connection.close()

        # Stale resolution asserts that an alert no longer fires. That claim only
        # holds after every metric in the plan was evaluated during this pass: in a
        # limited or resumed run the skipped metrics were never looked at, so
        # resolving their alerts would silently close incidents that are still open.
        if result.skipped_metrics == 0 and limit is None:
            self._resolve_stale(seen_fingerprints, run_at, dry_run, result)
        return result

    def _handle_alert(self, alert: Alert, now: datetime, dry_run: bool, result: PatrolResult) -> None:
        reason = find_suppression(alert, self.plan.suppress)
        if reason is not None:
            result.add_suppressed(alert, f"rule:{reason}")
            if not dry_run:
                self.store.log(
                    alert.fingerprint.key,
                    alert.deviation.bucket,
                    alert.score,
                    alert.severity.value,
                    "suppressed",
                    f"rule:{reason}",
                    now,
                )
            return

        previous = None if dry_run else self.store.get_open(alert.fingerprint.key)
        decision = evaluate(
            alert,
            previous,
            now=now,
            cooldown=self.plan.cooldown,
            min_severity=self.plan.min_severity,
            escalate=self.plan.escalate,
        )

        result.alerts.append(alert)

        if decision.push:
            self.notifier.send(alert, decision.state)
            result.pushed.append(alert)

        if not dry_run:
            self.store.put(decision.state)
            self.store.log(
                alert.fingerprint.key,
                alert.deviation.bucket,
                alert.score,
                alert.severity.value,
                "pushed" if decision.push else "suppressed",
                decision.reason,
                now,
            )

    def _resolve_stale(
        self,
        seen: set[str],
        now: datetime,
        dry_run: bool,
        result: PatrolResult,
    ) -> None:
        """Close fingerprints that did not fire in this run.

        A metric that came back inside its normal band means the incident is over.
        Marking it resolved is what allows the *next* occurrence to page again
        instead of being swallowed by the cooldown.
        """
        if dry_run:
            return
        for state in self.store.list_open():
            if state.fingerprint_key in seen:
                continue
            self.store.put(resolve(state.fingerprint_key, state, now))
            self.store.log(
                state.fingerprint_key,
                state.last_bucket,
                state.last_score,
                "resolved",
                "resolved",
                "no longer anomalous",
                now,
            )
            result.resolved.append(state.fingerprint_key)


def direction_label(direction: Direction) -> str:
    """Human word for a direction, used by notifiers."""
    return "上升" if direction is Direction.RISE else "下降"
