"""Node-level checkpointed execution for the patrol subsystem.

Why a graph instead of the straight loop in :mod:`openchatbi.patrol.runner`:
the loop records progress per *metric*, which is enough to skip finished work but
cannot resume *inside* a run — if the process dies between fetching a metric and
recording its alerts, that metric starts over from the top.

The graph below makes each metric its own node task and checkpoints the whole
state through ``SqliteSaver``. Restarting with the same ``thread_id`` lets
LangGraph replay only the tasks that never finished, so the resume granularity is
a node task rather than a metric boundary.

The sequential runner is deliberately kept as the default: it has no LangGraph
dependency and stays the cheapest way to run patrol from cron. This module is the
opt-in engine (``--engine graph``) used when durability matters.

Side effects (state writes, notifications) live in their own nodes and are made
idempotent through the audit log, because a checkpointed node may legitimately be
re-executed after a crash.
"""

from __future__ import annotations

import operator
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from openchatbi.patrol.dedup import evaluate as evaluate_decision
from openchatbi.patrol.dedup import find_suppression, resolve as resolve_state
from openchatbi.patrol.models import (
    Alert,
    AlertFingerprint,
    AlertState,
    Deviation,
    Direction,
    Observation,
    PatrolResult,
    Severity,
)
from openchatbi.patrol.runner import PatrolError, PatrolRunner, _build_alerts, _fetch_series
from openchatbi.patrol.store import AlertStateStore


class PatrolState(TypedDict, total=False):
    """Serializable graph state. Everything here must survive a msgpack round-trip."""

    run_at: str
    pending: list[str]
    evaluated: Annotated[list[str], operator.add]
    alerts: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]
    series: Annotated[int, operator.add]
    pushed: Annotated[list[dict[str, Any]], operator.add]
    suppressed: Annotated[list[dict[str, Any]], operator.add]
    resolved: Annotated[list[str], operator.add]
    summary: dict[str, Any]


def _alert_to_dict(alert: Alert) -> dict[str, Any]:
    return {
        "metric": alert.fingerprint.metric,
        "dimension": alert.fingerprint.dimension,
        "dimension_value": alert.fingerprint.dimension_value,
        "direction": alert.fingerprint.direction.value if alert.fingerprint.direction else None,
        "metric_label": alert.metric_label,
        "dimension_label": alert.dimension_label,
        "bucket": alert.deviation.bucket,
        "actual": alert.deviation.actual,
        "baseline": alert.deviation.baseline,
        "scale": alert.deviation.scale,
        "z": alert.deviation.z,
        "score": alert.deviation.score,
        "severity": alert.severity.value,
    }


def _alert_from_dict(payload: dict[str, Any]) -> Alert:
    direction = payload.get("direction")
    return Alert(
        fingerprint=AlertFingerprint(
            metric=payload["metric"],
            dimension=payload.get("dimension"),
            dimension_value=payload.get("dimension_value"),
            direction=Direction(direction) if direction else None,
        ),
        deviation=Deviation(
            bucket=payload["bucket"],
            actual=payload["actual"],
            baseline=payload["baseline"],
            scale=payload["scale"],
            z=payload["z"],
            score=payload["score"],
            direction=Direction(direction) if direction else Direction.DROP,
        ),
        metric_label=payload.get("metric_label", ""),
        dimension_label=payload.get("dimension_label"),
    )


class PatrolGraphRunner:
    """Run a patrol plan as a checkpointed LangGraph state machine.

    Args:
        plan: Parsed patrol plan.
        store: Alert state store (dedup/cooldown state and audit log).
        notifier: Anything with ``send(alert, state)``.
        database: Warehouse path; defaults to the plan's.
        checkpoint_path: SQLite file holding graph checkpoints. Reusing the same
            ``run_id`` against the same file is what makes resume work.
    """

    def __init__(
        self,
        plan: Any,
        store: AlertStateStore,
        notifier: Any,
        database: str | Path | None = None,
        checkpoint_path: str | Path = "patrol_checkpoints.sqlite",
    ) -> None:
        self.plan = plan
        self.store = store
        self.notifier = notifier
        self.checkpoint_path = Path(checkpoint_path)
        # Reuse the sequential runner's evaluation logic so both engines cannot
        # drift apart in how a metric becomes an alert.
        self.runner = PatrolRunner(plan, store, notifier, database=database)

    # -- nodes -------------------------------------------------------------
    def _prepare(self, state: PatrolState) -> dict[str, Any]:
        run_at = datetime.fromisoformat(state["run_at"])
        done = set(state.get("evaluated", []))
        pending = [metric.name for metric in self.plan.metrics if metric.name not in done]
        limit = self._limit
        if limit is not None and limit >= 0:
            pending = pending[:limit]
        return {"pending": pending, "evaluated": [], "run_at": state["run_at"]}

    def _fanout(self, state: PatrolState) -> list[Send] | str:
        pending = state.get("pending", [])
        if not pending:
            return "finish"
        return [Send("evaluate", {"metric": name, "run_at": state["run_at"]}) for name in pending]

    def _evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        metric = self._metric_by_name[payload["metric"]]
        run_at = datetime.fromisoformat(payload["run_at"])
        # The ledger is written by the graph engine too, so `--ledger` works the
        # same way regardless of engine. It is an audit view, not the resume
        # source — resumption is the checkpointer's job here.
        track = not self._dry_run
        if track:
            self.store.run_begin(self._run_id, metric.name, run_at)
        connection = self._connect()
        try:
            grouped = _fetch_series(connection, metric)
        except PatrolError as exc:
            if track:
                self.store.run_finish(self._run_id, metric.name, run_at, str(exc))
            return {"errors": [str(exc)], "evaluated": [metric.name]}
        except Exception as exc:  # noqa: BLE001 - one metric must not sink the run
            message = f"{metric.name}: {type(exc).__name__}: {exc}"
            if track:
                self.store.run_finish(self._run_id, metric.name, run_at, message)
            return {"errors": [message], "evaluated": [metric.name]}
        finally:
            connection.close()

        alerts: list[dict[str, Any]] = []
        series_count = 0
        for dimension_value, series in grouped.items():
            series_count += 1
            for alert in _build_alerts(metric, series, dimension_value, run_at.date()):
                alerts.append(_alert_to_dict(alert))
        if track:
            self.store.run_finish(self._run_id, metric.name, run_at)
        return {"series": series_count, "alerts": alerts, "evaluated": [metric.name]}

    def _decide(self, state: PatrolState) -> dict[str, Any]:
        run_at = datetime.fromisoformat(state["run_at"])
        pushed: list[dict[str, Any]] = []
        suppressed: list[dict[str, Any]] = []
        for payload in state.get("alerts", []):
            alert = _alert_from_dict(payload)
            reason = find_suppression(alert, self.plan.suppress)
            if reason is not None:
                suppressed.append({**payload, "reason": f"rule:{reason}"})
                continue
            previous = None if self._dry_run else self.store.get_open(alert.fingerprint.key)
            decision = evaluate_decision(
                alert,
                previous,
                now=run_at,
                cooldown=self.plan.cooldown,
                min_severity=self.plan.min_severity,
                escalate=self.plan.escalate,
            )
            if decision.push:
                pushed.append(
                    {
                        **payload,
                        "action": "push",
                        "reason": decision.reason,
                        "next_state": decision.state,
                    }
                )
            else:
                suppressed.append({**payload, "action": "suppress", "reason": decision.reason})
        return {"pushed": pushed, "suppressed": suppressed}

    def _commit(self, state: PatrolState) -> dict[str, Any]:
        """Persist alert lifecycle state. Upserts, so replay is harmless."""
        if self._dry_run:
            return {}
        run_at = datetime.fromisoformat(state["run_at"])
        for item in state.get("pushed", []):
            self.store.put(item["next_state"])
        for item in state.get("suppressed", []):
            self.store.log(
                item["metric"] if item.get("dimension") is None else f"{item['metric']}|{item['dimension_value']}",
                item["bucket"],
                item["score"],
                item["severity"],
                "suppressed",
                item["reason"],
                run_at,
            )
        return {}

    def _notify(self, state: PatrolState) -> dict[str, Any]:
        """Push alerts, skipping any that the audit log says were already sent."""
        if self._dry_run:
            return {}
        run_at = datetime.fromisoformat(state["run_at"])
        for item in state.get("pushed", []):
            alert = _alert_from_dict(item)
            if self._already_notified(alert):
                continue
            self.notifier.send(alert, item["next_state"])
            self.store.log(
                alert.fingerprint.key,
                alert.deviation.bucket,
                alert.score,
                alert.severity.value,
                "pushed",
                item["reason"],
                run_at,
            )
        return {}

    def _finish(self, state: PatrolState) -> dict[str, Any]:
        run_at = datetime.fromisoformat(state["run_at"])
        result = PatrolResult(
            run_at=run_at,
            run_id=self._run_id,
            series_evaluated=state.get("series", 0),
            skipped_metrics=max(0, len(self.plan.metrics) - len(state.get("evaluated", []))),
            failed_metrics=len(state.get("errors", [])),
        )
        result.alerts = [_alert_from_dict(item) for item in state.get("alerts", [])]
        result.pushed = [_alert_from_dict(item) for item in state.get("pushed", [])]
        result.suppressed = [
            (_alert_from_dict(item), item.get("reason", "")) for item in state.get("suppressed", [])
        ]
        result.errors = list(state.get("errors", []))
        result.resolved = list(state.get("resolved", []))
        return {"summary": {"errors": len(result.errors)}}

    # -- helpers -----------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.runner.database))

    def _already_notified(self, alert: Alert) -> bool:
        recent = self.store.recent_log(limit=200)
        key = alert.fingerprint.key
        bucket = alert.deviation.bucket
        return any(
            row["fingerprint_key"] == key and row["bucket"] == bucket and row["decision"] == "pushed"
            for row in recent
        )

    def _build_graph(self, checkpointer: Any) -> Any:
        graph = StateGraph(PatrolState)
        graph.add_node("prepare", self._prepare)
        graph.add_node("evaluate", self._evaluate)
        graph.add_node("decide", self._decide)
        graph.add_node("commit", self._commit)
        graph.add_node("notify", self._notify)
        graph.add_node("finish", self._finish)
        graph.set_entry_point("prepare")
        graph.add_conditional_edges("prepare", self._fanout, ["evaluate", "finish"])
        graph.add_edge("evaluate", "decide")
        graph.add_edge("decide", "commit")
        graph.add_edge("commit", "notify")
        graph.add_edge("notify", "finish")
        graph.add_edge("finish", END)
        return graph.compile(checkpointer=checkpointer)

    # -- public API --------------------------------------------------------
    def run(
        self,
        now: datetime | None = None,
        dry_run: bool = False,
        run_id: str = "patrol",
        resume: bool = False,
        limit: int | None = None,
        concurrency: int | None = None,
    ) -> PatrolResult:
        run_at = now or datetime.now()
        self._run_id = run_id
        self._limit = limit
        self._dry_run = dry_run
        self._metric_by_name = {metric.name: metric for metric in self.plan.metrics}

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.checkpoint_path), check_same_thread=False)
        saver = SqliteSaver(connection)
        graph = self._build_graph(saver)
        config: dict[str, Any] = {"configurable": {"thread_id": run_id}}
        # A plan can fan out to dozens of metric nodes at once. Against a real
        # warehouse that many concurrent queries is exactly what you do not want,
        # so the caller can cap it; 1 also makes interruption easy to reproduce.
        if concurrency:
            config["max_concurrency"] = int(concurrency)

        try:
            snapshot = graph.get_state(config)
            has_pending = bool(snapshot.next)
            if resume and has_pending:
                graph.invoke(None, config)
            else:
                graph.invoke({"run_at": run_at.isoformat()}, config)
            final = graph.get_state(config).values
        finally:
            connection.close()

        return self._result_from_state(final, run_at)

    def _result_from_state(self, values: dict[str, Any], run_at: datetime) -> PatrolResult:
        result = PatrolResult(
            run_at=run_at,
            run_id=self._run_id,
            series_evaluated=values.get("series", 0),
            skipped_metrics=max(0, len(self.plan.metrics) - len(values.get("evaluated", []))),
            failed_metrics=len(values.get("errors", [])),
        )
        result.alerts = [_alert_from_dict(item) for item in values.get("alerts", [])]
        result.pushed = [_alert_from_dict(item) for item in values.get("pushed", [])]
        result.suppressed = [
            (_alert_from_dict(item), item.get("reason", "")) for item in values.get("suppressed", [])
        ]
        result.errors = list(values.get("errors", []))
        result.resolved = list(values.get("resolved", []))
        return result
