"""Node-level checkpointing tests for the graph engine.

The point of the graph engine over the sequential runner is granularity: each
metric is its own node task, and ``SqliteSaver`` records which tasks finished. So a
crash *inside* one metric must not force the already-finished metrics to run again.

These tests pin that, plus the idempotency the extra re-executions make necessary.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

import openchatbi.patrol.graph as graph_module
from openchatbi.patrol.graph import PatrolGraphRunner
from openchatbi.patrol.notify import NullNotifier
from openchatbi.patrol.plan import PatrolPlan
from openchatbi.patrol.runner import PatrolRunner
from openchatbi.patrol.store import AlertStateStore

NOW = datetime(2026, 5, 31, 9, 0, 0)
RUN_ID = "graph-run"


class SimulatedCrash(BaseException):
    """Stand-in for the process dying: not an Exception, so node handlers cannot catch it."""


def build_db(path: Path, rows: list[tuple[str, float]]) -> None:
    connection = sqlite3.connect(str(path))
    connection.execute("CREATE TABLE m (bucket TEXT, value REAL)")
    connection.executemany("INSERT INTO m VALUES (?, ?)", rows)
    connection.commit()
    connection.close()


def stable_days(count: int, base: float = 100.0) -> list[tuple[str, float]]:
    jitter = [0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5]
    return [(f"2026-05-{day:02d}", base + jitter[day % len(jitter)]) for day in range(1, count + 1)]


def make_plan(names: list[str]) -> PatrolPlan:
    metrics = [
        {
            "name": name,
            "label": name.upper(),
            "sql": "SELECT bucket, value FROM m ORDER BY bucket",
            "history_points": 10,
            "min_samples": 5,
            "min_relative_change": 0.0,
        }
        for name in names
    ]
    return PatrolPlan.from_dict({"metrics": metrics})


@pytest.fixture
def warehouse(tmp_path: Path) -> Path:
    db = tmp_path / "warehouse.sqlite"
    build_db(db, stable_days(10))
    return db


def test_graph_matches_sequential_series_count(warehouse: Path, tmp_path: Path) -> None:
    plan = make_plan(["a", "b", "c"])
    with AlertStateStore(tmp_path / "state.sqlite") as store:
        sequential = PatrolRunner(plan, store, NullNotifier(), database=warehouse).run(now=NOW, run_id="seq")
        graph_result = PatrolGraphRunner(
            plan, store, NullNotifier(), database=warehouse, checkpoint_path=tmp_path / "cp.sqlite"
        ).run(now=NOW, run_id=RUN_ID)
        assert graph_result.series_evaluated == sequential.series_evaluated


def test_partial_run_then_resume_skips_finished_metric_nodes(
    warehouse: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """已完成指标的节点任务不重跑。

    真实崩溃很难在单元测试里稳定复现（LangGraph 的 Send 分支在线程池里跑，
    节点异常会变成未处理线程异常），所以这里用 ``limit`` 造出同样的状态：
    第一轮只跑一部分，检查点里留下已完成的任务；第二轮 resume 时，
    取数函数只应看到剩下的那几个指标。
    """
    plan = make_plan(["a", "b", "c", "d", "e"])
    calls: list[str] = []
    real_fetch = graph_module._fetch_series

    def recording_fetch(connection, metric):
        calls.append(metric.name)
        return real_fetch(connection, metric)

    monkeypatch.setattr(graph_module, "_fetch_series", recording_fetch)

    checkpoint = tmp_path / "cp.sqlite"
    with AlertStateStore(tmp_path / "state.sqlite") as store:
        runner = PatrolGraphRunner(
            plan, store, NullNotifier(), database=warehouse, checkpoint_path=checkpoint
        )
        first = runner.run(now=NOW, run_id=RUN_ID, limit=2)
        assert first.series_evaluated == 2
        assert calls == ["a", "b"]

        calls.clear()
        second = runner.run(now=NOW, run_id=RUN_ID, resume=True)

        # Only the incomplete metrics run again: a and b were checkpointed.
        assert "a" not in calls
        assert "b" not in calls
        assert "c" in calls
        assert set(calls) <= {"c", "d", "e"}
        # 累计值来自检查点：第一轮 2 条 + 本轮 3 条 = 5；本轮真正取数的只有 c/d/e。
        assert second.series_evaluated == 5


def test_resume_without_checkpoint_behaves_like_fresh_run(warehouse: Path, tmp_path: Path) -> None:
    plan = make_plan(["a", "b"])
    with AlertStateStore(tmp_path / "state.sqlite") as store:
        runner = PatrolGraphRunner(
            plan, store, NullNotifier(), database=warehouse, checkpoint_path=tmp_path / "cp.sqlite"
        )
        result = runner.run(now=NOW, run_id="never-started", resume=True)
        assert result.series_evaluated == 2


def test_limit_applies_to_pending_metrics(warehouse: Path, tmp_path: Path) -> None:
    plan = make_plan(["a", "b", "c", "d"])
    with AlertStateStore(tmp_path / "state.sqlite") as store:
        runner = PatrolGraphRunner(
            plan, store, NullNotifier(), database=warehouse, checkpoint_path=tmp_path / "cp.sqlite"
        )
        result = runner.run(now=NOW, run_id=RUN_ID, limit=2)
        assert result.series_evaluated == 2


def test_notifications_are_not_duplicated_on_replay(warehouse: Path, tmp_path: Path) -> None:
    """A checkpointed run may replay the notify node; the audit log must stop double sends."""
    plan = make_plan(["a"])
    sent: list[str] = []

    class Recorder:
        def send(self, alert, state):  # noqa: ANN001 - mirrors the notifier protocol
            sent.append(alert.fingerprint.key)

    with AlertStateStore(tmp_path / "state.sqlite") as store:
        runner = PatrolGraphRunner(
            plan, store, Recorder(), database=warehouse, checkpoint_path=tmp_path / "cp.sqlite"
        )
        runner.run(now=NOW, run_id=RUN_ID)
        first_round = len(sent)
        # Same run id, forcing the notify node to execute again.
        runner.run(now=NOW, run_id=RUN_ID)
        assert len(sent) == first_round
