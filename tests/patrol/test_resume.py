"""Resumption tests: the ledger must make an interrupted run safe to restart.

These tests pin three behaviours that are easy to break silently:

* a limited run records what it finished and leaves the rest untouched;
* ``resume`` skips exactly the finished metrics and evaluates the remainder;
* stale resolution never runs on a partial pass, because skipped metrics were
  never looked at and closing their alerts would hide live incidents.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from openchatbi.patrol.notify import NullNotifier
from openchatbi.patrol.plan import PatrolPlan
from openchatbi.patrol.runner import PatrolRunner
from openchatbi.patrol.store import AlertStateStore

NOW = datetime(2026, 5, 31, 9, 0, 0)
RUN_ID = "test-run"


def build_db(path: Path, rows: list[tuple[str, float]]) -> None:
    connection = sqlite3.connect(str(path))
    connection.execute("CREATE TABLE m (bucket TEXT, value REAL)")
    connection.executemany("INSERT INTO m VALUES (?, ?)", rows)
    connection.commit()
    connection.close()


def stable_days(count: int, base: float = 100.0) -> list[tuple[str, float]]:
    jitter = [0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5]
    return [(f"2026-05-{day:02d}", base + jitter[day % len(jitter)]) for day in range(1, count + 1)]


def make_plan(metric_names: list[str]) -> PatrolPlan:
    metrics = [
        {
            "name": name,
            "label": name.upper(),
            "sql": "SELECT bucket, value FROM m ORDER BY bucket",
            "history_points": 10,
            "min_samples": 5,
            "min_relative_change": 0.0,
        }
        for name in metric_names
    ]
    return PatrolPlan.from_dict({"metrics": metrics})


@pytest.fixture
def warehouse(tmp_path: Path) -> Path:
    db = tmp_path / "warehouse.sqlite"
    build_db(db, stable_days(10))
    return db


def test_full_run_marks_every_metric_done(warehouse: Path, tmp_path: Path) -> None:
    plan = make_plan(["a", "b", "c"])
    with AlertStateStore(tmp_path / "state.sqlite") as store:
        runner = PatrolRunner(plan, store, NullNotifier(), database=warehouse)
        result = runner.run(now=NOW, run_id=RUN_ID)
        assert result.skipped_metrics == 0
        assert store.run_states(RUN_ID) == {"a": "done", "b": "done", "c": "done"}


def test_limit_records_partial_progress(warehouse: Path, tmp_path: Path) -> None:
    plan = make_plan(["a", "b", "c", "d"])
    with AlertStateStore(tmp_path / "state.sqlite") as store:
        runner = PatrolRunner(plan, store, NullNotifier(), database=warehouse)
        runner.run(now=NOW, run_id=RUN_ID, limit=2)
        states = store.run_states(RUN_ID)
        assert states == {"a": "done", "b": "done"}
        assert "c" not in states and "d" not in states


def test_resume_skips_finished_metrics(warehouse: Path, tmp_path: Path) -> None:
    plan = make_plan(["a", "b", "c", "d"])
    with AlertStateStore(tmp_path / "state.sqlite") as store:
        runner = PatrolRunner(plan, store, NullNotifier(), database=warehouse)
        first = runner.run(now=NOW, run_id=RUN_ID, limit=2)
        second = runner.run(now=NOW, run_id=RUN_ID, resume=True)

        # Only the two remaining metrics are evaluated the second time.
        assert second.skipped_metrics == 2
        assert second.series_evaluated == 2
        # Earlier run's series count is preserved in its own result object.
        assert first.series_evaluated == 2
        assert store.run_states(RUN_ID) == {"a": "done", "b": "done", "c": "done", "d": "done"}


def test_resume_after_partial_run_covers_whole_plan(warehouse: Path, tmp_path: Path) -> None:
    plan = make_plan(["a", "b", "c"])
    with AlertStateStore(tmp_path / "state.sqlite") as store:
        runner = PatrolRunner(plan, store, NullNotifier(), database=warehouse)
        first = runner.run(now=NOW, run_id=RUN_ID, limit=1)
        second = runner.run(now=NOW, run_id=RUN_ID, resume=True)
        assert first.series_evaluated + second.series_evaluated == 3


def test_failed_metric_is_recorded_and_retried(
    warehouse: Path, tmp_path: Path
) -> None:
    good = {
        "name": "good",
        "label": "GOOD",
        "sql": "SELECT bucket, value FROM m ORDER BY bucket",
        "history_points": 10,
        "min_samples": 5,
        "min_relative_change": 0.0,
    }
    broken = dict(good, name="broken", sql="SELECT bucket, value FROM missing_table")
    plan = PatrolPlan.from_dict({"metrics": [good, broken]})

    with AlertStateStore(tmp_path / "state.sqlite") as store:
        runner = PatrolRunner(plan, store, NullNotifier(), database=warehouse)
        result = runner.run(now=NOW, run_id=RUN_ID)

        assert result.failed_metrics == 1
        assert store.run_states(RUN_ID) == {"good": "done", "broken": "failed"}

        # A failed metric is not "done", so resume must retry it rather than skip it.
        resumed = runner.run(now=NOW, run_id=RUN_ID, resume=True)
        assert resumed.skipped_metrics == 1
        assert resumed.failed_metrics == 1


def test_stale_resolution_skipped_on_partial_run(warehouse: Path, tmp_path: Path) -> None:
    """A limited or resumed pass must not close alerts it never evaluated."""
    plan = make_plan(["a", "b", "c"])
    with AlertStateStore(tmp_path / "state.sqlite") as store:
        runner = PatrolRunner(plan, store, NullNotifier(), database=warehouse)
        limited = runner.run(now=NOW, run_id=RUN_ID, limit=1)
        assert limited.resolved == []

        resumed = runner.run(now=NOW, run_id=RUN_ID, resume=True)
        assert resumed.resolved == []


def test_ledger_is_per_run_id(warehouse: Path, tmp_path: Path) -> None:
    plan = make_plan(["a", "b"])
    with AlertStateStore(tmp_path / "state.sqlite") as store:
        runner = PatrolRunner(plan, store, NullNotifier(), database=warehouse)
        runner.run(now=NOW, run_id="run-1", limit=1)
        runner.run(now=NOW, run_id="run-2", limit=1)

        assert store.run_states("run-1") == {"a": "done"}
        assert store.run_states("run-2") == {"a": "done"}
        # A fresh run id must not inherit another run's progress.
        fresh = runner.run(now=NOW, run_id="run-3")
        assert fresh.skipped_metrics == 0
        assert fresh.series_evaluated == 2
