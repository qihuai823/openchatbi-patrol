"""End-to-end tests for the patrol runner against a real SQLite file."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from openchatbi.patrol.models import Direction
from openchatbi.patrol.notify import NullNotifier
from openchatbi.patrol.plan import PatrolPlan, PlanError
from openchatbi.patrol.runner import PatrolError, PatrolRunner
from openchatbi.patrol.store import AlertStateStore

NOW = datetime(2026, 5, 31, 9, 0, 0)


def build_db(path: Path, rows: list[tuple[str, float]], dimension: bool = False) -> None:
    """Create a tiny warehouse with a ``m`` table holding one metric series."""
    connection = sqlite3.connect(str(path))
    if dimension:
        connection.execute("CREATE TABLE m (bucket TEXT, category TEXT, value REAL)")
        connection.executemany(
            "INSERT INTO m VALUES (?, ?, ?)",
            [(bucket, category, value) for bucket, category, value in rows],  # type: ignore[misc]
        )
    else:
        connection.execute("CREATE TABLE m (bucket TEXT, value REAL)")
        connection.executemany("INSERT INTO m VALUES (?, ?)", rows)
    connection.commit()
    connection.close()


def stable_days(count: int, base: float = 100.0) -> list[tuple[str, float]]:
    """A flat-ish series, deterministic so assertions are stable."""
    jitter = [0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5]
    return [(f"2026-05-{day:02d}", base + jitter[day % len(jitter)]) for day in range(1, count + 1)]


def make_plan(**metric_overrides: object) -> PatrolPlan:
    metric = {
        "name": "gmv",
        "label": "GMV",
        "sql": "SELECT bucket, value FROM m ORDER BY bucket",
        "history_points": 10,
        "min_samples": 5,
        "min_relative_change": 0.0,
    }
    metric.update(metric_overrides)
    return PatrolPlan.from_dict({"metrics": [metric]})


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    return tmp_path


def test_spike_produces_a_pushed_alert(workdir: Path) -> None:
    db = workdir / "wh.sqlite"
    build_db(db, [*stable_days(11), ("2026-05-12", 900.0)])

    notifier = NullNotifier()
    with AlertStateStore(workdir / "state.sqlite") as store:
        result = PatrolRunner(make_plan(), store, notifier, database=db).run(now=NOW)

    assert result.series_evaluated == 1
    assert result.detected == 1
    assert len(result.pushed) == 1
    assert len(notifier.sent) == 1
    alert = notifier.sent[0]
    assert alert.fingerprint.key == "gmv"
    assert alert.deviation.direction is Direction.RISE
    assert alert.score > 0.8
    assert "GMV" in alert.summary()


def test_second_run_inside_cooldown_is_suppressed(workdir: Path) -> None:
    db = workdir / "wh.sqlite"
    build_db(db, [*stable_days(11), ("2026-05-12", 900.0)])

    notifier = NullNotifier()
    with AlertStateStore(workdir / "state.sqlite") as store:
        runner = PatrolRunner(make_plan(), store, notifier, database=db)
        first = runner.run(now=NOW)
        second = runner.run(now=NOW + timedelta(hours=1))

    assert len(first.pushed) == 1
    assert len(second.pushed) == 0
    assert len(second.suppressed) == 0  # it reached the dedup gate, not a suppression rule
    assert len(notifier.sent) == 1
    with AlertStateStore(workdir / "state.sqlite") as store:
        state = store.get("gmv")
        assert state is not None
        assert state.push_count == 1


def test_dry_run_pushes_nothing_and_persists_nothing(workdir: Path) -> None:
    db = workdir / "wh.sqlite"
    build_db(db, [*stable_days(11), ("2026-05-12", 900.0)])

    notifier = NullNotifier()
    with AlertStateStore(workdir / "state.sqlite") as store:
        result = PatrolRunner(make_plan(), store, notifier, database=db).run(now=NOW, dry_run=True)

    assert result.series_evaluated == 1
    assert result.detected == 1
    assert len(result.pushed) == 1
    assert len(notifier.sent) == 1
    with AlertStateStore(workdir / "state.sqlite") as store:
        assert store.get("gmv") is None


def flat_days(count: int, base: float = 100.0) -> list[tuple[str, float]]:
    """A perfectly constant series: no deviation is possible."""
    return [(f"2026-05-{day:02d}", base) for day in range(1, count + 1)]


def test_incomplete_trailing_bucket_is_ignored(workdir: Path) -> None:
    """A mid-day run must not treat half a day as a collapse.

    This is the classic false-alarm generator: the newest bucket is still
    accumulating, so it looks like every metric fell off a cliff.
    """
    db = workdir / "wh.sqlite"
    # Data runs up to the run date itself, with the final day only partially loaded.
    build_db(db, [*stable_days(11), ("2026-05-31", 8.0)])

    notifier = NullNotifier()
    with AlertStateStore(workdir / "state.sqlite") as store:
        result = PatrolRunner(make_plan(), store, notifier, database=db).run(now=NOW)

    # The partial 2026-05-31 bucket is dropped, so the run judges 2026-05-11
    # instead. Ordinary jitter may still register a low-severity deviation, but
    # nothing about the half-loaded day may reach the result or the channel.
    assert result.pushed == []
    assert notifier.sent == []
    assert all(alert.deviation.bucket != "2026-05-31" for alert in result.alerts)


def test_incomplete_tail_can_be_disabled(workdir: Path) -> None:
    db = workdir / "wh.sqlite"
    build_db(db, [*stable_days(11), ("2026-05-31", 8.0)])

    notifier = NullNotifier()
    plan = make_plan(drop_incomplete_tail=False)
    with AlertStateStore(workdir / "state.sqlite") as store:
        result = PatrolRunner(plan, store, notifier, database=db).run(now=NOW)

    assert result.detected == 1
    assert result.alerts[0].deviation.direction is Direction.DROP


def test_closed_historical_buckets_are_never_dropped(workdir: Path) -> None:
    """A re-run over a closed period must still evaluate the last bucket."""
    db = workdir / "wh.sqlite"
    build_db(db, [*stable_days(11), ("2026-05-12", 900.0)])

    notifier = NullNotifier()
    with AlertStateStore(workdir / "state.sqlite") as store:
        result = PatrolRunner(make_plan(), store, notifier, database=db).run(now=NOW)

    assert result.detected == 1
    assert result.alerts[0].deviation.bucket == "2026-05-12"


def test_flat_metric_raises_nothing(workdir: Path) -> None:
    db = workdir / "wh.sqlite"
    build_db(db, flat_days(12))

    notifier = NullNotifier()
    with AlertStateStore(workdir / "state.sqlite") as store:
        result = PatrolRunner(make_plan(), store, notifier, database=db).run(now=NOW)

    assert result.series_evaluated == 1
    assert result.detected == 0
    assert notifier.sent == []


def test_low_severity_deviation_is_detected_but_not_pushed(workdir: Path) -> None:
    # Ordinary day-to-day jitter: a deviation exists, but it is below the push
    # floor. It must still show up in the result so operators can see the system
    # looked at the metric rather than skipping it.
    db = workdir / "wh.sqlite"
    build_db(db, stable_days(12))

    notifier = NullNotifier()
    with AlertStateStore(workdir / "state.sqlite") as store:
        result = PatrolRunner(make_plan(), store, notifier, database=db).run(now=NOW)

    assert result.series_evaluated == 1
    assert result.detected == 1
    assert result.alerts[0].severity.rank < 2  # below HIGH
    assert result.pushed == []
    assert notifier.sent == []


def test_direction_scope_filters_out_unwanted_moves(workdir: Path) -> None:
    db = workdir / "wh.sqlite"
    # A large RISE on a metric declared as drop-only must not alert.
    build_db(db, [*stable_days(11), ("2026-05-12", 900.0)])

    notifier = NullNotifier()
    with AlertStateStore(workdir / "state.sqlite") as store:
        result = PatrolRunner(make_plan(direction="drop"), store, notifier, database=db).run(now=NOW)

    assert result.detected == 0
    assert notifier.sent == []


def test_dimension_produces_one_alert_per_series(workdir: Path) -> None:
    db = workdir / "wh.sqlite"
    rows: list[tuple[str, str, float]] = []
    for bucket, value in stable_days(11):
        rows.append((bucket, "electronics", value))
        rows.append((bucket, "kitchen", value))
    rows.append(("2026-05-12", "electronics", 900.0))
    rows.append(("2026-05-12", "kitchen", 100.0))
    build_db(db, rows, dimension=True)  # type: ignore[arg-type]

    notifier = NullNotifier()
    plan = make_plan(sql="SELECT bucket, category, value FROM m ORDER BY bucket", dimension="category")
    with AlertStateStore(workdir / "state.sqlite") as store:
        result = PatrolRunner(plan, store, notifier, database=db).run(now=NOW)

    assert result.series_evaluated == 2  # electronics + kitchen
    assert result.detected == 1
    assert len(notifier.sent) == 1
    assert notifier.sent[0].fingerprint.key == "gmv|category=electronics"
    assert "electronics" in notifier.sent[0].summary()


def test_suppression_rule_blocks_the_push_but_keeps_the_sighting(workdir: Path) -> None:
    db = workdir / "wh.sqlite"
    build_db(db, [*stable_days(11), ("2026-05-12", 900.0)])

    notifier = NullNotifier()
    plan = PatrolPlan.from_dict(
        {
            "metrics": [
                {
                    "name": "gmv",
                    "sql": "SELECT bucket, value FROM m ORDER BY bucket",
                    "history_points": 10,
                    "min_samples": 5,
                    "min_relative_change": 0.0,
                }
            ],
            "suppress": [{"reason": "known migration window", "bucket_glob": "2026-05-12"}],
        }
    )
    with AlertStateStore(workdir / "state.sqlite") as store:
        result = PatrolRunner(plan, store, notifier, database=db).run(now=NOW)

    assert notifier.sent == []
    assert len(result.suppressed) == 1
    assert "known migration window" in result.suppressed[0][1]
    with AlertStateStore(workdir / "state.sqlite") as store:
        log = store.recent_log()
        assert log[0]["decision"] == "suppressed"
        assert "known migration window" in log[0]["reason"]


def test_missing_bucket_column_is_reported_as_an_error(workdir: Path) -> None:
    db = workdir / "wh.sqlite"
    build_db(db, stable_days(12))

    notifier = NullNotifier()
    plan = make_plan(sql="SELECT value FROM m ORDER BY value")
    with AlertStateStore(workdir / "state.sqlite") as store:
        result = PatrolRunner(plan, store, notifier, database=db).run(now=NOW)

    assert result.detected == 0
    assert any("must return a `bucket` column" in error for error in result.errors)


def test_declared_dimension_missing_from_query_is_reported(workdir: Path) -> None:
    db = workdir / "wh.sqlite"
    build_db(db, stable_days(12))

    notifier = NullNotifier()
    plan = make_plan(dimension="category")
    with AlertStateStore(workdir / "state.sqlite") as store:
        result = PatrolRunner(plan, store, notifier, database=db).run(now=NOW)

    assert any("dimension `category` declared" in error for error in result.errors)


def test_alert_is_resolved_when_the_metric_returns_to_normal(workdir: Path) -> None:
    db = workdir / "wh.sqlite"
    build_db(db, [*stable_days(11), ("2026-05-12", 900.0)])

    notifier = NullNotifier()
    with AlertStateStore(workdir / "state.sqlite") as store:
        runner = PatrolRunner(make_plan(), store, notifier, database=db)
        runner.run(now=NOW)
        assert store.get_open("gmv") is not None

        # The spike day is replaced by a normal value: the incident is over.
        connection = sqlite3.connect(str(db))
        connection.execute("UPDATE m SET value = 100.0 WHERE bucket = '2026-05-12'")
        connection.commit()
        connection.close()

        second = runner.run(now=NOW + timedelta(days=1))

    assert second.resolved == ["gmv"]
    with AlertStateStore(workdir / "state.sqlite") as store:
        assert store.get_open("gmv") is None
        assert store.get("gmv").status == "resolved"  # type: ignore[union-attr]


def test_missing_database_raises(workdir: Path) -> None:
    with AlertStateStore(workdir / "state.sqlite") as store:
        runner = PatrolRunner(make_plan(), store, NullNotifier(), database=workdir / "nope.sqlite")
        with pytest.raises(PatrolError, match="database not found"):
            runner.run(now=NOW)


def test_runner_requires_a_database(workdir: Path) -> None:
    with AlertStateStore(workdir / "state.sqlite") as store:
        with pytest.raises(PatrolError, match="no database configured"):
            PatrolRunner(make_plan(), store, NullNotifier())


def test_plan_error_type_is_a_value_error() -> None:
    assert issubclass(PlanError, ValueError)
