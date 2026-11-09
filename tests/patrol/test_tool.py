"""Tests for the read-only `patrol_status` agent tool."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from openchatbi.patrol import tool as patrol_tool
from openchatbi.patrol.models import AlertState, Severity
from openchatbi.patrol.plan import PatrolPlan
from openchatbi.patrol.store import AlertStateStore
from openchatbi.patrol.tool import _display_name, _split_fingerprint, format_status, patrol_status

TS = datetime(2026, 5, 31, 21, 0)


def _state(key: str, score: float, *, bucket: str = "2026-05-30", pushed: int = 0) -> AlertState:
    return AlertState(
        fingerprint_key=key,
        first_seen=TS,
        last_seen=TS,
        last_pushed=TS if pushed else None,
        push_count=pushed,
        last_score=score,
        last_bucket=bucket,
    )


def _plan(min_severity: str = "medium") -> PatrolPlan:
    return PatrolPlan.from_dict(
        {
            "defaults": {"min_severity": min_severity},
            "metrics": [
                {"name": "gmv", "label": "成交总额 GMV", "sql": "SELECT 1"},
                {"name": "gmv_by_category", "label": "分品类 GMV", "dimension": "category", "sql": "SELECT 1"},
            ],
        }
    )


# ---------------------------------------------------------------------------
# fingerprint helpers
# ---------------------------------------------------------------------------


def test_split_fingerprint_with_and_without_dimension() -> None:
    assert _split_fingerprint("gmv|category=office") == ("gmv", "category", "office")
    assert _split_fingerprint("order_count") == ("order_count", None, None)


def test_display_name_uses_the_plan_label() -> None:
    plan = _plan()
    assert _display_name("gmv_by_category|category=office", plan) == "分品类 GMV (office)"
    assert _display_name("gmv", plan) == "成交总额 GMV"


def test_display_name_falls_back_when_the_plan_lacks_the_metric() -> None:
    assert _display_name("unknown_metric|x=y", _plan()) == "unknown_metric (y)"
    assert _display_name("unknown_metric", None) == "unknown_metric"


# ---------------------------------------------------------------------------
# format_status: the severity floor is the whole point
# ---------------------------------------------------------------------------


def test_below_threshold_observations_are_hidden_by_default() -> None:
    states = [_state("gmv|category=office", 0.42, pushed=1)] + [
        _state(f"gmv_by_category|category={name}", 0.1) for name in ("beauty", "book", "clothes")
    ]

    text = format_status(states, _plan(), [])

    assert "1 open alert(s) at or above the medium threshold" in text
    assert "office" in text
    assert "beauty" not in text
    assert "3 lower-severity observation(s) hidden" in text


def test_include_below_threshold_shows_everything() -> None:
    states = [_state("gmv", 0.42, pushed=1), _state("gmv_by_category|category=beauty", 0.1)]

    text = format_status(states, _plan(), [], include_below_threshold=True)

    assert "2 open alert(s)" in text
    assert "threshold" not in text.split("\n")[0]
    assert "beauty" in text
    assert "hidden" not in text


def test_alerts_are_sorted_by_severity_then_score() -> None:
    states = [
        _state("gmv_by_category|category=low", 0.10),
        _state("gmv_by_category|category=high", 0.95),
        _state("gmv_by_category|category=mid", 0.45),
    ]

    text = format_status(states, _plan(), [], include_below_threshold=True)
    body = [line for line in text.split("\n") if line.startswith("[")]

    assert [Severity.from_score(s.last_score).value for s in (states[1], states[2], states[0])] == [
        "critical" if Severity.from_score(0.95).value == "critical" else Severity.from_score(0.95).value,
        "medium",
        "low",
    ]
    assert "high" in body[0]
    assert "mid" in body[1]
    assert "low" in body[2]


def test_no_open_alerts_is_stated_plainly() -> None:
    text = format_status([], _plan(), [])
    assert text.startswith("Patrol status: no open alerts")
    assert "Last patrol run: never" in text


def test_only_below_threshold_states_say_so_explicitly() -> None:
    text = format_status([_state("gmv", 0.1)], _plan(), [])
    assert "no open alerts at or above the medium threshold" in text
    assert "1 lower-severity observation(s) hidden" in text


def test_limit_truncates_and_reports_how_many_were_dropped() -> None:
    states = [_state(f"gmv_by_category|category=c{i}", 0.5 + i / 100) for i in range(5)]

    text = format_status(states, _plan(), [], limit=2)

    assert "5 open alert(s)" in text
    assert "3 more alert(s) not shown (limit=2)" in text


def test_last_run_uses_the_newest_log_entry() -> None:
    recent = [
        {"created_at": "2026-05-31T21:00:00", "decision": "push", "severity": "medium", "fingerprint_key": "gmv"},
        {"created_at": "2026-05-30T09:00:00", "decision": "suppressed", "severity": "low", "fingerprint_key": "gmv"},
    ]

    text = format_status([], _plan(), recent)

    assert "Last patrol run: 2026-05-31T21:00:00" in text


def test_include_recent_lists_the_audit_trail() -> None:
    recent = [
        {
            "created_at": "2026-05-31T21:00:00",
            "decision": "suppressed",
            "severity": "low",
            "reason": "below_min_severity:low<medium",
            "fingerprint_key": "gmv_by_category|category=phone",
        }
    ]

    text = format_status([], _plan(), recent, include_recent=True)

    assert "Recent decisions (newest first):" in text
    assert "分品类 GMV (phone) -> suppressed (below_min_severity:low<medium)" in text


def test_plan_path_is_rendered_with_forward_slashes() -> None:
    text = format_status([], _plan(), [], plan_path=Path("example_ecommerce/patrol_plan.yaml"))
    assert "Plan: example_ecommerce/patrol_plan.yaml, 2 metrics" in text


# ---------------------------------------------------------------------------
# failure paths: the tool must never raise
# ---------------------------------------------------------------------------


def test_unconfigured_patrol_explains_how_to_enable_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(patrol_tool, "_settings", lambda: None)

    text = patrol_status.invoke({"reasoning": "test"})

    assert "Patrol is not configured" in text
    assert "patrol.enabled" in text


def test_missing_state_db_suggests_the_cli_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text("metrics:\n  - name: gmv\n    sql: SELECT 1\n", encoding="utf-8")
    monkeypatch.setattr(patrol_tool, "_settings", lambda: (plan_file, tmp_path / "missing.sqlite"))

    text = patrol_status.invoke({"reasoning": "test"})

    assert "does not exist yet" in text
    assert "python -m openchatbi.patrol" in text


def test_broken_plan_does_not_abort_the_turn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text("metrics: []\n", encoding="utf-8")
    state_db = tmp_path / "state.sqlite"
    with AlertStateStore(state_db):
        pass
    monkeypatch.setattr(patrol_tool, "_settings", lambda: (plan_file, state_db))

    text = patrol_status.invoke({"reasoning": "test"})

    assert "no open alerts" in text


def test_corrupt_state_db_returns_a_message_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text("metrics:\n  - name: gmv\n    sql: SELECT 1\n", encoding="utf-8")
    state_db = tmp_path / "state.sqlite"
    state_db.write_bytes(b"this is not a sqlite database at all, not even close" * 40)
    monkeypatch.setattr(patrol_tool, "_settings", lambda: (plan_file, state_db))

    text = patrol_status.invoke({"reasoning": "test"})

    assert text.startswith("Patrol status unavailable")


# ---------------------------------------------------------------------------
# end to end against a real state db
# ---------------------------------------------------------------------------


def test_reads_a_real_state_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(
        "defaults:\n  min_severity: medium\n"
        "metrics:\n"
        "  - name: gmv\n    label: 成交总额 GMV\n    sql: SELECT 1\n"
        "  - name: gmv_by_category\n    label: 分品类 GMV\n    dimension: category\n    sql: SELECT 1\n",
        encoding="utf-8",
    )
    state_db = tmp_path / "state.sqlite"
    with AlertStateStore(state_db) as store:
        store.put(_state("gmv_by_category|category=office", 0.42, pushed=1))
        store.put(_state("gmv_by_category|category=beauty", 0.11))
        store.log(
            "gmv_by_category|category=office",
            "2026-05-30",
            0.42,
            "medium",
            "push",
            "new",
            TS,
        )
    monkeypatch.setattr(patrol_tool, "_settings", lambda: (plan_file, state_db))

    text = patrol_status.invoke({"reasoning": "test"})

    assert "1 open alert(s) at or above the medium threshold" in text
    assert "分品类 GMV (office)" in text
    assert "beauty" not in text
    assert "Last patrol run: 2026-05-31T21:00:00" in text
