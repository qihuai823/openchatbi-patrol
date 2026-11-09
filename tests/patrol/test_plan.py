"""Tests for patrol plan loading and validation."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
import yaml

from openchatbi.patrol.models import Severity
from openchatbi.patrol.plan import PatrolPlan, PlanError

MINIMAL = {
    "version": 1,
    "metrics": [
        {
            "name": "gmv",
            "sql": "SELECT bucket, value FROM m ORDER BY bucket",
        }
    ],
}


def test_minimal_plan_uses_documented_defaults() -> None:
    plan = PatrolPlan.from_dict(MINIMAL)

    assert len(plan.metrics) == 1
    metric = plan.metrics[0]
    assert metric.history_points == 35
    assert metric.min_samples == 14
    assert metric.direction == "both"
    assert metric.weekday_aware is True
    assert plan.cooldown == timedelta(hours=24)
    assert plan.min_severity is Severity.MEDIUM
    assert plan.escalate is True
    assert plan.suppress == []


def test_metric_overrides_defaults() -> None:
    plan = PatrolPlan.from_dict(
        {
            "defaults": {"cooldown_hours": 6, "min_severity": "high"},
            "metrics": [
                {
                    "name": "return_rate",
                    "sql": "SELECT bucket, value FROM m",
                    "direction": "rise",
                    "min_relative_change": 0.3,
                    "history_points": 20,
                    "min_samples": 10,
                }
            ],
        }
    )

    assert plan.cooldown == timedelta(hours=6)
    assert plan.min_severity is Severity.HIGH
    metric = plan.metrics[0]
    assert metric.direction == "rise"
    assert metric.min_relative_change == 0.3
    assert metric.history_points == 20


def test_dimension_and_direction_scope() -> None:
    plan = PatrolPlan.from_dict(
        {
            "metrics": [
                {
                    "name": "gmv_by_category",
                    "sql": "SELECT bucket, category, value FROM m",
                    "dimension": "category",
                    "direction": "drop",
                }
            ]
        }
    )
    metric = plan.metrics[0]

    assert metric.dimension == "category"
    assert metric.allows("drop") is True
    assert metric.allows("rise") is False
    assert metric.display("electronics") == "gmv_by_category (electronics)"


def test_suppression_rules_are_parsed() -> None:
    plan = PatrolPlan.from_dict(
        {
            **MINIMAL,
            "suppress": [{"reason": "partial month", "bucket_glob": "2025-11*"}],
        }
    )

    assert len(plan.suppress) == 1
    assert plan.suppress[0].reason == "partial month"
    assert plan.suppress[0].bucket_glob == "2025-11*"


def test_missing_metrics_is_rejected() -> None:
    with pytest.raises(PlanError, match="non-empty list"):
        PatrolPlan.from_dict({"version": 1})


def test_missing_sql_is_rejected() -> None:
    with pytest.raises(PlanError, match="missing a string `sql`"):
        PatrolPlan.from_dict({"metrics": [{"name": "gmv"}]})


def test_missing_name_is_rejected() -> None:
    with pytest.raises(PlanError, match="missing a string `name`"):
        PatrolPlan.from_dict({"metrics": [{"sql": "SELECT 1"}]})


def test_duplicate_metric_names_are_rejected() -> None:
    entry = {"name": "gmv", "sql": "SELECT bucket, value FROM m"}
    with pytest.raises(PlanError, match="duplicate metric name"):
        PatrolPlan.from_dict({"metrics": [entry, dict(entry)]})


def test_invalid_direction_is_rejected() -> None:
    with pytest.raises(PlanError, match="direction must be one of"):
        PatrolPlan.from_dict({"metrics": [{"name": "gmv", "sql": "SELECT 1", "direction": "sideways"}]})


def test_history_points_must_exceed_min_samples() -> None:
    with pytest.raises(PlanError, match="must be greater than min_samples"):
        PatrolPlan.from_dict(
            {"metrics": [{"name": "gmv", "sql": "SELECT 1", "history_points": 10, "min_samples": 10}]}
        )


def test_min_samples_floor_is_enforced() -> None:
    with pytest.raises(PlanError, match="min_samples must be >= 3"):
        PatrolPlan.from_dict({"metrics": [{"name": "gmv", "sql": "SELECT 1", "min_samples": 2}]})


def test_invalid_min_severity_is_rejected() -> None:
    with pytest.raises(PlanError, match="min_severity must be one of"):
        PatrolPlan.from_dict({"defaults": {"min_severity": "apocalyptic"}, "metrics": MINIMAL["metrics"]})


def test_suppression_without_reason_is_rejected() -> None:
    # An unexplained suppression is indistinguishable from a bug.
    with pytest.raises(PlanError, match="missing a `reason`"):
        PatrolPlan.from_dict({**MINIMAL, "suppress": [{"bucket_glob": "2025-11*"}]})


def test_load_from_file(tmp_path: Path) -> None:
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(yaml.safe_dump(MINIMAL), encoding="utf-8")

    plan = PatrolPlan.load(plan_file)

    assert plan.metrics[0].name == "gmv"


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(PlanError, match="not found"):
        PatrolPlan.load(tmp_path / "nope.yaml")


def test_shipped_ecommerce_plan_is_valid() -> None:
    """The plan we ship must actually load - guards against doc drift."""
    plan = PatrolPlan.load("example_ecommerce/patrol_plan.yaml")

    names = [metric.name for metric in plan.metrics]
    assert "gmv" in names
    assert "gmv_by_category" in names
    assert plan.database == "example_ecommerce/ecommerce.sqlite"
    assert any("2025-11" in rule.reason for rule in plan.suppress)
