"""Patrol plan loading and validation.

A plan is a YAML file describing *what* to watch and *how* to behave when
something looks off. Keeping it declarative means adding a monitored metric is a
config change, not a code change - which is the whole point of a patrol system
someone else has to operate.

Schema (all keys under ``defaults`` are optional and can be overridden per metric)::

    version: 1
    database: example_ecommerce/ecommerce.sqlite   # optional; falls back to config
    defaults:
      history_points: 35        # how many points of history the SQL should return
      min_samples: 14           # minimum history before a point can be judged
      min_relative_change: 0.05 # ignore moves smaller than 5% of baseline
      cooldown_hours: 24
      min_severity: medium
      escalate: true
      weekday_aware: true
      drop_incomplete_tail: true # skip a trailing bucket that is still in progress
    metrics:
      - name: gmv
        label: 成交总额 GMV
        direction: both         # both | rise | drop
        sql: |
          SELECT date(o.date_order_placed) AS bucket, SUM(oi.line_amount) AS value
          FROM Orders o JOIN Order_Items oi ON oi.order_id = o.order_id
          WHERE o.order_status != 'Canceled'
          GROUP BY 1 ORDER BY 1
    suppress:
      - reason: "2025-11 only contains one day of data"
        bucket_glob: "2025-11*"

Every metric SQL must return a ``bucket`` column (a sortable date label) and a
``value`` column (numeric). When ``dimension`` is set it must also return a column
named after that dimension, and one series is built per distinct value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from openchatbi.patrol.dedup import SuppressionRule
from openchatbi.patrol.models import Severity

VALID_DIRECTIONS = ("both", "rise", "drop")

DEFAULTS: dict[str, Any] = {
    "history_points": 35,
    "min_samples": 14,
    "min_relative_change": 0.05,
    "cooldown_hours": 24,
    "min_severity": "medium",
    "escalate": True,
    "weekday_aware": True,
    "drop_incomplete_tail": True,
}


class PlanError(ValueError):
    """Raised when a patrol plan is structurally invalid."""


@dataclass
class MetricSpec:
    """One monitored metric."""

    name: str
    sql: str
    label: str = ""
    dimension: str | None = None
    direction: str = "both"
    history_points: int = 35
    min_samples: int = 14
    min_relative_change: float = 0.05
    min_absolute_change: float | None = None
    weekday_aware: bool = True
    drop_incomplete_tail: bool = True

    def allows(self, direction: str) -> bool:
        """Whether an alert in ``direction`` is in scope for this metric."""
        return self.direction == "both" or self.direction == direction

    def display(self, dimension_value: str | None = None) -> str:
        """Human label used in alert summaries."""
        base = self.label or self.name
        return f"{base} ({dimension_value})" if dimension_value else base


@dataclass
class PatrolPlan:
    """A complete, validated patrol plan."""

    metrics: list[MetricSpec]
    cooldown: timedelta = timedelta(hours=24)
    min_severity: Severity = Severity.MEDIUM
    escalate: bool = True
    suppress: list[SuppressionRule] = field(default_factory=list)
    database: str | None = None
    version: int = 1

    @classmethod
    def load(cls, path: str | Path) -> PatrolPlan:
        """Load and validate a plan from a YAML file."""
        import yaml

        plan_path = Path(path)
        if not plan_path.exists():
            raise PlanError(f"patrol plan not found: {plan_path}")
        with plan_path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise PlanError("patrol plan must be a YAML mapping at the top level")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PatrolPlan:
        """Build and validate a plan from an already-parsed mapping."""
        defaults = dict(DEFAULTS)
        user_defaults = raw.get("defaults") or {}
        if not isinstance(user_defaults, dict):
            raise PlanError("`defaults` must be a mapping")
        defaults.update(user_defaults)

        raw_metrics = raw.get("metrics")
        if not isinstance(raw_metrics, list) or not raw_metrics:
            raise PlanError("`metrics` must be a non-empty list")

        metrics: list[MetricSpec] = []
        seen: set[str] = set()
        for index, entry in enumerate(raw_metrics):
            if not isinstance(entry, dict):
                raise PlanError(f"metrics[{index}] must be a mapping")
            metric = _build_metric(entry, defaults, index)
            if metric.name in seen:
                raise PlanError(f"duplicate metric name: {metric.name}")
            seen.add(metric.name)
            metrics.append(metric)

        severity_name = str(defaults["min_severity"]).lower()
        try:
            min_severity = Severity(severity_name)
        except ValueError as exc:
            raise PlanError(
                f"min_severity must be one of {[s.value for s in Severity]}, got {severity_name!r}"
            ) from exc

        rules = _build_suppression(raw.get("suppress"))

        return cls(
            metrics=metrics,
            cooldown=timedelta(hours=float(defaults["cooldown_hours"])),
            min_severity=min_severity,
            escalate=bool(defaults["escalate"]),
            suppress=rules,
            database=raw.get("database"),
            version=int(raw.get("version", 1)),
        )


def _build_metric(entry: dict[str, Any], defaults: dict[str, Any], index: int) -> MetricSpec:
    name = entry.get("name")
    if not name or not isinstance(name, str):
        raise PlanError(f"metrics[{index}] is missing a string `name`")

    sql = entry.get("sql")
    if not sql or not isinstance(sql, str):
        raise PlanError(f"metric {name!r} is missing a string `sql`")

    direction = str(entry.get("direction", "both")).lower()
    if direction not in VALID_DIRECTIONS:
        raise PlanError(f"metric {name!r}: direction must be one of {list(VALID_DIRECTIONS)}, got {direction!r}")

    dimension = entry.get("dimension")
    if dimension is not None and not isinstance(dimension, str):
        raise PlanError(f"metric {name!r}: `dimension` must be a string")

    def pick(key: str) -> Any:
        return entry[key] if key in entry else defaults[key]

    min_samples = int(pick("min_samples"))
    if min_samples < 3:
        raise PlanError(f"metric {name!r}: min_samples must be >= 3, got {min_samples}")

    history_points = int(pick("history_points"))
    if history_points <= min_samples:
        raise PlanError(
            f"metric {name!r}: history_points ({history_points}) must be greater than min_samples ({min_samples})"
        )

    min_absolute = entry.get("min_absolute_change")
    return MetricSpec(
        name=name,
        sql=sql,
        label=str(entry.get("label", name)),
        dimension=dimension,
        direction=direction,
        history_points=history_points,
        min_samples=min_samples,
        min_relative_change=float(pick("min_relative_change")),
        min_absolute_change=float(min_absolute) if min_absolute is not None else None,
        weekday_aware=bool(pick("weekday_aware")),
        drop_incomplete_tail=bool(pick("drop_incomplete_tail")),
    )


def _build_suppression(raw: Any) -> list[SuppressionRule]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise PlanError("`suppress` must be a list")

    rules: list[SuppressionRule] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise PlanError(f"suppress[{index}] must be a mapping")
        reason = entry.get("reason")
        if not reason:
            raise PlanError(f"suppress[{index}] is missing a `reason` - an unexplained suppression is a bug")
        rules.append(
            SuppressionRule(
                reason=str(reason),
                metric=entry.get("metric"),
                dimension=entry.get("dimension"),
                dimension_value=entry.get("dimension_value"),
                bucket_glob=entry.get("bucket_glob"),
            )
        )
    return rules
