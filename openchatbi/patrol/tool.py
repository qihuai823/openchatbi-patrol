"""Read-only ``patrol_status`` tool: "is anything wrong right now?"

The patrol pipeline itself stays a scheduled job (``python -m openchatbi.patrol``)
because it *mutates* alert state and pushes notifications - neither belongs
inside a conversational turn. This module only exposes the result of past runs
so the agent can answer questions like "any anomalies today?" or "what is still
unresolved?".

Design constraints, all deliberate:

* **Never raises.** A tool that throws aborts the agent turn. Every failure path
  (unconfigured, missing plan, missing state db, corrupt db) returns an
  explanatory string instead.
* **Never writes.** Only ``list_open`` / ``recent_log`` are called. Running a
  patrol from inside a chat would let a user reset the cooldown window and
  re-trigger notifications.
* **Absent by default.** The tool is only registered on the main graph when
  ``patrol.enabled`` is true, so deployments that do not use patrols do not pay
  for an extra tool in the LLM's tool list.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from openchatbi import config
from openchatbi.patrol.models import AlertState, Severity
from openchatbi.patrol.plan import PatrolPlan
from openchatbi.patrol.store import AlertStateStore
from openchatbi.utils import log


class PatrolStatusInput(BaseModel):
    reasoning: str = Field(description="Reason for checking the patrol status")
    include_recent: bool = Field(
        default=False,
        description=(
            "Also list the most recent patrol decisions, including alerts that were "
            "suppressed by cooldown or severity rules. Useful when the user asks why "
            "something was or was not reported."
        ),
    )
    include_below_threshold: bool = Field(
        default=False,
        description=(
            "Also list deviations below the plan's minimum severity. These are stored as "
            "open but were never pushed, so they are observations rather than alerts."
        ),
    )
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of entries to return")


def _settings() -> tuple[Path, Path] | None:
    """Return ``(plan_file, state_db)`` when patrol is enabled and configured."""
    try:
        cfg = config.get()
    except ValueError:
        return None
    patrol = getattr(cfg, "patrol", None)
    if patrol is None or not getattr(patrol, "enabled", False):
        return None
    plan_file = getattr(patrol, "plan_file", None)
    state_db = getattr(patrol, "state_db", None)
    if not plan_file or not state_db:
        return None
    return Path(plan_file), Path(state_db)


def _split_fingerprint(key: str) -> tuple[str, str | None, str | None]:
    """Reverse ``AlertFingerprint.key``: ``gmv|category=office`` -> (gmv, category, office)."""
    metric, _, rest = key.partition("|")
    if not rest:
        return metric, None, None
    dimension, _, value = rest.partition("=")
    return metric, dimension or None, value or None


def _display_name(key: str, plan: PatrolPlan | None) -> str:
    metric, _, value = _split_fingerprint(key)
    if plan is not None:
        spec = next((m for m in plan.metrics if m.name == metric), None)
        if spec is not None:
            return spec.display(value)
    return f"{metric} ({value})" if value else metric


def _format_state(state: AlertState, plan: PatrolPlan | None) -> str:
    severity = Severity.from_score(state.last_score)
    pushed = f"pushed {state.push_count}x" if state.push_count else "never pushed"
    return (
        f"[{severity.value.upper()}] {_display_name(state.fingerprint_key, plan)}\n"
        f"  bucket {state.last_bucket} | score {state.last_score:.2f} | {pushed} | "
        f"first seen {state.first_seen:%Y-%m-%d %H:%M} | last seen {state.last_seen:%Y-%m-%d %H:%M}"
    )


def format_status(
    open_states: list[AlertState],
    plan: PatrolPlan | None,
    recent: list[dict[str, object]],
    *,
    plan_path: Path | None = None,
    include_recent: bool = False,
    include_below_threshold: bool = False,
    limit: int = 20,
) -> str:
    """Render a patrol status report. Pure function so it can be tested directly.

    ``AlertStateStore`` keeps *every* detected deviation as ``open``, including the
    ones the severity floor rejected before they could ever be pushed. Listing all
    of them would bury the one alert that matters under a dozen observations, so
    the report is filtered to the plan's ``min_severity`` by default.
    """
    threshold = plan.min_severity if plan is not None else Severity.LOW

    def severity_of(state: AlertState) -> Severity:
        return Severity.from_score(state.last_score)

    ranked = sorted(open_states, key=lambda s: (severity_of(s).rank, s.last_score), reverse=True)
    visible = ranked if include_below_threshold else [s for s in ranked if severity_of(s).rank >= threshold.rank]
    hidden = len(ranked) - len(visible)
    total_visible = len(visible)
    truncated = max(0, total_visible - limit)
    visible = visible[:limit]

    qualifier = "" if include_below_threshold else f" at or above the {threshold.value} threshold"
    if visible:
        lines = [f"Patrol status: {total_visible} open alert(s){qualifier}"]
        lines.extend(_format_state(state, plan) for state in visible)
    elif hidden:
        lines = [f"Patrol status: no open alerts at or above the {threshold.value} threshold"]
    else:
        lines = ["Patrol status: no open alerts"]

    if hidden and not include_below_threshold:
        lines.append(
            f"{hidden} lower-severity observation(s) hidden (below the {threshold.value} threshold); "
            "pass include_below_threshold=true to list them"
        )
    if truncated:
        lines.append(f"{truncated} more alert(s) not shown (limit={limit})")

    if recent:
        last_run = max(str(entry.get("created_at", "")) for entry in recent)
        lines.append(f"Last patrol run: {last_run}")
    else:
        lines.append("Last patrol run: never - the patrol job has not been executed against this state db")

    if plan_path is not None:
        metrics = f", {len(plan.metrics)} metrics" if plan is not None else ""
        lines.append(f"Plan: {Path(plan_path).as_posix()}{metrics}")

    if include_recent and recent:
        lines.append("")
        lines.append("Recent decisions (newest first):")
        for entry in recent:
            lines.append(
                f"  {entry.get('created_at')} [{str(entry.get('severity', '')).upper()}] "
                f"{_display_name(str(entry.get('fingerprint_key', '')), plan)} -> "
                f"{entry.get('decision')} ({entry.get('reason')})"
            )
    return "\n".join(lines)


def _unavailable(message: str) -> str:
    log(f"patrol_status unavailable: {message}")
    return f"Patrol status unavailable: {message}"


@tool("patrol_status", args_schema=PatrolStatusInput, return_direct=False, infer_schema=True)
def patrol_status(
    reasoning: str,
    include_recent: bool = False,
    include_below_threshold: bool = False,
    limit: int = 20,
) -> str:
    """Report the anomalies detected by the proactive patrol job.

    Use this when the user asks whether anything looks wrong, what is currently
    alerting, or why an anomaly was or was not reported. This reads the state
    written by the scheduled patrol run; it does not run a new patrol and does
    not modify any alert state.

    Returns:
        str: Open alerts at or above the plan's severity floor, with their
        severity, last bucket and score, plus the timestamp of the most recent
        patrol run.
    """
    log(f"Check patrol status, Reasoning: {reasoning}")

    settings = _settings()
    if settings is None:
        return (
            "Patrol is not configured. Set `patrol.enabled: true`, `patrol.plan_file` and "
            "`patrol.state_db` in config.yaml to enable proactive anomaly reporting."
        )
    plan_path, state_db = settings

    plan: PatrolPlan | None = None
    try:
        plan = PatrolPlan.load(plan_path)
    except Exception as exc:  # noqa: BLE001 - a bad plan must not abort the turn
        log(f"Failed to load patrol plan {plan_path}: {exc}")

    if not state_db.exists():
        return _unavailable(
            f"state database {state_db} does not exist yet - run `python -m openchatbi.patrol "
            f"--plan {plan_path} --state {state_db}` first"
        )

    try:
        with AlertStateStore(state_db) as store:
            open_states = store.list_open()
            recent = store.recent_log(limit=limit)
    except Exception as exc:  # noqa: BLE001 - corrupt db must not abort the turn
        return _unavailable(f"cannot read {state_db}: {type(exc).__name__}: {exc}")

    return format_status(
        open_states,
        plan,
        recent,
        plan_path=plan_path,
        include_recent=include_recent,
        include_below_threshold=include_below_threshold,
        limit=limit,
    )


def patrol_status_tool() -> object:
    """Return the tool when patrol is enabled, else ``None``.

    Kept separate from :data:`patrol_status` so the graph builder can decide
    whether to register it without duplicating the config lookup.
    """
    return patrol_status if _settings() is not None else None
