"""Proactive patrol: turn the passive Q&A agent into a scheduled watchdog.

The interactive agent waits for a question. A patrol run does not: it evaluates a
declarative plan of business metrics on a schedule, flags deviations from each
metric's own history, and pushes only the alerts that are new, worsening, or due
for a reminder.

Why this lives beside the agent rather than inside the graph
------------------------------------------------------------
The agent graph is built around ``MessagesState``: every node consumes and
produces conversation messages, and the loop only advances when the LLM decides to
call a tool. A patrol run has no conversation, no LLM and no user - forcing it
through the graph would mean inventing a fake HumanMessage and parsing the answer
back out of prose. Keeping it as a separate, synchronous pipeline means it runs
unattended, costs nothing per run, and is testable without a model.

The two meet when a human asks about an alert: the patrol output (metric, bucket,
score, direction) is exactly the context the agent's ``text2sql`` tool needs to
drill down.

Public API
----------
``PatrolPlan``       declarative definition of what to watch
``PatrolRunner``     executes a plan once
``AlertStateStore``  SQLite-backed cooldown memory + audit trail
``ConsoleNotifier`` / ``JsonlNotifier`` / ``WebhookNotifier``  delivery
"""

from openchatbi.patrol.baseline import compute_baseline, detect, detect_series, score_from_z
from openchatbi.patrol.dedup import Decision, SuppressionRule, evaluate, find_suppression
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
from openchatbi.patrol.notify import ConsoleNotifier, JsonlNotifier, MultiNotifier, NullNotifier, WebhookNotifier
from openchatbi.patrol.plan import MetricSpec, PatrolPlan, PlanError
from openchatbi.patrol.runner import PatrolError, PatrolRunner
from openchatbi.patrol.store import AlertStateStore

__all__ = [
    "Alert",
    "AlertFingerprint",
    "AlertState",
    "AlertStateStore",
    "ConsoleNotifier",
    "Decision",
    "Deviation",
    "Direction",
    "JsonlNotifier",
    "MetricSpec",
    "MultiNotifier",
    "NullNotifier",
    "Observation",
    "PatrolError",
    "PatrolPlan",
    "PatrolResult",
    "PatrolRunner",
    "PlanError",
    "Severity",
    "SuppressionRule",
    "WebhookNotifier",
    "compute_baseline",
    "detect",
    "detect_series",
    "evaluate",
    "find_suppression",
    "score_from_z",
]
