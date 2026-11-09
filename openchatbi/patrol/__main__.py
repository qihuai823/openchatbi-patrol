"""Command line entry point for the patrol subsystem.

Examples
--------
Preview what a run would do, without touching state or sending anything::

    python -m openchatbi.patrol --plan example_ecommerce/patrol_plan.yaml --dry-run

Replay a run as if it happened at a given moment (useful for backtesting the
thresholds against history)::

    python -m openchatbi.patrol --plan example_ecommerce/patrol_plan.yaml --now 2026-06-01T09:00 --dry-run

Real scheduled run, alerts appended to a JSONL file::

    python -m openchatbi.patrol --plan patrol_plan.yaml --state /var/lib/patrol/state.sqlite \\
        --notify jsonl --jsonl /var/log/patrol/alerts.jsonl

Show the recent decision audit trail::

    python -m openchatbi.patrol --plan patrol_plan.yaml --history 20

Exit codes: 0 on a clean run, 1 when the plan or the warehouse is unusable,
2 when the run completed but reported errors for individual metrics. The distinct
codes let a scheduler tell "broken" apart from "degraded".
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from openchatbi.patrol.notify import ConsoleNotifier, JsonlNotifier, MultiNotifier, NullNotifier, WebhookNotifier
from openchatbi.patrol.plan import PatrolPlan, PlanError
from openchatbi.patrol.runner import PatrolError, PatrolRunner
from openchatbi.patrol.store import AlertStateStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m openchatbi.patrol",
        description="Evaluate a patrol plan and push new alerts.",
    )
    parser.add_argument("--plan", required=True, help="path to the patrol plan YAML")
    parser.add_argument("--db", default=None, help="override the warehouse path from the plan")
    parser.add_argument("--state", default="patrol_state.sqlite", help="alert state / audit database")
    parser.add_argument("--dry-run", action="store_true", help="evaluate only: no push, no state write")
    parser.add_argument("--now", default=None, help="ISO timestamp to run as (for backtesting)")
    parser.add_argument(
        "--notify",
        choices=("console", "jsonl", "webhook", "none"),
        default="console",
        help="delivery channel (default: console)",
    )
    parser.add_argument("--jsonl", default="patrol_alerts.jsonl", help="path for --notify jsonl")
    parser.add_argument("--webhook", default=None, help="URL for --notify webhook")
    parser.add_argument("--history", type=int, default=0, help="print the N most recent decisions and exit")
    return parser


def make_notifier(args: argparse.Namespace):
    if args.notify == "none":
        return NullNotifier()
    if args.notify == "jsonl":
        return JsonlNotifier(args.jsonl)
    if args.notify == "webhook":
        if not args.webhook:
            raise SystemExit("--notify webhook requires --webhook URL")
        return MultiNotifier([ConsoleNotifier(), WebhookNotifier(args.webhook)])
    return ConsoleNotifier()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        plan = PatrolPlan.load(args.plan)
    except PlanError as exc:
        print(f"plan error: {exc}", file=sys.stderr)
        return 1

    now: datetime | None = None
    if args.now:
        try:
            now = datetime.fromisoformat(args.now)
        except ValueError:
            print(f"--now is not a valid ISO timestamp: {args.now!r}", file=sys.stderr)
            return 1

    with AlertStateStore(args.state) as store:
        if args.history > 0:
            rows = store.recent_log(limit=args.history)
            if not rows:
                print("no decisions recorded yet")
                return 0
            print(f"{'when':<22}{'decision':<12}{'severity':<10}{'bucket':<12}reason")
            for row in rows:
                print(
                    f"{row['created_at'][:19]:<22}{row['decision']:<12}{row['severity']:<10}"
                    f"{row['bucket']:<12}{row['reason']}"
                )
            return 0

        try:
            runner = PatrolRunner(plan, store, make_notifier(args), database=args.db)
        except PatrolError as exc:
            print(f"runner error: {exc}", file=sys.stderr)
            return 1

        try:
            result = runner.run(now=now, dry_run=args.dry_run)
        except PatrolError as exc:
            print(f"run failed: {exc}", file=sys.stderr)
            return 1

    prefix = "[dry-run] " if args.dry_run else ""
    print(f"{prefix}{result.summary_line()}")
    for alert, reason in result.suppressed:
        print(f"{prefix}  suppressed {alert.fingerprint.key} ({reason})")
    for key in result.resolved:
        print(f"{prefix}  resolved {key}")
    for error in result.errors:
        print(f"{prefix}  ERROR {error}", file=sys.stderr)

    return 2 if result.errors else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
