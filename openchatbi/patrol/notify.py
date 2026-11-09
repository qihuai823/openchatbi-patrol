"""Notifiers for patrol alerts.

Three implementations cover the realistic cases:

* :class:`ConsoleNotifier` - for cron jobs that pipe stdout into a log collector.
* :class:`JsonlNotifier` - append-only JSON Lines, the easiest thing to ingest
  into anything later without writing a parser.
* :class:`WebhookNotifier` - a plain JSON POST. Works with Feishu/Lark, Slack,
  DingTalk or an internal gateway; the payload is intentionally generic and the
  text field carries the human-readable summary.

Uses ``urllib`` rather than ``requests`` so the patrol path has zero third-party
dependencies beyond PyYAML for the plan file.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from openchatbi.patrol.models import Alert, AlertState


class Notifier(Protocol):
    """Anything that can deliver an alert."""

    def send(self, alert: Alert, state: AlertState) -> None:  # pragma: no cover - protocol
        ...


def alert_payload(alert: Alert, state: AlertState) -> dict[str, Any]:
    """Build the transport-neutral representation of an alert.

    Kept as a plain dict so every notifier shares one wire format and tests can
    assert on it without touching a network.
    """
    deviation = alert.deviation
    return {
        "text": alert.summary(),
        "fingerprint": alert.fingerprint.key,
        "metric": alert.fingerprint.metric,
        "dimension": alert.fingerprint.dimension,
        "dimension_value": alert.fingerprint.dimension_value,
        "severity": alert.severity.value,
        "score": round(alert.score, 4),
        "direction": deviation.direction.value,
        "bucket": deviation.bucket,
        "actual": deviation.actual,
        "baseline": deviation.baseline,
        "scale": deviation.scale,
        "z_score": round(deviation.z, 3),
        "delta_pct": None if deviation.delta_pct is None else round(deviation.delta_pct, 2),
        "first_seen": state.first_seen.isoformat(),
        "push_count": state.push_count,
    }


class ConsoleNotifier:
    """Print alerts to stdout. The default for cron + log collector setups."""

    def __init__(self, prefix: str = "[patrol]") -> None:
        self.prefix = prefix

    def send(self, alert: Alert, state: AlertState) -> None:
        print(f"{self.prefix} {alert.summary()}")
        if state.push_count > 1:
            print(f"{self.prefix}   (recurring, pushed {state.push_count}x since {state.first_seen.isoformat()})")


class JsonlNotifier:
    """Append alerts as one JSON object per line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, alert: Alert, state: AlertState) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(alert_payload(alert, state), ensure_ascii=False) + "\n")


class WebhookNotifier:
    """POST the alert payload as JSON to a webhook URL."""

    def __init__(self, url: str, timeout: float = 10.0) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"webhook url must be http(s), got {url!r}")
        self.url = url
        self.timeout = timeout

    def send(self, alert: Alert, state: AlertState) -> None:
        body = json.dumps(alert_payload(alert, state), ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - scheme validated
                response.read()
        except urllib.error.URLError as exc:
            # A failing webhook must not abort the whole patrol run: the state has
            # already been persisted by the caller, so the next run still knows.
            print(f"[patrol] webhook delivery failed: {exc}")


class MultiNotifier:
    """Fan an alert out to several notifiers."""

    def __init__(self, notifiers: list[Notifier]) -> None:
        self.notifiers = notifiers

    def send(self, alert: Alert, state: AlertState) -> None:
        for notifier in self.notifiers:
            notifier.send(alert, state)


class NullNotifier:
    """Swallow alerts. Used by dry runs and tests."""

    def __init__(self) -> None:
        self.sent: list[Alert] = []

    def send(self, alert: Alert, state: AlertState) -> None:
        self.sent.append(alert)


def state_to_dict(state: AlertState) -> dict[str, Any]:
    """Serialize state for logging. Datetimes are converted by ``asdict`` + ``default=str``."""
    return asdict(state)
