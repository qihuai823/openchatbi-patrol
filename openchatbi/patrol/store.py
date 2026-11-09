"""SQLite persistence for patrol alert state and decision audit trail.

Two tables:

* ``alert_state`` - one row per fingerprint. This is the memory that makes the
  cooldown work across process restarts.
* ``alert_log`` - append-only record of every decision (push or suppress) with its
  reason. Without it, "why did nobody get paged for this?" is unanswerable, which
  is the single most common complaint about alerting systems.

Deliberately uses the stdlib ``sqlite3`` driver rather than SQLAlchemy: the patrol
subsystem must be importable and runnable without the warehouse ORM stack.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from openchatbi.patrol.models import AlertState

SCHEMA = """
CREATE TABLE IF NOT EXISTS alert_state (
    fingerprint_key TEXT PRIMARY KEY,
    first_seen      TEXT NOT NULL,
    last_seen       TEXT NOT NULL,
    last_pushed     TEXT,
    push_count      INTEGER NOT NULL DEFAULT 0,
    last_score      REAL NOT NULL DEFAULT 0.0,
    last_bucket     TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'open'
);

CREATE TABLE IF NOT EXISTS alert_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint_key TEXT NOT NULL,
    bucket          TEXT NOT NULL,
    score           REAL NOT NULL,
    severity        TEXT NOT NULL,
    decision        TEXT NOT NULL,
    reason          TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_alert_log_key ON alert_log(fingerprint_key);
CREATE INDEX IF NOT EXISTS idx_alert_log_created ON alert_log(created_at);
"""


class AlertStateStore:
    """Persistence for alert lifecycle state and the decision audit trail."""

    def __init__(self, path: str | Path = "patrol_state.sqlite") -> None:
        self.path = Path(path)
        if self.path.parent and str(self.path.parent) not in ("", "."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> AlertStateStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- reads -------------------------------------------------------------
    def get(self, fingerprint_key: str) -> AlertState | None:
        """Load the persisted state for one fingerprint."""
        row = self._conn.execute(
            "SELECT * FROM alert_state WHERE fingerprint_key = ?", (fingerprint_key,)
        ).fetchone()
        return self._row_to_state(row) if row else None

    def get_open(self, fingerprint_key: str) -> AlertState | None:
        """Load state only when the fingerprint is still unresolved.

        Returning ``None`` for resolved fingerprints makes the next occurrence look
        like a fresh problem, which is the intended semantics: a problem that went
        away and came back deserves a new page.
        """
        state = self.get(fingerprint_key)
        if state is None or not state.is_open():
            return None
        return state

    def list_open(self) -> list[AlertState]:
        """All unresolved fingerprints, most recently seen first."""
        rows = self._conn.execute(
            "SELECT * FROM alert_state WHERE status = 'open' ORDER BY last_seen DESC"
        ).fetchall()
        return [self._row_to_state(row) for row in rows]

    def recent_log(self, limit: int = 50) -> list[dict[str, object]]:
        """Most recent decisions, newest first - used by the CLI and for debugging."""
        rows = self._conn.execute("SELECT * FROM alert_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    # -- writes ------------------------------------------------------------
    def put(self, state: AlertState) -> None:
        """Insert or replace the state row for its fingerprint."""
        self._conn.execute(
            """
            INSERT INTO alert_state
                (fingerprint_key, first_seen, last_seen, last_pushed, push_count, last_score, last_bucket, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fingerprint_key) DO UPDATE SET
                first_seen  = excluded.first_seen,
                last_seen   = excluded.last_seen,
                last_pushed = excluded.last_pushed,
                push_count  = excluded.push_count,
                last_score  = excluded.last_score,
                last_bucket = excluded.last_bucket,
                status      = excluded.status
            """,
            (
                state.fingerprint_key,
                state.first_seen.isoformat(),
                state.last_seen.isoformat(),
                state.last_pushed.isoformat() if state.last_pushed else None,
                state.push_count,
                state.last_score,
                state.last_bucket,
                state.status,
            ),
        )
        self._conn.commit()

    def log(
        self,
        fingerprint_key: str,
        bucket: str,
        score: float,
        severity: str,
        decision: str,
        reason: str,
        created_at: datetime,
    ) -> None:
        """Append one decision to the audit trail."""
        self._conn.execute(
            "INSERT INTO alert_log (fingerprint_key, bucket, score, severity, decision, reason, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (fingerprint_key, bucket, score, severity, decision, reason, created_at.isoformat()),
        )
        self._conn.commit()

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _row_to_state(row: sqlite3.Row) -> AlertState:
        return AlertState(
            fingerprint_key=row["fingerprint_key"],
            first_seen=datetime.fromisoformat(row["first_seen"]),
            last_seen=datetime.fromisoformat(row["last_seen"]),
            last_pushed=datetime.fromisoformat(row["last_pushed"]) if row["last_pushed"] else None,
            push_count=int(row["push_count"]),
            last_score=float(row["last_score"]),
            last_bucket=row["last_bucket"],
            status=row["status"],
        )
