"""
loom.events.store
-------------------
Append-only, crash-safe event log backed by SQLite. Every model call,
tool invocation, approval decision, and file edit is recorded as an
event tied to a session_id. A killed process can be resumed by replaying
events for the last open session.

This is intentionally boring: SQLite gives us durability and zero-config
portability without running a separate service.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',   -- running | completed | failed | aborted
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,        -- model_call | tool_call | tool_result | approval | edit | stage | error
    agent TEXT,                -- planner | coder | tester | reviewer | orchestrator
    payload TEXT NOT NULL,     -- JSON blob
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, seq);
"""


@dataclass
class Event:
    seq: int
    session_id: str
    ts: float
    kind: str
    agent: Optional[str]
    payload: dict[str, Any]


class EventStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # -- sessions ---------------------------------------------------
    def new_session(self, task: str) -> str:
        sid = uuid.uuid4().hex[:12]
        now = time.time()
        self._conn.execute(
            "INSERT INTO sessions (id, task, status, created_at, updated_at) VALUES (?,?,?,?,?)",
            (sid, task, "running", now, now),
        )
        self._conn.commit()
        return sid

    def set_session_status(self, session_id: str, status: str) -> None:
        self._conn.execute(
            "UPDATE sessions SET status=?, updated_at=? WHERE id=?",
            (status, time.time(), session_id),
        )
        self._conn.commit()

    def last_open_session(self) -> Optional[str]:
        row = self._conn.execute(
            "SELECT id FROM sessions WHERE status='running' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None

    def session_task(self, session_id: str) -> Optional[str]:
        row = self._conn.execute("SELECT task FROM sessions WHERE id=?", (session_id,)).fetchone()
        return row[0] if row else None

    # -- events -------------------------------------------------------
    def append(self, session_id: str, kind: str, payload: dict[str, Any], agent: Optional[str] = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO events (session_id, ts, kind, agent, payload) VALUES (?,?,?,?,?)",
            (session_id, time.time(), kind, agent, json.dumps(payload, default=str)),
        )
        self._conn.commit()
        return cur.lastrowid

    def replay(self, session_id: str) -> Iterator[Event]:
        rows = self._conn.execute(
            "SELECT seq, session_id, ts, kind, agent, payload FROM events WHERE session_id=? ORDER BY seq",
            (session_id,),
        ).fetchall()
        for r in rows:
            yield Event(seq=r[0], session_id=r[1], ts=r[2], kind=r[3], agent=r[4], payload=json.loads(r[5]))

    def tail(self, session_id: str, n: int = 20) -> list[Event]:
        rows = self._conn.execute(
            "SELECT seq, session_id, ts, kind, agent, payload FROM events WHERE session_id=? ORDER BY seq DESC LIMIT ?",
            (session_id, n),
        ).fetchall()
        events = [Event(seq=r[0], session_id=r[1], ts=r[2], kind=r[3], agent=r[4], payload=json.loads(r[5])) for r in rows]
        return list(reversed(events))

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def stage(self, session_id: str, name: str, agent: Optional[str] = None):
        """Context manager that logs stage start/end (and errors) around a block of work."""
        self.append(session_id, "stage", {"name": name, "phase": "start"}, agent=agent)
        try:
            yield
        except Exception as e:
            self.append(session_id, "error", {"name": name, "error": str(e)}, agent=agent)
            raise
        else:
            self.append(session_id, "stage", {"name": name, "phase": "end"}, agent=agent)
