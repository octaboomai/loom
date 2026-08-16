"""
loom.report
--------------
Cross-session cost & audit report, built entirely from the local event
store — no new tracking needed, every number here already exists in
`.loom/events.sqlite3`, this just aggregates it into something a human
(or a manager who doesn't want to read raw JSON) can actually use.

Free tier: console summary table.
Team tier: CSV/JSON export with full audit detail — every approval and
denial, every credential swap, per session. This is the feature companies
actually pay for: "show me what the AI did across the team, and prove
nothing happened without a human approving it."
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from typing import Optional

from loom.events.store import EventStore


@dataclass
class SessionSummary:
    session_id: str
    task: str
    status: str
    created_at: float
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0
    approvals_granted: int = 0
    approvals_denied: int = 0
    key_swaps: list[str] = field(default_factory=list)
    committed: bool = False
    commit_sha: Optional[str] = None


def build_summaries(store: EventStore, limit: int = 200) -> list[SessionSummary]:
    rows = store._conn.execute(
        "SELECT id, task, status, created_at FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    summaries = []
    for sid, task, status, created_at in rows:
        s = SessionSummary(session_id=sid, task=task, status=status, created_at=created_at)
        for ev in store.replay(sid):
            if ev.kind == "model_response":
                s.model_calls += 1
                s.input_tokens += ev.payload.get("input_tokens", 0)
                s.output_tokens += ev.payload.get("output_tokens", 0)
            elif ev.kind == "approval":
                if ev.payload.get("approved"):
                    s.approvals_granted += 1
                else:
                    s.approvals_denied += 1
            elif ev.kind == "key_swap":
                s.key_swaps.append(ev.payload.get("failed_credential", "?"))
            elif ev.kind == "commit":
                s.committed = True
                s.commit_sha = ev.payload.get("sha")
        summaries.append(s)
    return summaries


def to_csv(summaries: list[SessionSummary]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["session_id", "task", "status", "created_at", "input_tokens", "output_tokens",
                      "model_calls", "approvals_granted", "approvals_denied", "key_swaps",
                      "committed", "commit_sha"])
    for s in summaries:
        writer.writerow([s.session_id, s.task, s.status, s.created_at, s.input_tokens, s.output_tokens,
                          s.model_calls, s.approvals_granted, s.approvals_denied, "; ".join(s.key_swaps),
                          s.committed, s.commit_sha or ""])
    return buf.getvalue()


def to_json(summaries: list[SessionSummary]) -> str:
    return json.dumps([s.__dict__ for s in summaries], indent=2, default=str)
