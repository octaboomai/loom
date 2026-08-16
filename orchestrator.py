"""
loom.orchestrator
--------------------
The hierarchical coordination engine: Planner -> Coder -> Validate ->
Tester -> Reviewer, with a bounded feedback loop back to the Coder if the
Tester fails or the Reviewer requests changes. Every stage is wrapped in
an EventStore.stage() context manager so a crash mid-run leaves a replayable
trail, and the whole run is resumable via `loom resume`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from loom.agents.base import ApprovalFn
from loom.agents.coder import build_coder
from loom.agents.planner import build_planner
from loom.agents.reviewer import build_reviewer
from loom.agents.tester import build_tester
from loom.config import LoomConfig
from loom.context.repo_map import build_repo_map, rank_relevant_files, render_map
from loom.events.store import EventStore
from loom.router import ModelRouter
from loom.tools import git_ops
from loom.validation.runner import run_validation

StatusFn = Callable[[str], None]  # simple progress callback for the CLI/TUI


@dataclass
class RunResult:
    session_id: str
    committed: bool
    commit_sha: Optional[str]
    verdict: str          # "approved" | "gave_up" | "no_changes"
    branch: Optional[str]


class Orchestrator:
    def __init__(self, config: LoomConfig, event_store: EventStore, router: Optional[ModelRouter] = None):
        self.config = config
        self.events = event_store
        self.router = router or ModelRouter(config)

    def _guardrail_preamble(self) -> str:
        if not self.config.team.guardrails and not self.config.team.forbidden_paths:
            return ""
        lines = ["TEAM GUARDRAILS (must be respected):"]
        lines += [f"- {g}" for g in self.config.team.guardrails]
        if self.config.team.forbidden_paths:
            lines.append(f"- Never touch these paths: {', '.join(self.config.team.forbidden_paths)}")
        return "\n".join(lines) + "\n\n"

    def run(
        self,
        task: str,
        approval_fn: Optional[ApprovalFn] = None,
        status: Optional[StatusFn] = None,
        max_repair_loops: int = 2,
        create_branch: bool = True,
        auto_commit: bool = False,
    ) -> RunResult:
        say = status or (lambda msg: None)
        session_id = self.events.new_session(task)
        preamble = self._guardrail_preamble()

        repo = git_ops.ensure_repo(self.config.repo_root)
        branch = None
        if create_branch:
            slug = "-".join(task.lower().split()[:6])
            slug = "".join(c for c in slug if c.isalnum() or c == "-")[:60] or session_id
            try:
                branch = git_ops.create_task_branch(repo, slug)
            except Exception:
                branch = git_ops.current_branch(repo)

        # ---- Context: repo map ----
        say("Mapping repository...")
        with self.events.stage(session_id, "repo_map"):
            maps = build_repo_map(self.config.repo_root)
            relevant = rank_relevant_files(maps, task)
            repo_map_text = render_map(relevant)
        self.events.append(session_id, "context", {"files_included": len(relevant), "files_total": len(maps)})

        # ---- Planner ----
        say("Planning...")
        planner = build_planner(self.router, self.config)
        with self.events.stage(session_id, "plan", agent="planner"):
            plan = planner.run(
                f"{preamble}TASK: {task}\n\nREPO MAP (most relevant files):\n{repo_map_text}\n\n"
                "Produce the implementation plan now.",
                self.events, session_id, approval_fn=approval_fn,
            )
        self.events.append(session_id, "artifact", {"name": "plan", "content": plan})

        coder_feedback = ""
        verdict = "gave_up"
        last_coder_summary = ""

        for attempt in range(max_repair_loops + 1):
            say(f"Coding (attempt {attempt + 1}/{max_repair_loops + 1})...")
            coder = build_coder(self.router, self.config)
            with self.events.stage(session_id, f"code_attempt_{attempt}", agent="coder"):
                coder_prompt = (
                    f"{preamble}TASK: {task}\n\nPLAN FROM PLANNER:\n{plan}\n"
                )
                if coder_feedback:
                    coder_prompt += f"\nFEEDBACK FROM PREVIOUS ATTEMPT (fix these before anything else):\n{coder_feedback}\n"
                coder_prompt += "\nImplement this now."
                last_coder_summary = coder.run(coder_prompt, self.events, session_id, approval_fn=approval_fn)
            self.events.append(session_id, "artifact", {"name": f"coder_summary_{attempt}", "content": last_coder_summary})

            # ---- Validate ----
            say("Running validation (lint/types/tests)...")
            with self.events.stage(session_id, f"validate_{attempt}"):
                report = run_validation(self.config.repo_root)
            report_text = report.render()
            self.events.append(session_id, "artifact", {"name": f"validation_{attempt}", "content": report_text})

            # ---- Tester ----
            say("Testing...")
            tester = build_tester(self.router, self.config)
            with self.events.stage(session_id, f"test_{attempt}", agent="tester"):
                tester_verdict = tester.run(
                    f"{preamble}TASK: {task}\n\nCODER SUMMARY:\n{last_coder_summary}\n\n"
                    f"AUTOMATED VALIDATION REPORT:\n{report_text}\n\nGive your verdict.",
                    self.events, session_id, approval_fn=approval_fn,
                )
            self.events.append(session_id, "artifact", {"name": f"tester_verdict_{attempt}", "content": tester_verdict})

            if "VERDICT: FAIL" in tester_verdict.upper().replace(" ", " "):
                coder_feedback = f"Tester reported FAIL:\n{tester_verdict}"
                continue  # skip straight to another coder attempt

            # ---- Reviewer ----
            say("Reviewing...")
            reviewer = build_reviewer(self.router, self.config)
            with self.events.stage(session_id, f"review_{attempt}", agent="reviewer"):
                review = reviewer.run(
                    f"{preamble}TASK: {task}\n\nCODER SUMMARY:\n{last_coder_summary}\n\n"
                    f"TESTER VERDICT:\n{tester_verdict}\n\nReview and decide.",
                    self.events, session_id, approval_fn=approval_fn,
                )
            self.events.append(session_id, "artifact", {"name": f"review_{attempt}", "content": review})

            if "DECISION: APPROVE" in review.upper().replace(" ", " "):
                verdict = "approved"
                break
            else:
                coder_feedback = f"Reviewer requested changes:\n{review}"

        committed = False
        commit_sha = None
        if verdict == "approved":
            diff_stat = git_ops.diff_stat(repo)
            if diff_stat.strip():
                if auto_commit or (approval_fn and approval_fn("git_commit", {"summary": last_coder_summary})):
                    say("Committing...")
                    git_ops.stage_all(repo)
                    msg = f"loom: {task}\n\n{last_coder_summary[:500]}"
                    commit_sha = git_ops.commit(repo, msg, session_id)
                    committed = True
                    self.events.append(session_id, "commit", {"sha": commit_sha})
            else:
                verdict = "no_changes"

        self.events.set_session_status(session_id, "completed" if verdict == "approved" else "failed")
        return RunResult(session_id=session_id, committed=committed, commit_sha=commit_sha,
                          verdict=verdict, branch=branch)
