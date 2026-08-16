"""
loom.config
------------
Central configuration: API credentials, model routing tiers, and the
team-shared `.loom/` directory (guardrails, custom skills, preferences).

Design note: "local vs cloud" from the original blueprint is scoped down
in this build to "cheap/fast model vs. frontier model" — both served via
the Anthropic API. A true local-model backend (Ollama/vLLM) is a clean
extension point (see router.py: ModelRouter.complete) but is not wired
up here, to keep this build honest about what's actually implemented.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_MODELS = {
    "anthropic": {
        "fast": "claude-haiku-4-5-20251001",   # cheap triage / simple reads
        "core": "claude-sonnet-5",              # planning, coding, review
        "deep": "claude-opus-4-8",              # hard architectural planning
    },
    # No safe default model names for other providers — naming conventions
    # vary by vendor and go stale fast. Set explicitly via:
    #   loom keys add openai sk-... --model core=gpt-5.1,fast=gpt-5.1-mini
    "openai": {},
}

DEFAULT_PROVIDER_PRIORITY = ["anthropic", "openai"]

LOOM_DIR_NAME = ".loom"


@dataclass
class TeamConfig:
    """Loaded from <repo>/.loom/config.json if present. Shared via git."""
    guardrails: list[str] = field(default_factory=list)   # e.g. "never edit /migrations"
    forbidden_paths: list[str] = field(default_factory=list)
    approval_required_for: list[str] = field(default_factory=lambda: ["shell", "git_push"])
    model_overrides: dict = field(default_factory=dict)      # {"anthropic": {"core": "..."}, ...}
    provider_priority: Optional[list[str]] = None            # e.g. ["openai", "anthropic"] to prefer OpenAI first
    custom_skills_dir: Optional[str] = None

    @classmethod
    def load(cls, repo_root: Path) -> "TeamConfig":
        cfg_path = repo_root / LOOM_DIR_NAME / "config.json"
        if not cfg_path.exists():
            return cls()
        try:
            data = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            return cls()
        return cls(
            guardrails=data.get("guardrails", []),
            forbidden_paths=data.get("forbidden_paths", []),
            approval_required_for=data.get("approval_required_for", ["shell", "git_push"]),
            model_overrides=data.get("model_overrides", {}),
            provider_priority=data.get("provider_priority"),
            custom_skills_dir=data.get("custom_skills_dir"),
        )


@dataclass
class LoomConfig:
    repo_root: Path
    models: dict = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULT_MODELS.items()})
    provider_priority: list[str] = field(default_factory=lambda: list(DEFAULT_PROVIDER_PRIORITY))
    team: TeamConfig = field(default_factory=TeamConfig)
    auto_approve: bool = False
    max_agent_turns: int = 40

    @classmethod
    def load(cls, repo_root: Optional[Path] = None, auto_approve: bool = False) -> "LoomConfig":
        root = (repo_root or Path.cwd()).resolve()
        team = TeamConfig.load(root)
        models = {k: dict(v) for k, v in DEFAULT_MODELS.items()}
        for provider, overrides in team.model_overrides.items():
            models.setdefault(provider, {}).update(overrides)
        priority = team.provider_priority or list(DEFAULT_PROVIDER_PRIORITY)
        return cls(repo_root=root, models=models, provider_priority=priority, team=team, auto_approve=auto_approve)

    def loom_dir(self) -> Path:
        d = self.repo_root / LOOM_DIR_NAME
        d.mkdir(exist_ok=True)
        self._ensure_gitignored()
        return d

    def _ensure_gitignored(self) -> None:
        """The event DB (and its WAL/SHM files) are local run state, not
        something to commit — auto-add `.loom/` to the repo's .gitignore
        so `loom run`'s own auto-commit never sweeps it in."""
        gitignore = self.repo_root / ".gitignore"
        entry = f"{LOOM_DIR_NAME}/"
        try:
            existing = gitignore.read_text() if gitignore.exists() else ""
            if entry not in existing.splitlines():
                sep = "" if (not existing or existing.endswith("\n")) else "\n"
                gitignore.write_text(existing + sep + entry + "\n")
        except OSError:
            pass  # best-effort; not fatal if repo_root isn't writable yet

    def event_db_path(self) -> Path:
        return self.loom_dir() / "events.sqlite3"
