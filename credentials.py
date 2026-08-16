"""
loom.credentials
--------------------
Stores API keys OUTSIDE any git repo, at ~/.loom/credentials.json — never
inside a project's .loom/ directory (that one holds the event log and is
per-project). This means a key can never end up committed by accident, no
matter what a project's .gitignore does or doesn't say.

A user can register any number of keys, across any number of providers:

    loom keys add anthropic sk-ant-... --name work
    loom keys add anthropic sk-ant-... --name personal
    loom keys add openai sk-... --model gpt-5.1 --base-url https://openrouter.ai/api/v1 --name openrouter

The router tries them in the order they were added, per provider, and
tries providers in `provider_priority` order — see router.py for the
failover logic itself.
"""
from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

CREDENTIALS_PATH = Path.home() / ".loom" / "credentials.json"

# Providers whose native SDK needs no base_url override to reach their own
# API; anything else (OpenAI-compatible third parties, local servers) is
# expected to pass --base-url explicitly.
DEFAULT_BASE_URLS: dict[str, Optional[str]] = {"anthropic": None, "openai": None}

# Shortcuts for `loom keys add <name> ...` — every one of these is an
# OpenAI-compatible endpoint under the hood, so they all route through the
# same OpenAIProvider adapter; this just saves typing --base-url. Verified
# against each vendor's own docs as of Aug 2026 — vendors do change these,
# so if a preset ever 404s, `--base-url` always overrides it directly.
PROVIDER_PRESETS: dict[str, str] = {
    "deepseek": "https://api.deepseek.com/v1",
    "kimi": "https://api.moonshot.ai/v1",
    "moonshot": "https://api.moonshot.ai/v1",       # alias for kimi
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
}


@dataclass
class Credential:
    provider: str                 # "anthropic" | "openai"
    name: str                     # user-chosen label, unique per provider
    api_key: str
    base_url: Optional[str] = None
    models: dict[str, str] = field(default_factory=dict)  # tier -> model string ("fast"/"core"/"deep")

    def masked_key(self) -> str:
        if len(self.api_key) <= 8:
            return "*" * len(self.api_key)
        return self.api_key[:6] + "…" + self.api_key[-4:]


class CredentialStore:
    def __init__(self, path: Path = CREDENTIALS_PATH):
        self.path = path
        self._data: dict[str, list[dict]] = self._load()

    def _load(self) -> dict[str, list[dict]]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))
        try:
            os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)  # 0600 — best-effort; no-op on some Windows setups
        except OSError:
            pass

    def add(self, provider: str, api_key: str, name: str = "default",
            base_url: Optional[str] = None, models: Optional[dict[str, str]] = None) -> None:
        entries = self._data.setdefault(provider, [])
        entries[:] = [e for e in entries if e["name"] != name]  # replace if same name re-added
        entries.append(asdict(Credential(
            provider=provider, name=name, api_key=api_key,
            base_url=base_url or DEFAULT_BASE_URLS.get(provider),
            models=models or {},
        )))
        self._save()

    def remove(self, provider: str, name: str) -> bool:
        entries = self._data.get(provider, [])
        before = len(entries)
        entries[:] = [e for e in entries if e["name"] != name]
        self._save()
        return len(entries) < before

    def list_all(self) -> list[Credential]:
        out = []
        for provider, entries in self._data.items():
            for e in entries:
                out.append(Credential(**e))
        return out

    def for_provider(self, provider: str) -> list[Credential]:
        return [Credential(**e) for e in self._data.get(provider, [])]

    def providers_configured(self) -> list[str]:
        return [p for p, entries in self._data.items() if entries]

    def env_fallback(self) -> list[Credential]:
        """If no keys were registered via `loom keys add`, fall back to the
        classic single-key-via-environment-variable setup so existing users
        (and the earlier version of this README) keep working unchanged."""
        out = []
        if os.environ.get("ANTHROPIC_API_KEY"):
            out.append(Credential(provider="anthropic", name="env", api_key=os.environ["ANTHROPIC_API_KEY"]))
        if os.environ.get("OPENAI_API_KEY"):
            out.append(Credential(provider="openai", name="env", api_key=os.environ["OPENAI_API_KEY"],
                                   models={} ))
        return out
