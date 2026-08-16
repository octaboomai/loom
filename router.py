"""
loom.router
------------
Resolves each agent role to a model tier (fast/core/deep), then tries
configured credentials in order until one succeeds. A "credential" is one
(provider, API key) pair — you can register any number of them, across
any number of providers, via `loom keys add`. If one is out of quota,
unauthenticated, or unreachable, the router transparently moves to the
next one and logs the swap; it does NOT fail over on a genuine bug (a
malformed request), since retrying that with a different key wouldn't help
— see providers/*.py for how each backend classifies its own errors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from loom.config import LoomConfig
from loom.credentials import Credential, CredentialStore
from loom.providers.base import LLMResponse, Provider, ProviderConnectionError, QuotaOrAuthError

SwapCallback = Callable[[Credential, Exception], None]


@dataclass
class UsageTracker:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    by_model: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, model_label: str, usage) -> None:
        self.calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        m = self.by_model.setdefault(model_label, {"input": 0, "output": 0, "calls": 0})
        m["input"] += usage.input_tokens
        m["output"] += usage.output_tokens
        m["calls"] += 1

    def summary(self) -> str:
        lines = [f"total: {self.calls} calls, {self.input_tokens} in / {self.output_tokens} out tokens"]
        for model, m in self.by_model.items():
            lines.append(f"  {model}: {m['calls']} calls, {m['input']} in / {m['output']} out")
        return "\n".join(lines)


class NoCredentialsError(RuntimeError):
    pass


class AllCredentialsFailedError(RuntimeError):
    pass


class ModelRouter:
    ROLE_TIER = {
        "triage": "fast",
        "planner": "deep",
        "coder": "core",
        "tester": "core",
        "reviewer": "core",
        "security": "core",
    }

    def __init__(self, config: LoomConfig, credential_store: Optional[CredentialStore] = None):
        self.config = config
        self.store = credential_store or CredentialStore()
        self.usage = UsageTracker()
        self._provider_cache: dict[tuple[str, str], Provider] = {}

    def tier_for(self, role: str) -> str:
        return self.ROLE_TIER.get(role, "core")

    def model_for(self, role: str) -> str:
        """Convenience accessor: the model that would be tried FIRST for this
        role, given current config/credentials. Mainly useful for logging/
        tests; complete() does its own resolution per attempt."""
        candidates = self._candidates(role)
        if candidates:
            return candidates[0][1]
        tier = self.tier_for(role)
        return self.config.models.get("anthropic", {}).get(tier, "unknown")

    def _candidates(self, role: str) -> list[tuple[Credential, str]]:
        tier = self.tier_for(role)
        env_creds = {c.provider: c for c in self.store.env_fallback()}
        candidates: list[tuple[Credential, str]] = []
        for provider in self.config.provider_priority:
            entries = self.store.for_provider(provider)
            if not entries and provider in env_creds:
                entries = [env_creds[provider]]
            for cred in entries:
                model = cred.models.get(tier) or self.config.models.get(provider, {}).get(tier)
                if model:
                    candidates.append((cred, model))
        return candidates

    def _get_provider(self, cred: Credential) -> Provider:
        key = (cred.provider, cred.name)
        if key not in self._provider_cache:
            if cred.provider == "anthropic":
                from loom.providers.anthropic_provider import AnthropicProvider
                self._provider_cache[key] = AnthropicProvider(cred.api_key, cred.base_url)
            elif cred.provider == "openai":
                from loom.providers.openai_provider import OpenAIProvider
                self._provider_cache[key] = OpenAIProvider(cred.api_key, cred.base_url)
            else:
                raise RuntimeError(f"Unknown provider: {cred.provider!r}")
        return self._provider_cache[key]

    def complete(
        self,
        role: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict]] = None,
        max_tokens: int = 4096,
        on_swap: Optional[SwapCallback] = None,
    ) -> LLMResponse:
        candidates = self._candidates(role)
        if not candidates:
            raise NoCredentialsError(
                "No API credentials configured. Either export ANTHROPIC_API_KEY (or OPENAI_API_KEY), "
                "or run: loom keys add anthropic sk-ant-...  /  loom keys add openai sk-... --model <model>"
            )

        failures: list[str] = []
        for cred, model in candidates:
            provider = self._get_provider(cred)
            try:
                resp = provider.complete(model=model, system=system, messages=messages,
                                          tools=tools, max_tokens=max_tokens)
            except (QuotaOrAuthError, ProviderConnectionError) as e:
                failures.append(f"{cred.provider}:{cred.name} ({model}) — {e}")
                if on_swap:
                    on_swap(cred, e)
                continue
            self.usage.record(resp.model_label, resp.usage)
            return resp

        raise AllCredentialsFailedError(
            "All configured API credentials failed:\n" + "\n".join(f"  - {f}" for f in failures)
        )
