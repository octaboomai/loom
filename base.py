"""
loom.providers.base
----------------------
Every LLM backend (Anthropic, OpenAI-compatible, ...) implements the same
`Provider.complete()` signature and returns the same `LLMResponse` shape.
This is what lets loom.agents.base — the tool-use loop every agent
shares — stay completely unaware of which backend actually answered.

Design choice: normalized content blocks are plain dicts, not custom
classes. That keeps them trivially JSON-serializable for the event store
and directly reusable as the next request's conversation history for any
provider, with no extra wrapping/unwrapping step.

    {"type": "text", "text": "..."}
    {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
    {"type": "tool_result", "tool_use_id": "...", "content": "..."}
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class LLMResponse:
    content: list[dict[str, Any]]
    stop_reason: str          # normalized: "tool_use" | "end_turn" | "max_tokens" | "other"
    usage: Usage
    model_label: str = ""     # e.g. "anthropic:claude-sonnet-5" — which credential actually served this


class ProviderError(Exception):
    """Base class for provider failures. Not caught by the router's failover
    logic on its own — only the two subclasses below are, deliberately, so a
    genuine bug (bad tool schema, programming error) surfaces immediately
    instead of being silently swallowed as a false 'credential failed'."""


class QuotaOrAuthError(ProviderError):
    """Billing, quota, rate-limit, or authentication failure — the router
    should try the next configured credential."""


class ProviderConnectionError(ProviderError):
    """Network/timeout failure reaching the provider — also worth trying
    the next credential (which may point at a different, reachable host)."""


class Provider:
    """Base class for a backend adapter. Subclasses implement complete()."""

    name: str = "base"

    def complete(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict]],
        max_tokens: int,
    ) -> LLMResponse:
        raise NotImplementedError
