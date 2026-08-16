"""
loom.providers.anthropic_provider
------------------------------------
Thin adapter — our normalized message/content-block format already mirrors
Anthropic's native shape (that was the original design), so this is mostly
error classification: turning SDK exceptions into the two failover-worthy
categories (QuotaOrAuthError, ProviderConnectionError) so the router can
transparently move to the next configured credential.
"""
from __future__ import annotations

from typing import Any, Optional

from loom.providers.base import LLMResponse, Provider, ProviderConnectionError, QuotaOrAuthError, Usage

STOP_REASON_MAP = {
    "tool_use": "tool_use",
    "end_turn": "end_turn",
    "max_tokens": "max_tokens",
    "stop_sequence": "end_turn",
}

# 400s that ARE a quota/billing problem contain one of these phrases in the
# error body — a 400 for some other reason (e.g. a malformed tool schema)
# should NOT trigger failover, since switching keys won't fix a bad request.
BILLING_PHRASES = ("credit balance", "quota", "billing", "insufficient")


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        import anthropic  # local import: keep this optional-ish and give a clear error if missing
        self._anthropic = anthropic
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = anthropic.Anthropic(**kwargs)

    def complete(self, model: str, system: str, messages: list[dict[str, Any]],
                 tools: Optional[list[dict]], max_tokens: int) -> LLMResponse:
        a = self._anthropic
        kwargs: dict[str, Any] = dict(model=model, max_tokens=max_tokens, system=system, messages=messages)
        if tools:
            kwargs["tools"] = tools
        try:
            resp = self.client.messages.create(**kwargs)
        except a.APIConnectionError as e:
            raise ProviderConnectionError(str(e)) from e
        except (a.AuthenticationError, a.PermissionDeniedError, a.RateLimitError) as e:
            raise QuotaOrAuthError(str(e)) from e
        except a.BadRequestError as e:
            body_text = str(getattr(e, "body", "") or "") + str(e)
            if any(p in body_text.lower() for p in BILLING_PHRASES):
                raise QuotaOrAuthError(str(e)) from e
            raise  # a genuinely malformed request — surface it, don't hide it as a credential problem

        content = []
        for block in resp.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})

        return LLMResponse(
            content=content,
            stop_reason=STOP_REASON_MAP.get(resp.stop_reason, "other"),
            usage=Usage(input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens),
            model_label=f"anthropic:{model}",
        )
