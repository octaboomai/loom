"""
loom.providers.openai_provider
----------------------------------
Adapter for OpenAI's Chat Completions API — and, by pointing `base_url` at
a different host, anything that speaks the same wire format. That covers a
lot of ground as of this writing: OpenAI itself, OpenRouter, Groq,
Together, Fireworks, and self-hosted servers (Ollama, vLLM, LM Studio all
expose an OpenAI-compatible endpoint). One adapter, many usable backends —
that's the point, rather than writing a bespoke client per vendor.

The real work here is translation, not networking: our normalized message/
content-block format mirrors Anthropic's shape (tool results bundled
inside a user message, assistant tool calls as content blocks). OpenAI's
Chat Completions API wants tool results as their own separate `role:
"tool"` messages, and represents assistant tool calls via a `tool_calls`
field rather than inline content blocks. Both directions are handled below.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from loom.providers.base import LLMResponse, Provider, ProviderConnectionError, QuotaOrAuthError, Usage

FINISH_REASON_MAP = {
    "tool_calls": "tool_use",
    "stop": "end_turn",
    "length": "max_tokens",
}

BILLING_PHRASES = ("insufficient_quota", "quota", "billing", "credit")


def _to_openai_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]

    for msg in messages:
        role, content = msg["role"], msg["content"]

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        # content is a list of normalized blocks
        if content and all(b.get("type") == "tool_result" for b in content):
            # Anthropic-style: multiple tool results bundled in one user
            # message. OpenAI wants each as its own role:"tool" message.
            for block in content:
                out.append({
                    "role": "tool",
                    "tool_call_id": block["tool_use_id"],
                    "content": block["content"],
                })
            continue

        # assistant message: text block(s) + tool_use block(s)
        text_parts = [b["text"] for b in content if b.get("type") == "text"]
        tool_use_blocks = [b for b in content if b.get("type") == "tool_use"]
        entry: dict[str, Any] = {"role": role, "content": "\n".join(text_parts) or None}
        if tool_use_blocks:
            entry["tool_calls"] = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {"name": b["name"], "arguments": json.dumps(b["input"])},
                }
                for b in tool_use_blocks
            ]
        out.append(entry)

    return out


def _to_openai_tools(tools: Optional[list[dict]]) -> Optional[list[dict]]:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        try:
            import openai
        except ImportError as e:
            raise RuntimeError(
                "The 'openai' package isn't installed. Install it with: pip install -e \".[openai]\""
            ) from e
        self._openai = openai
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = openai.OpenAI(**kwargs)

    def complete(self, model: str, system: str, messages: list[dict[str, Any]],
                 tools: Optional[list[dict]], max_tokens: int) -> LLMResponse:
        o = self._openai
        oa_messages = _to_openai_messages(system, messages)
        oa_tools = _to_openai_tools(tools)
        kwargs: dict[str, Any] = dict(model=model, messages=oa_messages, max_tokens=max_tokens)
        if oa_tools:
            kwargs["tools"] = oa_tools

        try:
            resp = self.client.chat.completions.create(**kwargs)
        except o.APIConnectionError as e:
            raise ProviderConnectionError(str(e)) from e
        except (o.AuthenticationError, o.PermissionDeniedError, o.RateLimitError) as e:
            raise QuotaOrAuthError(str(e)) from e
        except o.BadRequestError as e:
            body_text = str(getattr(e, "body", "") or "") + str(e)
            if any(p in body_text.lower() for p in BILLING_PHRASES):
                raise QuotaOrAuthError(str(e)) from e
            raise

        choice = resp.choices[0]
        message = choice.message
        content: list[dict[str, Any]] = []
        if message.content:
            content.append({"type": "text", "text": message.content})
        for tc in (message.tool_calls or []):
            try:
                parsed_input = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                parsed_input = {"_raw_arguments": tc.function.arguments}
            content.append({"type": "tool_use", "id": tc.id, "name": tc.function.name, "input": parsed_input})

        usage = resp.usage
        return LLMResponse(
            content=content,
            stop_reason=FINISH_REASON_MAP.get(choice.finish_reason, "other"),
            usage=Usage(input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens),
            model_label=f"openai:{model}",
        )
