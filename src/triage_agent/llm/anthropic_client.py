"""Anthropic implementation of the swappable LLM layer."""
from __future__ import annotations

from typing import Any

import anthropic

from .base import LLMResult, StopReason, ToolCall, ToolResult

DEFAULT_MODEL = "claude-sonnet-5-5"

# Server-side refusal fallback: on a policy decline the API re-runs the
# request on a fallback model chosen by refusal category, inside the same call.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "final",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
}


def to_result(response: Any) -> LLMResult:
    """Map a Messages API response onto the provider-neutral LLMResult."""
    tool_calls = [
        ToolCall(id=block.id, name=block.name, arguments=block.input)
        for block in response.content
        if block.type == "tool_use"
    ]
    text = "".join(
        block.text for block in response.content if block.type == "text"
    )
    stop: StopReason = _STOP_REASONS.get(response.stop_reason, "other")
    return LLMResult(
        stop_reason=stop,
        # Append the content unchanged (thinking blocks included) so the
        # model keeps the record of its own reasoning and tool calls.
        assistant_message={"role": "assistant", "content": response.content},
        tool_calls=tool_calls,
        final_text=text if stop == "final" and text else None,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


class AnthropicClient:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def step(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResult:
        response = self._client.beta.messages.create(
            model=self._model,
            max_tokens=16000,
            system=system,
            messages=messages,
            tools=tools,
            betas=[_FALLBACK_BETA],
            fallbacks="default",
        )
        return to_result(response)

    def tool_results_messages(self, results: list[ToolResult]) -> list[dict[str, Any]]:
        # All results go back in one user message, linked by tool_use_id.
        return [{
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": r.call_id,
                    "content": r.content,
                    "is_error": r.is_error,
                }
                for r in results
            ],
        }]
