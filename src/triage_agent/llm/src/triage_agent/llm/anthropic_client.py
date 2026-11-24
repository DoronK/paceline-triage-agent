"""Anthropic implementation of the swappable LLM layer."""
from __future__ import annotations

from typing import Any

import anthropic

from .base import LLMResult, ToolCall


class AnthropicClient:
    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-5"):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def step(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResult:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=system,
            messages=messages,
            tools=tools,
        )
        tool_calls = [
            ToolCall(name=block.name, arguments=block.input)
            for block in response.content
            if block.type == "tool_use"
        ]
        final_text = "".join(
            block.text for block in response.content if block.type == "text"
        ) or None
        return LLMResult(
            tool_calls=tool_calls,
            final_text=final_text if response.stop_reason == "end_turn" else None,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
