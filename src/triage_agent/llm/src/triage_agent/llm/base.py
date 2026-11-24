"""Swappable LLM layer: any provider implements LLMClient."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResult:
    """One step of the agent loop: either tool calls to run, or final text."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    final_text: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(Protocol):
    def step(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResult:
        """Run one reasoning step. Provider decides tool-call vs final answer."""
        ...
