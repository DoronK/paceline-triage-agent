"""Swappable LLM layer: any provider implements LLMClient.

The agent loop stays provider-neutral: each provider returns its own
assistant turn (``LLMResult.assistant_message``) and builds its own
tool-result turn (``LLMClient.tool_results_message``), so the native
tool-call / tool-result linkage survives the round trip.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

StopReason = Literal["final", "tool_use", "max_tokens", "refusal", "other"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """The outcome of one tool call, sent back to the model."""

    call_id: str
    content: str
    is_error: bool = False


@dataclass
class LLMResult:
    """One step of the agent loop: tool calls to run, or final text."""

    stop_reason: StopReason
    assistant_message: dict[str, Any]
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

    def tool_results_message(self, results: list[ToolResult]) -> dict[str, Any]:
        """Build the provider-native message that returns tool results."""
        ...
