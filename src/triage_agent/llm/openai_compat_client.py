"""OpenAI-compatible implementation of the swappable LLM layer.

One client covers every provider that speaks the OpenAI chat-completions
format with tool calls, including free ones: Google Gemini (AI Studio key),
Groq, OpenRouter ``:free`` models and a local Ollama. Talks HTTP through
httpx directly, so no extra SDK is needed.
"""
from __future__ import annotations

import json
from typing import Any

import httpx

from .base import LLMResult, StopReason, ToolCall, ToolResult

# Base URLs for the named presets; LLM_BASE_URL overrides any of them.
PRESET_BASE_URLS: dict[str, str] = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
}

_STOP_REASONS: dict[str, StopReason] = {
    "stop": "final",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


def to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the agent's Anthropic-shaped tool schemas to OpenAI functions."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def _arguments(raw: Any) -> dict[str, Any]:
    """Tool arguments arrive as a JSON string; a bad one becomes {}.

    An empty dict makes the dispatcher raise KeyError, which goes back to the
    model as an error result instead of crashing the loop.
    """
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def to_result(response: dict[str, Any]) -> LLMResult:
    """Map a chat-completions response onto the provider-neutral LLMResult."""
    choice = response["choices"][0]
    message = choice.get("message") or {}
    raw_calls = message.get("tool_calls") or []
    tool_calls = [
        ToolCall(id=c.get("id") or f"call_{i}",
                 name=c["function"]["name"],
                 arguments=_arguments(c["function"].get("arguments")))
        for i, c in enumerate(raw_calls)
    ]
    # Some providers report "stop" even when they return tool calls.
    stop: StopReason = ("tool_use" if tool_calls
                        else _STOP_REASONS.get(choice.get("finish_reason"), "other"))
    text = message.get("content") or ""

    assistant: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
    if raw_calls:
        # Keep each call as returned (extra fields such as Gemini's thought
        # signature included) but make sure every one carries the id we use.
        assistant["tool_calls"] = [
            {**c, "id": tc.id} for c, tc in zip(raw_calls, tool_calls)
        ]
    usage = response.get("usage") or {}
    return LLMResult(
        stop_reason=stop,
        assistant_message=assistant,
        tool_calls=tool_calls,
        final_text=text if stop == "final" and text else None,
        input_tokens=usage.get("prompt_tokens", 0) or 0,
        output_tokens=usage.get("completion_tokens", 0) or 0,
    )


class OpenAICompatClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 max_tokens: int = 8192, timeout: float = 120.0,
                 transport: httpx.BaseTransport | None = None):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(base_url=base_url.rstrip("/") + "/",
                                  headers=headers, timeout=timeout,
                                  transport=transport)
        self._model = model
        self._max_tokens = max_tokens

    def step(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResult:
        resp = self._http.post("chat/completions", json={
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "system", "content": system}, *messages],
            "tools": to_openai_tools(tools),
        })
        resp.raise_for_status()
        return to_result(resp.json())

    def tool_results_messages(self, results: list[ToolResult]) -> list[dict[str, Any]]:
        # One "tool" message per result, linked by tool_call_id. The format
        # has no error flag, so errors are marked in the text.
        return [
            {
                "role": "tool",
                "tool_call_id": r.call_id,
                "content": f"ERROR: {r.content}" if r.is_error else r.content,
            }
            for r in results
        ]
