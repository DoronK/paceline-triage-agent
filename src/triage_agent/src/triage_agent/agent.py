"""The tool-calling triage agent loop.

The agent receives a Sentry issue id, investigates with read-only tools, and
returns a structured triage report. Loop is bounded and every step is logged
with token usage so cost per triage is measurable.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .llm.base import LLMClient
from .sentry_client import SentryReadOnlyClient

MAX_STEPS = 8

SYSTEM_PROMPT = """You are a mobile crash/performance triage engineer for an
Android app. Investigate the given Sentry issue using the read-only tools.
When you have enough evidence, answer with ONLY a JSON object:
{
  "severity": "critical|high|medium|low",
  "suspected_component": "string",
  "is_regression": true|false,
  "summary": "one paragraph, plain language",
  "recommended_action": "one concrete next step",
  "confidence": "high|medium|low"
}
Do not invent stack frames, releases or numbers you did not fetch."""


def _tools_schema() -> list[dict[str, Any]]:
    return [
        {
            "name": "get_issue",
            "description": "Fetch Sentry issue metadata (title, counts, first/last seen, status).",
            "input_schema": {
                "type": "object",
                "properties": {"issue_id": {"type": "string"}},
                "required": ["issue_id"],
            },
        },
        {
            "name": "get_latest_event",
            "description": "Fetch the latest event (stacktrace, tags, context) for an issue.",
            "input_schema": {
                "type": "object",
                "properties": {"issue_id": {"type": "string"}},
                "required": ["issue_id"],
            },
        },
        {
            "name": "list_releases",
            "description": "List recent app releases (to assess regressions).",
            "input_schema": {"type": "object", "properties": {}},
        },
    ]


class TriageAgent:
    def __init__(self, llm: LLMClient, sentry: SentryReadOnlyClient,
                 on_event: Callable[[str, dict[str, Any]], None] | None = None):
        self._llm = llm
        self._sentry = sentry
        self._emit = on_event or (lambda kind, data: None)

    def _dispatch(self, name: str, args: dict[str, Any]) -> Any:
        if name == "get_issue":
            return self._sentry.get_issue(args["issue_id"])
        if name == "get_latest_event":
            return self._sentry.get_latest_event(args["issue_id"])
        if name == "list_releases":
            return self._sentry.list_releases()
        raise ValueError(f"unknown tool: {name}")

    def triage(self, issue_id: str) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": f"Triage Sentry issue {issue_id}."}
        ]
        total_in = total_out = 0
        for _ in range(MAX_STEPS):
            result = self._llm.step(SYSTEM_PROMPT, messages, _tools_schema())
            total_in += result.input_tokens
            total_out += result.output_tokens
            if result.final_text is not None:
                report = json.loads(result.final_text)
                self._emit("usage", {"input_tokens": total_in,
                                     "output_tokens": total_out})
                return {"issue_id": issue_id, "report": report,
                        "usage": {"input_tokens": total_in,
                                  "output_tokens": total_out}}
            tool_results = []
            for call in result.tool_calls:
                data = self._dispatch(call.name, call.arguments)
                self._emit("tool_call", {"name": call.name,
                                         "arguments": call.arguments})
                tool_results.append({"tool": call.name, "result": data})
            messages.append({"role": "assistant",
                             "content": f"[tool calls: {len(result.tool_calls)}]"})
            messages.append({"role": "user",
                             "content": json.dumps(tool_results, default=str)})
        raise RuntimeError(f"triage did not complete within {MAX_STEPS} steps")
