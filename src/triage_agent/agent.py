"""The tool-calling triage agent loop.

The agent receives a Sentry issue id, investigates with read-only tools, and
returns a structured triage report. Loop is bounded and every step is logged
with token usage so cost per triage is measurable.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

import httpx

from .llm.base import LLMClient, LLMResult, ToolResult
from .sentry_client import SentryReadOnlyClient

MAX_STEPS = 8

REPORT_KEYS = {
    "severity",
    "suspected_component",
    "is_regression",
    "summary",
    "recommended_action",
    "confidence",
}

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

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class TriageError(RuntimeError):
    """The triage could not produce a report."""


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


def parse_report(text: str) -> dict[str, Any]:
    """Pull the JSON report out of the model's answer.

    Tolerates a code fence or prose around the object. Raises ValueError when
    no JSON object with every report key is found.
    """
    fenced = _FENCE.search(text)
    candidate = fenced.group(1) if fenced else text
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in the answer")
    report = json.loads(candidate[start:end + 1])
    if not isinstance(report, dict):
        raise ValueError("the answer is not a JSON object")
    missing = REPORT_KEYS - report.keys()
    if missing:
        raise ValueError(f"report is missing keys: {sorted(missing)}")
    return report


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

    def _run_tools(self, result: LLMResult) -> list[ToolResult]:
        """Run each call; a failure goes back to the model as an error result."""
        outcomes = []
        for call in result.tool_calls:
            self._emit("tool_call", {"name": call.name,
                                     "arguments": call.arguments})
            try:
                data = self._dispatch(call.name, call.arguments)
                outcomes.append(ToolResult(call.id, json.dumps(data, default=str)))
            except httpx.HTTPStatusError as e:
                outcomes.append(ToolResult(
                    call.id, f"Sentry returned HTTP {e.response.status_code}",
                    is_error=True))
            except (httpx.HTTPError, KeyError, ValueError) as e:
                outcomes.append(ToolResult(
                    call.id, f"{type(e).__name__}: {e}", is_error=True))
        return outcomes

    def triage(self, issue_id: str) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": f"Triage Sentry issue {issue_id}."}
        ]
        total_in = total_out = 0

        def usage() -> dict[str, int]:
            return {"input_tokens": total_in, "output_tokens": total_out}

        for _ in range(MAX_STEPS):
            result = self._llm.step(SYSTEM_PROMPT, messages, _tools_schema())
            total_in += result.input_tokens
            total_out += result.output_tokens
            messages.append(result.assistant_message)

            if result.stop_reason == "refusal":
                raise TriageError("the model declined to triage this issue")
            if result.stop_reason == "max_tokens":
                raise TriageError("the model's answer was cut off at max_tokens")

            if result.stop_reason == "tool_use" and result.tool_calls:
                messages.extend(
                    self._llm.tool_results_messages(self._run_tools(result)))
                continue

            if result.final_text is None:
                raise TriageError(
                    f"model stopped ({result.stop_reason}) with no answer")
            try:
                report = parse_report(result.final_text)
            except ValueError as e:
                # Ask once more for the bare object; this counts as a step.
                messages.append({"role": "user", "content":
                                 f"Your answer could not be parsed ({e}). "
                                 "Reply with ONLY the JSON object."})
                continue
            self._emit("usage", usage())
            return {"issue_id": issue_id, "report": report, "usage": usage()}

        raise TriageError(f"triage did not complete within {MAX_STEPS} steps")
