"""Agent-loop behaviour with a scripted fake LLM and a mocked Sentry API."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from triage_agent.agent import TriageAgent, TriageError, parse_report
from triage_agent.llm.anthropic_client import AnthropicClient, to_result
from triage_agent.llm.base import LLMResult, ToolCall, ToolResult
from triage_agent.sentry_client import SentryReadOnlyClient

REPORT = {
    "severity": "high",
    "suspected_component": "plan generation",
    "is_regression": True,
    "summary": "s",
    "recommended_action": "a",
    "confidence": "medium",
}


class FakeLLM:
    """Replays scripted LLMResults and records every request it saw."""

    def __init__(self, script: list[LLMResult]):
        self._script = list(script)
        self.seen: list[list[dict[str, Any]]] = []

    def step(self, system, messages, tools) -> LLMResult:
        self.seen.append(list(messages))
        return self._script.pop(0)

    def tool_results_messages(self, results: list[ToolResult]) -> list[dict[str, Any]]:
        return [{"role": "user", "results": results}]


def tool_step(*calls: ToolCall) -> LLMResult:
    return LLMResult("tool_use", {"role": "assistant", "content": "calls"},
                     tool_calls=list(calls), input_tokens=10, output_tokens=5)


def final_step(text: str) -> LLMResult:
    return LLMResult("final", {"role": "assistant", "content": text},
                     final_text=text, input_tokens=10, output_tokens=5)


def sentry(handler) -> SentryReadOnlyClient:
    return SentryReadOnlyClient("t", "org", "proj",
                                transport=httpx.MockTransport(handler))


def ok_sentry() -> SentryReadOnlyClient:
    return sentry(lambda req: httpx.Response(200, json={"id": "1", "title": "NPE"}))


def test_tool_results_are_linked_to_their_call_ids():
    llm = FakeLLM([tool_step(ToolCall("tu_1", "get_issue", {"issue_id": "1"})),
                   final_step(json.dumps(REPORT))])
    out = TriageAgent(llm, ok_sentry()).triage("1")

    assert out["report"] == REPORT
    assert out["usage"] == {"input_tokens": 20, "output_tokens": 10}
    second_request = llm.seen[1]
    assert second_request[1] == {"role": "assistant", "content": "calls"}
    (result,) = second_request[2]["results"]
    assert result.call_id == "tu_1" and not result.is_error
    assert json.loads(result.content)["title"] == "NPE"


def test_sentry_http_error_goes_back_to_the_model():
    llm = FakeLLM([tool_step(ToolCall("tu_1", "get_issue", {"issue_id": "9"})),
                   final_step(json.dumps(REPORT))])
    agent = TriageAgent(llm, sentry(lambda req: httpx.Response(404)))
    agent.triage("9")

    (result,) = llm.seen[1][2]["results"]
    assert result.is_error and "404" in result.content


def test_unknown_tool_and_missing_argument_are_error_results():
    llm = FakeLLM([tool_step(ToolCall("a", "resolve_issue", {}),
                             ToolCall("b", "get_issue", {})),
                   final_step(json.dumps(REPORT))])
    TriageAgent(llm, ok_sentry()).triage("1")

    results = llm.seen[1][2]["results"]
    assert [r.call_id for r in results] == ["a", "b"]
    assert all(r.is_error for r in results)


def test_fenced_report_with_prose_parses():
    text = "Here is the triage:\n```json\n" + json.dumps(REPORT) + "\n```\nDone."
    assert parse_report(text) == REPORT


def test_unparseable_answer_gets_one_corrective_turn():
    llm = FakeLLM([final_step("I think it is a crash."),
                   final_step(json.dumps(REPORT))])
    out = TriageAgent(llm, ok_sentry()).triage("1")

    assert out["report"] == REPORT
    assert "could not be parsed" in llm.seen[1][-1]["content"]


def test_report_missing_keys_is_rejected():
    with pytest.raises(ValueError, match="missing keys"):
        parse_report('{"severity": "low"}')


@pytest.mark.parametrize("stop", ["max_tokens", "refusal"])
def test_cut_off_or_refused_answer_stops_immediately(stop):
    llm = FakeLLM([LLMResult(stop, {"role": "assistant", "content": []})])
    with pytest.raises(TriageError):
        TriageAgent(llm, ok_sentry()).triage("1")
    assert len(llm.seen) == 1


def test_step_limit_raises():
    calls = [tool_step(ToolCall(str(i), "list_releases", {})) for i in range(8)]
    with pytest.raises(TriageError, match="8 steps"):
        TriageAgent(FakeLLM(calls), ok_sentry()).triage("1")


def test_anthropic_response_mapping_keeps_raw_content():
    content = [
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text="checking"),
        SimpleNamespace(type="tool_use", id="tu_7", name="get_issue",
                        input={"issue_id": "1"}),
    ]
    resp = SimpleNamespace(content=content, stop_reason="tool_use",
                           usage=SimpleNamespace(input_tokens=3, output_tokens=4))
    result = to_result(resp)

    assert result.stop_reason == "tool_use"
    assert result.final_text is None
    assert result.assistant_message["content"] is content
    assert result.tool_calls == [ToolCall("tu_7", "get_issue", {"issue_id": "1"})]


def test_anthropic_tool_results_message_shape():
    (msg,) = AnthropicClient.tool_results_messages(
        None, [ToolResult("tu_1", "{}"), ToolResult("tu_2", "boom", True)])
    assert msg["role"] == "user"
    assert [b["tool_use_id"] for b in msg["content"]] == ["tu_1", "tu_2"]
    assert all(b["type"] == "tool_result" for b in msg["content"])
    assert msg["content"][1]["is_error"] is True
