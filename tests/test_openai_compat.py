"""OpenAI-compatible client: response mapping, request shape, provider wiring."""
from __future__ import annotations

import json

import httpx
import pytest

from triage_agent.agent import TriageAgent, _tools_schema
from triage_agent.config import Settings
from triage_agent.llm.anthropic_client import AnthropicClient
from triage_agent.llm.base import ToolCall, ToolResult
from triage_agent.llm.openai_compat_client import (
    OpenAICompatClient, to_openai_tools, to_result)
from triage_agent.main import build_llm
from triage_agent.sentry_client import SentryReadOnlyClient

REPORT = {
    "severity": "low", "suspected_component": "sync", "is_regression": False,
    "summary": "s", "recommended_action": "a", "confidence": "low",
}


def completion(message: dict, finish: str, usage=(7, 3)) -> dict:
    return {"choices": [{"message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}}


def call(id_, name, args) -> dict:
    return {"id": id_, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def test_tool_call_response_maps_and_keeps_extra_fields():
    raw = call("c1", "get_issue", {"issue_id": "1"})
    raw["extra_content"] = {"google": {"thought_signature": "sig"}}
    result = to_result(completion({"role": "assistant", "content": None,
                                   "tool_calls": [raw]}, "tool_calls"))

    assert result.stop_reason == "tool_use"
    assert result.tool_calls == [ToolCall("c1", "get_issue", {"issue_id": "1"})]
    assert result.assistant_message["tool_calls"][0]["extra_content"] == {
        "google": {"thought_signature": "sig"}}
    assert (result.input_tokens, result.output_tokens) == (7, 3)


def test_tool_calls_with_stop_finish_reason_still_count_as_tool_use():
    result = to_result(completion({"role": "assistant", "content": "",
                                   "tool_calls": [call("c1", "list_releases", {})]},
                                  "stop"))
    assert result.stop_reason == "tool_use"


def test_missing_call_id_and_bad_arguments_are_tolerated():
    raw = {"type": "function", "function": {"name": "get_issue", "arguments": "{oops"}}
    result = to_result(completion({"role": "assistant", "tool_calls": [raw]},
                                  "tool_calls"))
    (tc,) = result.tool_calls
    assert tc.id == "call_0" and tc.arguments == {}
    assert result.assistant_message["tool_calls"][0]["id"] == "call_0"


@pytest.mark.parametrize("finish,stop", [("stop", "final"), ("length", "max_tokens"),
                                         ("content_filter", "refusal"), ("odd", "other")])
def test_finish_reason_mapping(finish, stop):
    result = to_result(completion({"role": "assistant", "content": "hi"}, finish))
    assert result.stop_reason == stop
    assert result.final_text == ("hi" if stop == "final" else None)


def test_tools_convert_to_openai_functions():
    tools = to_openai_tools(_tools_schema())
    assert {t["function"]["name"] for t in tools} == {
        "get_issue", "get_latest_event", "list_releases"}
    assert all(t["type"] == "function" and "parameters" in t["function"]
               for t in tools)


def test_tool_results_are_one_message_each_with_error_marked():
    msgs = OpenAICompatClient("http://x", "m").tool_results_messages(
        [ToolResult("c1", "{}"), ToolResult("c2", "boom", is_error=True)])
    assert msgs == [
        {"role": "tool", "tool_call_id": "c1", "content": "{}"},
        {"role": "tool", "tool_call_id": "c2", "content": "ERROR: boom"},
    ]


def test_full_triage_round_trip_over_http():
    """Two LLM turns through a mocked endpoint, tool results linked by id."""
    requests: list[dict] = []
    replies = [
        completion({"role": "assistant", "content": None,
                    "tool_calls": [call("c1", "get_issue", {"issue_id": "1"})]},
                   "tool_calls"),
        completion({"role": "assistant", "content": json.dumps(REPORT)}, "stop"),
    ]

    def llm_handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1beta/openai/chat/completions"
        assert req.headers["authorization"] == "Bearer k"
        requests.append(json.loads(req.content))
        return httpx.Response(200, json=replies.pop(0))

    llm = OpenAICompatClient(
        "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-x",
        api_key="k", transport=httpx.MockTransport(llm_handler))
    sentry = SentryReadOnlyClient("t", "o", "p", transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"id": "1", "title": "NPE"})))

    out = TriageAgent(llm, sentry).triage("1")

    assert out["report"] == REPORT
    assert out["usage"] == {"input_tokens": 14, "output_tokens": 6}
    first, second = requests
    assert first["model"] == "gemini-x"
    assert first["messages"][0]["role"] == "system"
    assert second["messages"][2]["tool_calls"][0]["id"] == "c1"
    tool_msg = second["messages"][3]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c1"
    assert json.loads(tool_msg["content"])["title"] == "NPE"


def test_no_api_key_sends_no_auth_header():
    def handler(req: httpx.Request) -> httpx.Response:
        assert "authorization" not in req.headers
        return httpx.Response(200, json=completion(
            {"role": "assistant", "content": "x"}, "stop"))

    OpenAICompatClient("http://localhost:11434/v1", "qwen",
                       transport=httpx.MockTransport(handler)).step("s", [], [])


def settings(**kw) -> Settings:
    return Settings(sentry_token="t", sentry_org="o", sentry_project="p", **kw)


def test_build_llm_picks_provider():
    assert isinstance(build_llm(settings(anthropic_api_key="k")), AnthropicClient)
    gem = build_llm(settings(llm_provider="gemini", llm_model="m", llm_api_key="k"))
    assert isinstance(gem, OpenAICompatClient)
    assert str(gem._http.base_url).startswith(
        "https://generativelanguage.googleapis.com/v1beta/openai")
    custom = build_llm(settings(llm_provider="ollama", llm_model="m",
                                llm_base_url="http://box:11434/v1"))
    assert str(custom._http.base_url) == "http://box:11434/v1/"


@pytest.mark.parametrize("kw,msg", [
    ({"llm_provider": "gemini"}, "LLM_MODEL"),
    ({"llm_provider": "openai_compat", "llm_model": "m"}, "LLM_BASE_URL"),
    ({"llm_provider": "nope"}, "unsupported"),
])
def test_build_llm_rejects_incomplete_config(kw, msg):
    with pytest.raises(RuntimeError, match=msg):
        build_llm(settings(**kw))
