"""Report sinks: formatting, delivery with a mocked HTTP layer, wiring."""
from __future__ import annotations

import httpx
import pytest
import respx

from triage_agent.config import Settings
from triage_agent.sinks import SinkError, build_sink, issue_url
from triage_agent.sinks.github_sink import GitHubIssueSink, format_github_issue
from triage_agent.sinks.slack import SlackWebhookSink, format_slack_payload

REPORT = {"severity": "high", "suspected_component": "plan generation",
          "is_regression": True, "summary": "NPE when generating a plan.",
          "recommended_action": "Add a null check in PlanGenerator.generate.",
          "confidence": "high"}
URL = "https://sentry.io/organizations/paceline-31/issues/123/"
SETTINGS = dict(sentry_token="t", sentry_org="o", sentry_project="p")


def test_issue_url_from_api_base():
    assert issue_url("https://sentry.io/api/0", "paceline-31", "123") == URL


def test_slack_payload_has_the_key_fields():
    text = format_slack_payload(REPORT, URL)["text"]
    assert "HIGH" in text and URL in text
    assert "plan generation" in text
    assert "null check" in text
    assert "yes" in text and "high" in text


def test_github_issue_format_and_labels_are_informational_only():
    issue = format_github_issue(REPORT, URL)
    assert issue["title"].startswith("[triage] high:")
    assert URL in issue["body"] and "PlanGenerator" in issue["body"]
    assert issue["labels"] == ["triage", "severity:high"]


@respx.mock
def test_slack_sink_posts_and_raises_on_failure():
    route = respx.post("https://hooks.slack.com/services/T/B/xxx").mock(
        side_effect=[httpx.Response(200, text="ok"),
                     httpx.Response(500, text="boom")])
    sink = SlackWebhookSink("https://hooks.slack.com/services/T/B/xxx")
    sink.deliver(REPORT, issue_id="123", issue_url=URL)
    assert route.called
    with pytest.raises(SinkError, match="500"):
        sink.deliver(REPORT, issue_id="123", issue_url=URL)


@respx.mock
def test_github_sink_creates_issue_with_auth():
    route = respx.post("https://api.github.com/repos/o/r/issues").mock(
        return_value=httpx.Response(201, json={"number": 42}))
    GitHubIssueSink("tok", "o/r").deliver(REPORT, issue_id="123", issue_url=URL)
    assert route.calls[0].request.headers["Authorization"] == "Bearer tok"


@respx.mock
def test_github_sink_retries_without_labels_on_422():
    route = respx.post("https://api.github.com/repos/o/r/issues").mock(
        side_effect=[httpx.Response(422, json={"message": "Validation Failed"}),
                     httpx.Response(201, json={"number": 43})])
    GitHubIssueSink("tok", "o/r").deliver(REPORT, issue_id="123", issue_url=URL)
    assert route.call_count == 2
    assert "labels" not in route.calls[1].request.read().decode()


def test_build_sink_defaults_to_none():
    assert build_sink(Settings(**SETTINGS)) is None


def test_build_sink_requires_credentials():
    with pytest.raises(SinkError, match="SLACK_WEBHOOK_URL"):
        build_sink(Settings(**SETTINGS, report_sink="slack"))
    with pytest.raises(SinkError, match="GITHUB_TOKEN"):
        build_sink(Settings(**SETTINGS, report_sink="github"))
    with pytest.raises(SinkError, match="unsupported"):
        build_sink(Settings(**SETTINGS, report_sink="pagerduty"))


def test_build_sink_picks_implementation():
    sink = build_sink(Settings(**SETTINGS, report_sink="slack",
                               slack_webhook_url="https://hooks.slack.com/x"))
    assert isinstance(sink, SlackWebhookSink)
    sink = build_sink(Settings(**SETTINGS, report_sink="github",
                               github_token="t", github_repo="o/r"))
    assert isinstance(sink, GitHubIssueSink)


def test_misconfigured_sink_keeps_the_report(monkeypatch):
    """REPORT_SINK=slack with no webhook must not lose a finished triage."""
    from triage_agent import service

    class FakeSentry:
        def __init__(self, **_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    class FakeAgent:
        def __init__(self, **_):
            pass

        def triage(self, issue_id):
            return {"report": dict(REPORT)}

    monkeypatch.setattr(service, "SentryReadOnlyClient", FakeSentry)
    monkeypatch.setattr(service, "TriageAgent", FakeAgent)
    out = service.run_triage("123", settings=Settings(**SETTINGS, report_sink="slack"),
                             llm=object())
    assert out["report"] == REPORT
    assert "SLACK_WEBHOOK_URL" in out["sink_error"]
