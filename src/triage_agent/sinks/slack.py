"""Slack sink: post a short triage summary to an incoming webhook."""
from __future__ import annotations

from typing import Any

import httpx

from .base import SinkError

_SEVERITY_EMOJI = {"critical": ":red_circle:", "high": ":large_orange_circle:",
                   "medium": ":large_yellow_circle:", "low": ":white_circle:"}


def format_slack_payload(report: dict[str, Any], issue_url: str) -> dict[str, Any]:
    """Pure: the incoming-webhook JSON for one report."""
    severity = str(report.get("severity", "unknown"))
    emoji = _SEVERITY_EMOJI.get(severity, ":question:")
    lines = [
        f"{emoji} *Triage: {severity.upper()}* - <{issue_url}|Sentry issue>",
        f"*Component:* {report.get('suspected_component', 'unknown')}",
        f"*Regression:* {'yes' if report.get('is_regression') else 'no'}"
        f" · *Confidence:* {report.get('confidence', 'unknown')}",
        f"*Action:* {report.get('recommended_action', 'n/a')}",
    ]
    return {"text": "\n".join(lines)}


class SlackWebhookSink:
    """Delivers reports to a Slack incoming webhook URL."""

    def __init__(self, webhook_url: str, client: httpx.Client | None = None):
        if not webhook_url:
            raise SinkError("SLACK_WEBHOOK_URL is required for REPORT_SINK=slack")
        self._url = webhook_url
        self._client = client or httpx.Client(timeout=15.0)

    def deliver(self, report: dict[str, Any], *, issue_id: str,
                issue_url: str) -> None:
        resp = self._client.post(self._url, json=format_slack_payload(report, issue_url))
        if resp.status_code != 200:
            raise SinkError(f"Slack webhook returned {resp.status_code}: "
                            f"{resp.text[:200]}")
