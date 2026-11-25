"""Report sinks, chosen by REPORT_SINK (unset = report stays in the response)."""
from __future__ import annotations

from .base import ReportSink, SinkError, issue_url

__all__ = ["ReportSink", "SinkError", "build_sink", "issue_url"]


def build_sink(settings) -> ReportSink | None:
    """The sink named by REPORT_SINK, or None when unset (current behavior)."""
    name = (settings.report_sink or "").strip().lower()
    if not name:
        return None
    if name == "slack":
        from .slack import SlackWebhookSink
        return SlackWebhookSink(settings.slack_webhook_url or "")
    if name == "github":
        from .github_sink import GitHubIssueSink
        return GitHubIssueSink(settings.github_token or "",
                               settings.github_repo or "")
    raise SinkError(f"unsupported REPORT_SINK: {name!r} (slack | github)")
