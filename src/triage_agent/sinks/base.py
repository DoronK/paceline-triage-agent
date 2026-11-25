"""Pluggable report sinks: anywhere a triage report should go after a run.

A sink mirrors the LLMClient idea: the pipeline stays destination-neutral and
any object implementing ``ReportSink`` can be dropped in. With no sink
configured (REPORT_SINK unset) the report only exists in the response, as
before. Writing back to Sentry is deliberately out of scope (read-only
design).
"""
from __future__ import annotations

from typing import Any, Protocol


class SinkError(RuntimeError):
    """A sink delivery failed. Callers should log and keep the report."""


class ReportSink(Protocol):
    def deliver(self, report: dict[str, Any], *, issue_id: str,
                issue_url: str) -> None:
        """Deliver one triage report. Raises SinkError on failure."""
        ...


def issue_url(sentry_base_url: str, org: str, issue_id: str) -> str:
    """Browser URL for a Sentry issue, derived from the API base URL."""
    root = sentry_base_url.removesuffix("/api/0").rstrip("/")
    return f"{root}/organizations/{org}/issues/{issue_id}/"
