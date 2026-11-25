"""GitHub sink: file the triage report as an issue in the app repo.

Agent-filed issues carry only informational labels (``triage`` plus
``severity:<level>``) - never work labels. If the repo does not have those
labels, the issue is filed without labels rather than failing the delivery.
"""
from __future__ import annotations

from typing import Any

import httpx

from .base import SinkError

API = "https://api.github.com"


def format_github_issue(report: dict[str, Any], issue_url: str) -> dict[str, Any]:
    """Pure: title, body and labels for the GitHub issue."""
    severity = str(report.get("severity", "unknown"))
    title = (f"[triage] {severity}: "
             f"{str(report.get('summary', 'untitled issue'))[:80]}")
    body = (
        f"Automated crash triage for {issue_url}\n\n"
        f"- **Severity:** {severity}\n"
        f"- **Suspected component:** {report.get('suspected_component', 'unknown')}\n"
        f"- **Regression:** {'yes' if report.get('is_regression') else 'no'}\n"
        f"- **Confidence:** {report.get('confidence', 'unknown')}\n\n"
        f"**Summary**\n{report.get('summary', 'n/a')}\n\n"
        f"**Recommended action**\n{report.get('recommended_action', 'n/a')}\n\n"
        "---\nFiled by the Paceline triage agent. Verify before acting on it."
    )
    return {"title": title, "body": body,
            "labels": ["triage", f"severity:{severity}"]}


class GitHubIssueSink:
    """Creates one GitHub issue per triage report."""

    def __init__(self, token: str, repo: str, client: httpx.Client | None = None):
        if not token or not repo:
            raise SinkError("GITHUB_TOKEN and GITHUB_REPO are required for "
                            "REPORT_SINK=github")
        self._repo = repo
        self._client = client or httpx.Client(
            base_url=API, timeout=15.0,
            headers={"Authorization": f"Bearer {token}",
                     "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28"})

    def deliver(self, report: dict[str, Any], *, issue_id: str,
                issue_url: str) -> None:
        payload = format_github_issue(report, issue_url)
        resp = self._client.post(f"/repos/{self._repo}/issues", json=payload)
        if resp.status_code == 422:  # labels missing on the repo: retry bare
            resp = self._client.post(f"/repos/{self._repo}/issues",
                                     json={k: payload[k] for k in ("title", "body")})
        if resp.status_code not in (200, 201):
            raise SinkError(f"GitHub issue create returned {resp.status_code}: "
                            f"{resp.text[:200]}")
