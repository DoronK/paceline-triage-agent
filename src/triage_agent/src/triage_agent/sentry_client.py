"""Read-only Sentry REST API client. No write endpoints exist here by design."""
from __future__ import annotations

from typing import Any

import httpx


class SentryReadOnlyClient:
    def __init__(self, token: str, org: str, project: str,
                 base_url: str = "https://sentry.io/api/0"):
        self._org = org
        self._project = project
        self._http = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0,
        )

    def list_issues(self, query: str = "", limit: int = 25) -> list[dict[str, Any]]:
        resp = self._http.get(
            f"/projects/{self._org}/{self._project}/issues/",
            params={"query": query, "limit": limit},
        )
        resp.raise_for_status()
        return resp.json()

    def get_issue(self, issue_id: str) -> dict[str, Any]:
        resp = self._http.get(f"/issues/{issue_id}/")
        resp.raise_for_status()
        return resp.json()

    def get_latest_event(self, issue_id: str) -> dict[str, Any]:
        resp = self._http.get(f"/issues/{issue_id}/events/latest/")
        resp.raise_for_status()
        return resp.json()

    def list_releases(self, limit: int = 10) -> list[dict[str, Any]]:
        resp = self._http.get(
            f"/projects/{self._org}/{self._project}/releases/",
            params={"limit": limit},
        )
        resp.raise_for_status()
        return resp.json()
