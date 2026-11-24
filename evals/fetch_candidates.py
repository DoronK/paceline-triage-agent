"""Pull issues from the configured Sentry project and write a golden-dataset
template to evals/dataset/golden_private.json (git-ignored) for owner labeling.

The owner fills in labels (severity, suspected_component, is_regression) from
their knowledge of what actually happened; the file never leaves the machine.

Usage:
    python -m evals.fetch_candidates --limit 25 --query ""
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT_PATH = Path("evals/dataset/golden_private.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--query", default="")
    args = parser.parse_args()

    from triage_agent.config import Settings, load_env
    from triage_agent.sentry_client import SentryReadOnlyClient

    load_env()
    settings = Settings.from_env()
    with SentryReadOnlyClient(token=settings.sentry_token, org=settings.sentry_org,
                              project=settings.sentry_project) as sentry:
        issues = sentry.list_issues(query=args.query, limit=args.limit)

    cases = [
        {
            "id": i.get("shortId") or i.get("id"),
            "issue_id": str(i.get("id")),
            "title": i.get("title"),
            "status": i.get("status"),
            "count": i.get("count"),
            "firstSeen": i.get("firstSeen"),
            "lastSeen": i.get("lastSeen"),
            "labels": {
                "severity": "",
                "suspected_component": "",
                "is_regression": None,
            },
        }
        for i in issues
    ]
    OUT_PATH.write_text(json.dumps(cases, indent=2))
    print(f"wrote {len(cases)} candidates to {OUT_PATH} (git-ignored; "
          f"fill labels before running live evals)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
