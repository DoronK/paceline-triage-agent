"""Fetch live Sentry issues as unlabeled candidates for the golden dataset.

Writes ``evals/dataset/golden_private.json`` (git-ignored, never committed):
each entry is a complete eval case - the issue, its latest event and the
project's releases - with an empty ``labels`` object for the owner to fill
in. ``validate_dataset`` refuses to score a case whose labels are not a
non-empty subset, so an unlabeled file fails loudly instead of producing
meaningless numbers.

The file contains real issue data. Keep it local; commit only synthetic or
scrubbed cases.

Usage:
    python -m evals.fetch_candidates --limit 25
    python -m evals.fetch_candidates --query "is:unresolved level:error" --limit 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from triage_agent.config import Settings, load_env
from triage_agent.sentry_client import SentryReadOnlyClient

DEFAULT_OUT = Path("evals/dataset/golden_private.json")
LABEL_KEYS = ("severity", "suspected_component", "is_regression")


def build_case(client: SentryReadOnlyClient, issue_stub: dict,
               releases: list[dict]) -> dict:
    """One full eval case for an issue from list_issues, labels left empty."""
    issue_id = str(issue_stub["id"])
    case = {
        "id": issue_id,
        "note": "TODO: fill 'labels' with the owner's verdict "
                f"(subset of {', '.join(LABEL_KEYS)}), then delete this note.",
        "issue": client.get_issue(issue_id),
        "releases": releases,
        "labels": {},
    }
    try:
        case["latest_event"] = client.get_latest_event(issue_id)
    except Exception as e:  # noqa: BLE001 - an issue can have no event yet
        print(f"  {issue_id}: no latest event ({type(e).__name__}), case kept without one",
              file=sys.stderr)
    return case


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fetch_candidates")
    p.add_argument("--limit", type=int, default=25,
                   help="max issues to fetch (default 25)")
    p.add_argument("--query", default="is:unresolved",
                   help="Sentry issue search query (default 'is:unresolved')")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help=f"output path (default {DEFAULT_OUT}, git-ignored)")
    args = p.parse_args(argv)

    load_env()
    settings = Settings.from_env()
    with SentryReadOnlyClient(settings.sentry_token, settings.sentry_org,
                              settings.sentry_project) as client:
        stubs = client.list_issues(query=args.query, limit=args.limit)
        if not stubs:
            print(f"No issues match {args.query!r}.", file=sys.stderr)
            return 1
        releases = client.list_releases(limit=25)
        cases = [build_case(client, stub, releases) for stub in stubs]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cases, indent=2, default=str) + "\n")
    print(f"Wrote {len(cases)} unlabeled cases to {out}")
    print("Next: fill each case's 'labels', then run:")
    print(f"  python -m evals.run_evals --dataset {out}")
    print("This file is git-ignored and contains real issue data - never commit it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
