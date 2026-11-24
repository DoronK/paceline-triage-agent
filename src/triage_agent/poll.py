"""Scheduled poll: triage new unresolved Sentry issues, each exactly once.

State lives in a small JSON file (default ``.triage_poll_state.json``,
git-ignored; override with TRIAGE_STATE_PATH). In CI the workflow caches
that file between runs. An issue whose triage raises is recorded under
``failed`` with its error and retried on the next run - it is never marked
triaged until a report exists.

Usage:
    python -m triage_agent.poll                 # triage up to 5 new issues
    python -m triage_agent.poll --max 10 --query "is:unresolved level:error"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings, load_env
from .sentry_client import SentryReadOnlyClient
from .service import run_triage

DEFAULT_STATE = Path(".triage_poll_state.json")
DEFAULT_REPORTS_DIR = Path("evals/results")


def load_state(path: Path) -> dict[str, Any]:
    """The poll state, or a fresh one. Corrupt files start fresh, never crash."""
    try:
        state = json.loads(path.read_text())
        if isinstance(state.get("triaged"), list):
            return state
    except (OSError, ValueError):
        pass
    return {"triaged": [], "failed": {}, "last_run": None}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(json.dumps(state, indent=2) + "\n")


def poll(settings: Settings, *, max_issues: int = 5, query: str = "is:unresolved",
         state_path: Path = DEFAULT_STATE,
         reports_dir: Path = DEFAULT_REPORTS_DIR) -> dict[str, Any]:
    """Triage up to max_issues not-yet-triaged issues. Returns a summary."""
    state = load_state(state_path)
    done = set(state["triaged"])
    with SentryReadOnlyClient(settings.sentry_token, settings.sentry_org,
                              settings.sentry_project,
                              base_url=settings.sentry_base_url) as client:
        stubs = client.list_issues(query=query, limit=50)
    new = [s for s in stubs if str(s["id"]) not in done][:max_issues]

    results: list[dict[str, Any]] = []
    for stub in new:
        issue_id = str(stub["id"])
        entry: dict[str, Any] = {"id": issue_id,
                                 "title": (stub.get("title") or "")[:80]}
        try:
            out = run_triage(issue_id, settings=settings)
        except Exception as e:  # noqa: BLE001 - retried next run
            entry["error"] = f"{type(e).__name__}: {e}"
            state["failed"][issue_id] = entry["error"]
        else:
            entry["report"] = out["report"]
            state["triaged"].append(issue_id)
            state["failed"].pop(issue_id, None)
            if reports_dir is not None:
                reports_dir.mkdir(parents=True, exist_ok=True)
                (reports_dir / f"triage-{issue_id}.json").write_text(
                    json.dumps(out, indent=2, default=str) + "\n")
        results.append(entry)

    state["last_run"] = datetime.now(timezone.utc).isoformat()
    save_state(state_path, state)
    return {"new_issues": len(new), "triaged": sum(1 for r in results if "report" in r),
            "errors": sum(1 for r in results if "error" in r),
            "total_triaged": len(state["triaged"]), "results": results}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="poll")
    p.add_argument("--max", type=int, default=5, dest="max_issues",
                   help="max new issues to triage this run (default 5)")
    p.add_argument("--query", default="is:unresolved",
                   help="Sentry issue search query (default 'is:unresolved')")
    args = p.parse_args(argv)

    load_env()
    state_path = Path(os.environ.get("TRIAGE_STATE_PATH", str(DEFAULT_STATE)))
    reports_dir = Path(os.environ.get("TRIAGE_REPORTS_DIR", str(DEFAULT_REPORTS_DIR)))
    summary = poll(Settings.from_env(), max_issues=args.max_issues,
                   query=args.query, state_path=state_path, reports_dir=reports_dir)

    for r in summary["results"]:
        if "report" in r:
            rep = r["report"]
            print(f"TRIAGED {r['id']} [{rep.get('severity')}] "
                  f"{r['title']} -> {rep.get('suspected_component')}")
        else:
            print(f"FAILED  {r['id']} {r['title']}: {r['error']}", file=sys.stderr)
    print(f"poll: {summary['triaged']} triaged, {summary['errors']} failed, "
          f"{summary['total_triaged']} triaged all-time")
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
