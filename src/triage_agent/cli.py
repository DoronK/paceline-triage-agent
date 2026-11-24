"""`triage <issue_id>`: run one triage from the terminal and print the report."""
from __future__ import annotations

import argparse
import json
import sys

from .agent import TriageError
from .config import load_env
from .service import run_triage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="triage", description="Triage one Sentry issue and print the report as JSON.")
    parser.add_argument("issue_id", help="Sentry issue ID")
    args = parser.parse_args(argv)

    load_env()
    try:
        report = run_triage(args.issue_id)
    except TriageError as e:
        print(f"triage failed: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:  # configuration errors (missing env vars, provider)
        print(f"config error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
