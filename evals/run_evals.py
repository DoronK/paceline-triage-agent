"""Eval runner: triage accuracy vs golden labels + cost per triage.

Usage:
    python -m evals.run_evals --dataset evals/dataset/synthetic_sample.json

Reports only what it measures. No numbers are ever hand-written into docs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    cases = json.loads(Path(args.dataset).read_text())
    print(f"Loaded {len(cases)} cases from {args.dataset}")
    print("Eval execution against live agent is wired in PR #3.")
    # v1 scaffold: dataset validation only.
    for case in cases:
        assert "issue" in case and "labels" in case, f"malformed case: {case.get('id')}"
    print("Dataset schema OK.")


if __name__ == "__main__":
    main()
