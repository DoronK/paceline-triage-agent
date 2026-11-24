"""Eval runner: triage accuracy vs golden labels, tokens/cost, optional LLM judge.

Usage:
    python -m evals.run_evals --dataset evals/dataset/synthetic_sample.json
    python -m evals.run_evals --dataset ... --llm stub          # offline, CI
    python -m evals.run_evals --dataset ... --judge \\
        --price-in 3 --price-out 15                              # $ per 1M tokens

Live mode uses the provider configured by LLM_PROVIDER & co. Sentry is
replayed from the dataset, so no Sentry credentials are needed.
Reports only what it measures. No numbers are ever hand-written into docs.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals import harness


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def render_markdown(summary: dict[str, Any], model: str) -> str:
    lines = [f"## Eval results ({model})", "",
             f"Cases: {summary['cases']} · errors: {summary['errors']}", "",
             "| Field | Accuracy |", "|---|---|"]
    lines += [f"| {k} | {_pct(v)} |" for k, v in summary["accuracy"].items()]
    lines.append("")
    tok = summary["avg_tokens"]
    if tok:
        lines.append(f"Avg tokens per triage: {tok['input_tokens']:.0f} in / "
                     f"{tok['output_tokens']:.0f} out")
    cost = summary["avg_cost_usd"]
    lines.append("Avg cost per triage: " +
                 ("not computed (pass --price-in and --price-out)" if cost is None
                  else f"${cost:.5f}"))
    if summary["judged_cases"]:
        lines.append(f"LLM-judge avg score: {summary['judge_avg_score']:.2f} / 5 "
                     f"({summary['judged_cases']} cases)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="run_evals")
    p.add_argument("--dataset", required=True)
    p.add_argument("--llm", choices=("live", "stub"), default="live",
                   help="live: configured provider (default); stub: offline baseline")
    p.add_argument("--judge", action="store_true",
                   help="also score summary/action with the configured LLM as judge")
    p.add_argument("--price-in", type=float, help="$ per 1M input tokens")
    p.add_argument("--price-out", type=float, help="$ per 1M output tokens")
    p.add_argument("--rpm", type=float,
                   help="max LLM requests per minute (e.g. 5 for Gemini free tier)")
    p.add_argument("--out", help="report JSON path (default evals/results/run-<utc>.json)")
    p.add_argument("--validate-only", action="store_true")
    args = p.parse_args(argv)

    cases = json.loads(Path(args.dataset).read_text())
    harness.validate_dataset(cases)
    print(f"Loaded {len(cases)} cases from {args.dataset}. Dataset schema OK.")
    if args.validate_only:
        return 0

    llm = harness.StubLLM() if args.llm == "stub" else harness.live_llm()
    if args.rpm:
        llm = harness.Throttled(llm, args.rpm)
    judge = None
    if args.judge:
        if args.llm == "stub":
            print("--judge is skipped with --llm stub (no model to judge with).")
        else:
            judge = llm
    run = harness.run_eval(cases, llm, judge, args.price_in, args.price_out)

    model = args.llm if args.llm == "stub" else getattr(llm, "_model", type(llm).__name__)
    report = {"dataset": args.dataset, "llm": str(model),
              "created_at": datetime.now(timezone.utc).isoformat(), **run}
    out = Path(args.out or f"evals/results/run-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
    if str(out) != "/dev/null":
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))

    print(render_markdown(run["summary"], str(model)))
    for r in run["results"]:
        if r["error"]:
            first_line = r["error"].splitlines()[0][:200]
            print(f"  {r['id']}: ERROR {first_line}", file=sys.stderr)
    print(f"\nFull report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
