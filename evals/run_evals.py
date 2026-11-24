"""Eval runner: executes the triage agent over a golden dataset and reports
per-field accuracy and measured cost. Reports only what it measures - no
numbers are ever hand-written into docs.

Dataset formats:
- Golden (live): {"id", "issue_id", "labels": {...}} - runs the real agent
  against Sentry. Requires SENTRY_* and provider env vars (.env supported).
- Synthetic (offline/CI): cases without "issue_id" are schema-validated only.

Usage:
    python -m evals.run_evals --dataset evals/dataset/synthetic_sample.json
    python -m evals.run_evals --dataset evals/dataset/golden_private.json --out results.json
    python -m evals.run_evals --dataset ... --judge        # adds LLM-as-judge (1-5)
    python -m evals.run_evals --dataset ... --price-in 3.0 --price-out 15.0
        # USD per 1M tokens; without these, cost is reported in tokens only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .compare import LABEL_FIELDS, compare_report

JUDGE_PROMPT = """You are evaluating a crash-triage report for an Android app.
Sentry issue (ground truth metadata):
{issue}

Triage report produced by the agent:
{report}

Score the report's faithfulness to the evidence on a 1-5 scale
(1 = invented claims, 5 = fully grounded). Answer with ONLY JSON:
{"score": <1-5>, "rationale": "one sentence"}"""


def _validate(cases: list[dict]) -> list[str]:
    errors = []
    for case in cases:
        if "issue" not in case and "issue_id" not in case:
            errors.append(f"{case.get('id', '?')}: needs issue or issue_id")
        labels = case.get("labels")
        if not isinstance(labels, dict) or not all(k in labels for k in LABEL_FIELDS):
            errors.append(f"{case.get('id', '?')}: labels must cover {LABEL_FIELDS}")
    return errors


def _judge(llm, issue: dict, report: dict) -> dict | None:
    from triage_agent.llm.base import LLMResult  # local import: eval-only path
    result: LLMResult = llm.step(
        "You are a strict but fair evaluator.",
        [{"role": "user",
          "content": JUDGE_PROMPT.format(issue=json.dumps(issue, default=str)[:4000],
                                         report=json.dumps(report, default=str))}],
        [],
    )
    if result.final_text:
        try:
            return json.loads(result.final_text)
        except json.JSONDecodeError:
            return {"score": None, "rationale": "judge returned non-JSON"}
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", help="write per-case results JSON here")
    parser.add_argument("--judge", action="store_true",
                        help="add LLM-as-judge faithfulness score (1-5)")
    parser.add_argument("--price-in", type=float, help="USD per 1M input tokens")
    parser.add_argument("--price-out", type=float, help="USD per 1M output tokens")
    args = parser.parse_args()

    cases = json.loads(Path(args.dataset).read_text())
    errors = _validate(cases)
    if errors:
        for e in errors:
            print(f"dataset error: {e}", file=sys.stderr)
        return 2

    live_cases = [c for c in cases if "issue_id" in c]
    if not live_cases:
        print(f"Loaded {len(cases)} offline cases from {args.dataset}")
        print("Dataset schema OK (no issue_id fields -> nothing to execute).")
        return 0

    # Live execution path: real agent against Sentry.
    from triage_agent.config import Settings, load_env
    from triage_agent.service import build_llm, run_triage
    from triage_agent.sentry_client import SentryReadOnlyClient

    load_env()
    settings = Settings.from_env()
    judge_llm = build_llm(settings) if args.judge else None
    results = []
    total_in = total_out = 0

    for case in live_cases:
        entry = {"id": case.get("id"), "issue_id": case["issue_id"]}
        try:
            outcome = run_triage(case["issue_id"], settings=settings)
        except Exception as e:  # noqa: BLE001 - record and continue the batch
            entry["error"] = str(e)
            results.append(entry)
            print(f"{entry['id']}: ERROR {e}")
            continue
        report = outcome.get("report", outcome)
        usage = outcome.get("usage", {})
        total_in += usage.get("input_tokens", 0)
        total_out += usage.get("output_tokens", 0)
        entry["report"] = report
        entry["usage"] = usage
        entry["checks"] = compare_report(report, case["labels"])
        if judge_llm is not None:
            with SentryReadOnlyClient(token=settings.sentry_token,
                                      org=settings.sentry_org,
                                      project=settings.sentry_project) as s:
                issue_meta = s.get_issue(case["issue_id"])
            entry["judge"] = _judge(judge_llm, issue_meta, report)
        c = entry["checks"]
        print(f"{entry['id']}: score {c['score']:.2f} "
              f"(severity={c['severity']}, component={c['suspected_component']}, "
              f"regression={c['is_regression']}) "
              f"tokens in={usage.get('input_tokens', 0)} out={usage.get('output_tokens', 0)}")
        results.append(entry)

    scored = [r for r in results if "checks" in r]
    summary = {
        "cases": len(live_cases),
        "completed": len(scored),
        "mean_score": (sum(r["checks"]["score"] for r in scored) / len(scored)
                       if scored else None),
        "tokens": {"input": total_in, "output": total_out},
    }
    if args.price_in is not None and args.price_out is not None:
        summary["cost_usd"] = (total_in * args.price_in
                               + total_out * args.price_out) / 1_000_000
    print(json.dumps({"summary": summary}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary,
                                              "results": results}, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
