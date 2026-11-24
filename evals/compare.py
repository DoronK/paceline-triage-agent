"""Pure comparison helpers for eval scoring. Offline; no network or LLM here.

Matching rules (documented in README):
- severity: exact match on the enum string.
- is_regression: exact boolean match.
- suspected_component: case-insensitive substring in either direction, so
  "plan generation" matches "Plan Generator". A heuristic - review disagreements
  case by case rather than tightening the rule to inflate the score.
"""
from __future__ import annotations

from typing import Any

LABEL_FIELDS = ("severity", "suspected_component", "is_regression")


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def match_severity(got: Any, want: Any) -> bool:
    return _norm(got) == _norm(want)


def match_regression(got: Any, want: Any) -> bool:
    return bool(got) == bool(want)


def match_component(got: Any, want: Any) -> bool:
    g, w = _norm(got), _norm(want)
    if not g or not w:
        return False
    return g in w or w in g


def compare_report(report: dict[str, Any], labels: dict[str, Any]) -> dict[str, Any]:
    """Compare one triage report against golden labels.

    Returns {"severity": bool, "suspected_component": bool,
             "is_regression": bool, "score": float} where score is the share
    of matched fields.
    """
    checks = {
        "severity": match_severity(report.get("severity"), labels.get("severity")),
        "suspected_component": match_component(
            report.get("suspected_component"), labels.get("suspected_component")),
        "is_regression": match_regression(
            report.get("is_regression"), labels.get("is_regression")),
    }
    checks["score"] = sum(1 for ok in checks.values() if ok) / len(LABEL_FIELDS)
    return checks
