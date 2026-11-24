"""Offline unit tests for eval comparison helpers (no network, no LLM)."""
from evals.compare import (compare_report, match_component, match_regression,
                           match_severity)


def test_severity_exact_case_insensitive():
    assert match_severity("High", "high")
    assert not match_severity("low", "high")


def test_regression_boolean():
    assert match_regression(True, True)
    assert not match_regression(False, True)


def test_component_substring_either_direction():
    assert match_component("Plan Generator", "plan generation")
    assert match_component("sentry", "Sentry SDK initialization")
    assert not match_component("network", "plan generation")
    assert not match_component("", "anything")


def test_compare_report_score():
    report = {"severity": "high", "suspected_component": "plan generation",
              "is_regression": True}
    labels = {"severity": "high", "suspected_component": "Plan Generator",
              "is_regression": False}
    checks = compare_report(report, labels)
    assert checks["severity"] is True
    assert checks["suspected_component"] is True
    assert checks["is_regression"] is False
    assert abs(checks["score"] - 2 / 3) < 1e-9
