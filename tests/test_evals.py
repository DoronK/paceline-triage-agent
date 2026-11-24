"""Eval harness: replayed Sentry, scoring, cost, judge, runner (no network)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import harness
from evals.harness import (
    StubLLM, component_matches, cost_usd, judge_report, replay_sentry,
    run_case, run_eval, score_report, validate_dataset,
)
from evals.run_evals import main
from triage_agent.llm.base import LLMResult

DATASET = Path(__file__).parent.parent / "evals" / "dataset" / "synthetic_sample.json"

CASE = {
    "id": "T-1",
    "issue": {"id": "42", "title": "NPE in PlanGenerator", "count": 10},
    "latest_event": {"entries": [{"type": "exception"}]},
    "releases": [{"version": "1.2.0"}],
    "labels": {"severity": "high", "suspected_component": "plan generation",
               "is_regression": True},
}


def test_replay_sentry_serves_case_data():
    with replay_sentry(CASE) as s:
        assert s.get_issue("42")["title"] == "NPE in PlanGenerator"
        assert s.get_latest_event("42") == CASE["latest_event"]
        assert s.list_releases() == [{"version": "1.2.0"}]


def test_replay_sentry_404s_unknown_issue_and_missing_event():
    import httpx
    case = {**CASE}
    del case["latest_event"]
    with replay_sentry(case) as s:
        with pytest.raises(httpx.HTTPStatusError):
            s.get_issue("999")
        with pytest.raises(httpx.HTTPStatusError):
            s.get_latest_event("42")


@pytest.mark.parametrize("pred,gold,ok", [
    ("plan generation", "plan generation", True),
    ("Plan Generation (PlanGenerator)", "plan generation", True),
    ("workout plan generator", "plan generation", True),
    ("GPS tracking", "plan generation", False),
    ("", "plan generation", False),
])
def test_component_matches(pred, gold, ok):
    assert component_matches(pred, gold) is ok


def test_score_report_per_field():
    report = {"severity": "HIGH", "suspected_component": "plan generator",
              "is_regression": False}
    assert score_report(report, CASE["labels"]) == {
        "severity": True, "suspected_component": True, "is_regression": False}


def test_score_report_only_scores_labelled_fields():
    assert score_report({"severity": "low"}, {"severity": "low"}) == {"severity": True}


def test_cost_usd_needs_both_prices():
    usage = {"input_tokens": 1_000_000, "output_tokens": 500_000}
    assert cost_usd(usage, 3.0, 15.0) == pytest.approx(10.5)
    assert cost_usd(usage, None, 15.0) is None


def test_run_case_with_stub_llm_scores_and_counts_tokens():
    result = run_case(CASE, StubLLM())
    assert result["error"] is None
    assert set(result["scores"]) == {"severity", "suspected_component", "is_regression"}
    assert result["usage"]["input_tokens"] > 0


def test_run_case_records_agent_failure():
    class Refuser(StubLLM):
        def step(self, system, messages, tools):
            return LLMResult("refusal", {"role": "assistant", "content": ""})

    result = run_case(CASE, Refuser())
    assert "declined" in result["error"]
    assert result["scores"] == {}


def test_judge_report_parses_score():
    class Judge(StubLLM):
        def step(self, system, messages, tools):
            assert tools == []
            return LLMResult("final", {"role": "assistant"},
                             final_text='```json\n{"score": 4, "reason": "ok"}\n```',
                             input_tokens=7, output_tokens=3)

    out = judge_report(Judge(), CASE, {"summary": "s", "recommended_action": "a"})
    assert out == {"score": 4, "reason": "ok",
                   "usage": {"input_tokens": 7, "output_tokens": 3}}


def test_judge_report_rejects_out_of_range():
    class Judge(StubLLM):
        def step(self, system, messages, tools):
            return LLMResult("final", {"role": "assistant"},
                             final_text='{"score": 9, "reason": "x"}')

    assert "error" in judge_report(Judge(), CASE, {"summary": "s"})


def test_run_eval_aggregates():
    summary = run_eval([CASE, {**CASE, "id": "T-2"}], StubLLM())["summary"]
    assert summary["cases"] == 2 and summary["errors"] == 0
    assert set(summary["accuracy"]) == {"severity", "suspected_component", "is_regression"}
    assert summary["avg_tokens"]["input_tokens"] > 0
    assert summary["avg_cost_usd"] is None


def test_validate_dataset_flags_bad_cases():
    with pytest.raises(ValueError, match="T-9"):
        validate_dataset([{"id": "T-9", "issue": {"id": "1"}}])
    with pytest.raises(ValueError, match="issue.id"):
        validate_dataset([{"id": "T-8", "issue": {}, "labels": {"severity": "low"}}])
    with pytest.raises(ValueError, match="severity"):
        validate_dataset([{"id": "T-7", "issue": {"id": "1"},
                           "labels": {"severity": "huge"}}])


def test_committed_dataset_is_valid():
    cases = json.loads(DATASET.read_text())
    validate_dataset(cases)
    assert len(cases) >= 8


def test_main_stub_writes_report(tmp_path, capsys):
    out = tmp_path / "r.json"
    assert main(["--dataset", str(DATASET), "--llm", "stub", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["summary"]["cases"] == len(data["results"])
    assert "severity" in capsys.readouterr().out


def test_main_validate_only(capsys):
    assert main(["--dataset", str(DATASET), "--validate-only"]) == 0
    assert "Dataset schema OK" in capsys.readouterr().out


def test_stub_is_default_only_when_asked(monkeypatch):
    # live mode builds the configured provider, never a stub silently
    called = {}
    monkeypatch.setattr(harness, "live_llm", lambda: called.setdefault("live", StubLLM()))
    assert main(["--dataset", str(DATASET), "--out", "/dev/null"]) == 0
    assert "live" in called


def test_throttled_spaces_calls():
    now = [0.0]
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        now[0] += s

    llm = harness.Throttled(StubLLM(), rpm=6, clock=lambda: now[0], sleep=sleep)
    msgs = [{"role": "user", "content": "Triage Sentry issue 42."}]
    llm.step("s", msgs, [])
    now[0] += 4.0
    llm.step("s", msgs, [])
    llm.step("s", msgs, [])
    assert sleeps == [pytest.approx(6.0), pytest.approx(10.0)]
