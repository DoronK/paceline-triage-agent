"""Scheduled poll: dedupe, retries, caps, state handling (no network)."""
from __future__ import annotations

import json

import pytest

from triage_agent.config import Settings
from triage_agent.poll import load_state, main, poll, save_state

SETTINGS = Settings(sentry_token="t", sentry_org="o", sentry_project="p")
STUBS = [{"id": "1", "title": "one"}, {"id": "2", "title": "two"},
         {"id": "3", "title": "three"}]
REPORT = {"severity": "low", "suspected_component": "x", "is_regression": False}


class FakeSentry:
    def __init__(self, stubs): self._stubs = stubs
    def __enter__(self): return self
    def __exit__(self, *a): return None
    def list_issues(self, query="", limit=25): return self._stubs[:limit]


@pytest.fixture
def patched(monkeypatch, tmp_path):
    monkeypatch.setattr("triage_agent.poll.SentryReadOnlyClient",
                        lambda *a, **k: FakeSentry(STUBS))
    calls = []

    def fake_run(issue_id, settings=None):
        calls.append(issue_id)
        if issue_id == "2":
            raise RuntimeError("llm down")
        return {"report": {**REPORT}, "usage": {"input_tokens": 1, "output_tokens": 1}}
    monkeypatch.setattr("triage_agent.poll.run_triage", fake_run)
    return {"calls": calls, "state": tmp_path / "state.json",
            "reports": tmp_path / "reports"}


def test_state_roundtrip_and_corrupt_fresh_start(tmp_path):
    path = tmp_path / "s.json"
    assert load_state(path) == {"triaged": [], "failed": {}, "last_run": None}
    save_state(path, {"triaged": ["1"], "failed": {}, "last_run": "t"})
    assert load_state(path)["triaged"] == ["1"]
    path.write_text("{not json")
    assert load_state(path)["triaged"] == []


def test_poll_triages_new_issues_and_records_failures(patched):
    out = poll(SETTINGS, state_path=patched["state"], reports_dir=patched["reports"])
    assert out["triaged"] == 2 and out["errors"] == 1
    state = load_state(patched["state"])
    assert sorted(state["triaged"]) == ["1", "3"]
    assert state["failed"] == {"2": "RuntimeError: llm down"}
    assert (patched["reports"] / "triage-1.json").exists()
    assert not (patched["reports"] / "triage-2.json").exists()


def test_poll_dedupes_and_retries_failures(patched):
    poll(SETTINGS, state_path=patched["state"], reports_dir=patched["reports"])
    assert patched["calls"] == ["1", "2", "3"]
    patched["calls"].clear()
    out = poll(SETTINGS, state_path=patched["state"], reports_dir=patched["reports"])
    # only the failed issue is retried; triaged ones are never re-run
    assert patched["calls"] == ["2"]
    assert out["errors"] == 1


def test_poll_respects_max_issues(patched):
    out = poll(SETTINGS, max_issues=1, state_path=patched["state"],
               reports_dir=patched["reports"])
    assert out["new_issues"] == 1 and patched["calls"] == ["1"]


def test_main_runs_end_to_end(patched, monkeypatch, capsys):
    monkeypatch.setenv("SENTRY_TOKEN", "t")
    monkeypatch.setenv("SENTRY_ORG", "o")
    monkeypatch.setenv("SENTRY_PROJECT", "p")
    monkeypatch.setenv("TRIAGE_STATE_PATH", str(patched["state"]))
    monkeypatch.setenv("TRIAGE_REPORTS_DIR", str(patched["reports"]))
    assert main(["--max", "3"]) == 1  # issue 2 fails -> exit 1
    out = capsys.readouterr().out
    assert "TRIAGED 1" in out and "2 triaged" in out
