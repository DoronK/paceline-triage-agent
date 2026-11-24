"""CLI entry point and .env loading (no network)."""
from __future__ import annotations

import json

import pytest

from triage_agent import cli
from triage_agent.agent import TriageError
from triage_agent.config import load_env

SENTRY_VARS = ("SENTRY_TOKEN", "SENTRY_ORG", "SENTRY_PROJECT")


@pytest.fixture
def clean_env(monkeypatch):
    for var in (*SENTRY_VARS, "LLM_PROVIDER", "TRIAGE_TEST_VAR"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_load_env_reads_file(tmp_path, clean_env):
    env = tmp_path / ".env"
    env.write_text('TRIAGE_TEST_VAR="from file"\n')
    load_env(env)
    import os
    assert os.environ["TRIAGE_TEST_VAR"] == "from file"


def test_load_env_does_not_override_existing_vars(tmp_path, clean_env):
    clean_env.setenv("TRIAGE_TEST_VAR", "from shell")
    env = tmp_path / ".env"
    env.write_text("TRIAGE_TEST_VAR=from file\n")
    load_env(env)
    import os
    assert os.environ["TRIAGE_TEST_VAR"] == "from shell"


def test_load_env_missing_file_is_fine(tmp_path, clean_env):
    load_env(tmp_path / "nope.env")  # no exception


def test_cli_prints_report_json(monkeypatch, capsys, clean_env):
    seen = {}

    def fake_run(issue_id, settings=None):
        seen["issue_id"] = issue_id
        return {"severity": "high", "issue_id": issue_id}

    monkeypatch.setattr(cli, "run_triage", fake_run)
    monkeypatch.setattr(cli, "load_env", lambda *a, **k: None)
    assert cli.main(["123"]) == 0
    assert seen["issue_id"] == "123"
    assert json.loads(capsys.readouterr().out) == {"severity": "high", "issue_id": "123"}


def test_cli_missing_sentry_config_exits_2(monkeypatch, capsys, clean_env):
    monkeypatch.setattr(cli, "load_env", lambda *a, **k: None)
    assert cli.main(["123"]) == 2
    assert "SENTRY_TOKEN, SENTRY_ORG and SENTRY_PROJECT" in capsys.readouterr().err


def test_cli_triage_error_exits_1(monkeypatch, capsys, clean_env):
    def boom(issue_id, settings=None):
        raise TriageError("model gave up")

    monkeypatch.setattr(cli, "run_triage", boom)
    monkeypatch.setattr(cli, "load_env", lambda *a, **k: None)
    assert cli.main(["123"]) == 1
    assert "model gave up" in capsys.readouterr().err


def test_cli_requires_issue_id(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main([])
    assert e.value.code == 2
