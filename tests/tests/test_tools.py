"""Smoke tests for the read-only tool surface."""


def test_tools_are_read_only():
    from triage_agent.agent import _tools_schema
    names = {t["name"] for t in _tools_schema()}
    assert names == {"get_issue", "get_latest_event", "list_releases"}
    # By design there is no tool that writes, resolves or assigns anything.
    assert not any("resolve" in n or "assign" in n or "update" in n for n in names)
