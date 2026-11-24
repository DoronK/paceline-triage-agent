"""Eval harness: run the real agent loop on dataset cases and score it.

Sentry is replayed from each case's fixture data through the real
``SentryReadOnlyClient`` (httpx mock transport), so evals never touch live
Sentry. Every number this module reports is measured here; prices are only
applied when the caller passes them.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

from triage_agent.agent import TriageAgent
from triage_agent.config import Settings
from triage_agent.llm.base import LLMClient, LLMResult, ToolCall, ToolResult
from triage_agent.sentry_client import SentryReadOnlyClient

SEVERITIES = ("critical", "high", "medium", "low")
LABEL_KEYS = ("severity", "suspected_component", "is_regression")


# --- dataset ---------------------------------------------------------------

def validate_dataset(cases: list[dict[str, Any]]) -> None:
    """Raise ValueError naming the first malformed case."""
    for i, case in enumerate(cases):
        cid = case.get("id", f"#{i}")
        if not isinstance(case.get("issue"), dict) or not isinstance(case.get("labels"), dict):
            raise ValueError(f"{cid}: needs 'issue' and 'labels' objects")
        if not case["issue"].get("id"):
            raise ValueError(f"{cid}: issue.id is required")
        labels = case["labels"]
        unknown = set(labels) - set(LABEL_KEYS)
        if unknown or not labels:
            raise ValueError(f"{cid}: labels must be a non-empty subset of {LABEL_KEYS}")
        if "severity" in labels and labels["severity"] not in SEVERITIES:
            raise ValueError(f"{cid}: severity must be one of {SEVERITIES}")
        if "is_regression" in labels and not isinstance(labels["is_regression"], bool):
            raise ValueError(f"{cid}: is_regression must be a boolean")
        if "suspected_component" in labels and not str(labels["suspected_component"]).strip():
            raise ValueError(f"{cid}: suspected_component must be non-empty")


# --- replayed Sentry -------------------------------------------------------

def replay_sentry(case: dict[str, Any]) -> SentryReadOnlyClient:
    """A read-only client whose HTTP layer serves this case's fixture data."""
    issue = case["issue"]
    iid = str(issue["id"])

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path.endswith("/releases/"):
            return httpx.Response(200, json=case.get("releases", []))
        if path == f"/api/0/issues/{iid}/events/latest/":
            if "latest_event" in case:
                return httpx.Response(200, json=case["latest_event"])
        elif path == f"/api/0/issues/{iid}/":
            return httpx.Response(200, json=issue)
        return httpx.Response(404, json={"detail": "not found"})

    return SentryReadOnlyClient("eval", "eval-org", "eval-project",
                                transport=httpx.MockTransport(handler))


# --- scoring ---------------------------------------------------------------

_STOP = {"the", "a", "an", "of", "and", "in", "on", "for", "to"}
_SUFFIXES = ("ation", "ator", "ion", "ing", "ers", "er", "or", "s")


def _stem(word: str) -> str:
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def _tokens(text: str) -> set[str]:
    split_camel = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    words = re.split(r"[^a-z0-9]+", split_camel.lower())
    return {_stem(w) for w in words if w and w not in _STOP}


def component_matches(predicted: str, golden: str) -> bool:
    """True when every content word of the golden label appears in the prediction."""
    gold = _tokens(golden)
    return bool(gold) and gold <= _tokens(predicted or "")


def score_report(report: dict[str, Any], labels: dict[str, Any]) -> dict[str, bool]:
    """Per-field correctness, for the fields this case labels."""
    scores: dict[str, bool] = {}
    if "severity" in labels:
        scores["severity"] = str(report.get("severity", "")).strip().lower() == labels["severity"]
    if "suspected_component" in labels:
        scores["suspected_component"] = component_matches(
            str(report.get("suspected_component", "")), labels["suspected_component"])
    if "is_regression" in labels:
        scores["is_regression"] = report.get("is_regression") is labels["is_regression"]
    return scores


def cost_usd(usage: dict[str, int], price_in: float | None,
             price_out: float | None) -> float | None:
    """Dollar cost from token usage and per-million-token prices, if both given."""
    if price_in is None or price_out is None:
        return None
    return (usage["input_tokens"] * price_in + usage["output_tokens"] * price_out) / 1e6


# --- LLM-as-judge ----------------------------------------------------------

JUDGE_PROMPT = """You grade crash-triage reports for an Android app.
You get the Sentry data the triager could see, the reference labels, and the
triager's summary and recommended action. Score 1-5:
5 = accurate, grounded in the data, actionable next step
3 = partly right or vague
1 = wrong, invented details, or useless
Reply with ONLY a JSON object: {"score": <1-5>, "reason": "<one sentence>"}"""

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def judge_report(llm: LLMClient, case: dict[str, Any],
                 report: dict[str, Any]) -> dict[str, Any]:
    """Score the free-text parts of a report. Returns {score, reason, usage} or {error}."""
    payload = {
        "sentry_issue": case["issue"],
        "latest_event": case.get("latest_event"),
        "releases": case.get("releases"),
        "reference_labels": case["labels"],
        "triage_summary": report.get("summary"),
        "triage_recommended_action": report.get("recommended_action"),
    }
    try:
        result = llm.step(JUDGE_PROMPT, [{"role": "user",
                                          "content": json.dumps(payload, default=str)}], [])
        match = _JSON_OBJ.search(result.final_text or "")
        verdict = json.loads(match.group(0)) if match else {}
        score = verdict.get("score")
        if not isinstance(score, int) or not 1 <= score <= 5:
            return {"error": f"judge gave no valid score: {result.final_text!r}"}
        return {"score": score, "reason": str(verdict.get("reason", "")),
                "usage": {"input_tokens": result.input_tokens,
                          "output_tokens": result.output_tokens}}
    except Exception as e:  # noqa: BLE001 - a judge failure must not stop the run
        return {"error": f"{type(e).__name__}: {e}"}


# --- LLMs ------------------------------------------------------------------

class StubLLM:
    """Deterministic offline baseline: fetch the issue, then answer a fixed report.

    Exercises the full pipeline in CI without a network or API key. Its scores
    are a floor, not a measure of any real model.
    """

    REPORT = {"severity": "medium", "suspected_component": "unknown",
              "is_regression": False, "summary": "Stub baseline report.",
              "recommended_action": "Investigate manually.", "confidence": "low"}

    def step(self, system: str, messages: list[dict[str, Any]],
             tools: list[dict[str, Any]]) -> LLMResult:
        if len(messages) == 1:
            issue_id = messages[0]["content"].split()[-1].rstrip(".")
            return LLMResult("tool_use", {"role": "assistant", "content": "stub"},
                             tool_calls=[ToolCall("stub-1", "get_issue",
                                                  {"issue_id": issue_id})],
                             input_tokens=100, output_tokens=20)
        text = json.dumps(self.REPORT)
        return LLMResult("final", {"role": "assistant", "content": text},
                         final_text=text, input_tokens=200, output_tokens=60)

    def tool_results_messages(self, results: list[ToolResult]) -> list[dict[str, Any]]:
        return [{"role": "user", "content": r.content} for r in results]


class Throttled:
    """Spaces calls to the wrapped LLM so at most `rpm` happen per minute.

    Free tiers (e.g. Gemini: 5 requests/min) reject faster runs with HTTP 429.
    """

    def __init__(self, llm: LLMClient, rpm: float, clock=time.monotonic, sleep=time.sleep):
        self._llm, self._gap = llm, 60.0 / rpm
        self._clock, self._sleep = clock, sleep
        self._last: float | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)

    def step(self, system, messages, tools) -> LLMResult:
        if self._last is not None:
            wait = self._gap - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
        self._last = self._clock()
        return self._llm.step(system, messages, tools)

    def tool_results_messages(self, results: list[ToolResult]) -> list[dict[str, Any]]:
        return self._llm.tool_results_messages(results)


def live_llm() -> LLMClient:
    """The provider configured by LLM_PROVIDER & co. (no Sentry vars needed)."""
    from triage_agent.main import build_llm
    return build_llm(Settings.from_env(require_sentry=False))


# --- running ---------------------------------------------------------------

def run_case(case: dict[str, Any], llm: LLMClient,
             judge_llm: LLMClient | None = None) -> dict[str, Any]:
    """Triage one case. Agent failures are recorded, not raised."""
    out: dict[str, Any] = {"id": case.get("id"), "error": None, "report": None,
                           "scores": {}, "usage": None, "judge": None}
    try:
        with replay_sentry(case) as sentry:
            triaged = TriageAgent(llm=llm, sentry=sentry).triage(str(case["issue"]["id"]))
    except Exception as e:  # noqa: BLE001 - one bad case must not stop the run
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    out["report"], out["usage"] = triaged["report"], triaged["usage"]
    out["scores"] = score_report(triaged["report"], case["labels"])
    if judge_llm is not None:
        out["judge"] = judge_report(judge_llm, case, triaged["report"])
    return out


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def run_eval(cases: list[dict[str, Any]], llm: LLMClient,
             judge_llm: LLMClient | None = None, price_in: float | None = None,
             price_out: float | None = None) -> dict[str, Any]:
    """Run every case. Returns {"summary": ..., "results": [...]}.

    Accuracy per field counts an errored case as wrong for every field it labels.
    """
    results = [run_case(c, llm, judge_llm) for c in cases]
    accuracy: dict[str, float] = {}
    for key in LABEL_KEYS:
        labelled = [(c, r) for c, r in zip(cases, results) if key in c["labels"]]
        if labelled:
            correct = sum(1 for _, r in labelled if r["scores"].get(key))
            accuracy[key] = correct / len(labelled)

    ok = [r for r in results if r["error"] is None]
    avg_tokens = None
    if ok:
        avg_tokens = {k: _mean([r["usage"][k] for r in ok])
                      for k in ("input_tokens", "output_tokens")}
    costs = [c for r in ok if (c := cost_usd(r["usage"], price_in, price_out)) is not None]
    judged = [r["judge"]["score"] for r in ok if r["judge"] and "score" in r["judge"]]

    summary = {
        "cases": len(cases),
        "errors": len(results) - len(ok),
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "avg_cost_usd": _mean(costs),
        "judge_avg_score": _mean(judged),
        "judged_cases": len(judged),
    }
    return {"summary": summary, "results": results}
