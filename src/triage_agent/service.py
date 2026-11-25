"""Wiring shared by every entry point (API, CLI): settings -> clients -> report."""
from __future__ import annotations

from typing import Any

from .agent import TriageAgent
from .config import Settings
from .llm.anthropic_client import AnthropicClient
from .llm.base import LLMClient
from .llm.openai_compat_client import PRESET_BASE_URLS, OpenAICompatClient
from .sentry_client import SentryReadOnlyClient
from .sinks import build_sink, issue_url


def build_llm(s: Settings) -> LLMClient:
    """Pick the LLM client named by LLM_PROVIDER."""
    if s.llm_provider == "anthropic":
        return AnthropicClient(api_key=s.anthropic_api_key, model=s.anthropic_model)
    if s.llm_provider == "openai_compat" or s.llm_provider in PRESET_BASE_URLS:
        base_url = s.llm_base_url or PRESET_BASE_URLS.get(s.llm_provider)
        if not base_url:
            raise RuntimeError("LLM_BASE_URL is required for openai_compat")
        if not s.llm_model:
            raise RuntimeError(f"LLM_MODEL is required for {s.llm_provider}")
        return OpenAICompatClient(base_url=base_url, model=s.llm_model,
                                  api_key=s.llm_api_key)
    raise RuntimeError(f"unsupported provider: {s.llm_provider}")


def run_triage(issue_id: str, settings: Settings | None = None,
               llm: LLMClient | None = None) -> dict[str, Any]:
    """Triage one Sentry issue. Raises TriageError if the agent fails.

    When REPORT_SINK is set, the report is also delivered to that sink. A
    sink failure never fails the triage: it lands in the result's
    ``sink_error`` field instead.
    """
    s = settings or Settings.from_env()
    with SentryReadOnlyClient(token=s.sentry_token, org=s.sentry_org,
                              project=s.sentry_project,
                              base_url=s.sentry_base_url) as sentry:
        result = TriageAgent(llm=llm or build_llm(s), sentry=sentry).triage(issue_id)
    try:  # building the sink can fail too (missing secret, unknown name)
        sink = build_sink(s)
        if sink is not None:
            sink.deliver(result["report"], issue_id=issue_id,
                         issue_url=issue_url(s.sentry_base_url, s.sentry_org, issue_id))
            result["sink"] = type(sink).__name__
    except Exception as e:  # noqa: BLE001 - delivery must not lose the report
        result["sink_error"] = f"{type(e).__name__}: {e}"
    return result
