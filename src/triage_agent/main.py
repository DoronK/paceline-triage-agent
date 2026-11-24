"""FastAPI service: POST /triage {issue_id} -> structured triage report."""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .agent import TriageAgent, TriageError
from .config import Settings
from .llm.anthropic_client import AnthropicClient
from .llm.base import LLMClient
from .llm.openai_compat_client import PRESET_BASE_URLS, OpenAICompatClient
from .sentry_client import SentryReadOnlyClient

app = FastAPI(title="Paceline Triage Agent", version="0.1.0")


class TriageRequest(BaseModel):
    issue_id: str


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return Settings.from_env()


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


@lru_cache(maxsize=1)
def _llm() -> LLMClient:
    return build_llm(_settings())


@app.post("/triage")
def triage(req: TriageRequest) -> dict:
    s = _settings()
    with SentryReadOnlyClient(token=s.sentry_token, org=s.sentry_org,
                              project=s.sentry_project,
                              base_url=s.sentry_base_url) as sentry:
        try:
            return TriageAgent(llm=_llm(), sentry=sentry).triage(req.issue_id)
        except TriageError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
