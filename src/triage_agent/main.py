"""FastAPI service: POST /triage {issue_id} -> structured triage report."""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .agent import TriageAgent, TriageError
from .config import Settings
from .llm.anthropic_client import AnthropicClient
from .sentry_client import SentryReadOnlyClient

app = FastAPI(title="Paceline Triage Agent", version="0.1.0")


class TriageRequest(BaseModel):
    issue_id: str


@lru_cache(maxsize=1)
def _settings() -> Settings:
    settings = Settings.from_env()
    if settings.llm_provider != "anthropic":
        raise RuntimeError(f"unsupported provider: {settings.llm_provider}")
    return settings


@lru_cache(maxsize=1)
def _llm() -> AnthropicClient:
    s = _settings()
    return AnthropicClient(api_key=s.anthropic_api_key, model=s.anthropic_model)


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
