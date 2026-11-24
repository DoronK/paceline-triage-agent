"""FastAPI service: POST /triage {issue_id} -> structured triage report."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .agent import TriageAgent
from .config import Settings
from .llm.anthropic_client import AnthropicClient
from .sentry_client import SentryReadOnlyClient

app = FastAPI(title="Paceline Triage Agent", version="0.1.0")


class TriageRequest(BaseModel):
    issue_id: str


@app.post("/triage")
def triage(req: TriageRequest) -> dict:
    settings = Settings.from_env()
    if settings.llm_provider != "anthropic":
        raise RuntimeError(f"unsupported provider: {settings.llm_provider}")
    agent = TriageAgent(
        llm=AnthropicClient(api_key=settings.anthropic_api_key,
                            model=settings.anthropic_model),
        sentry=SentryReadOnlyClient(token=settings.sentry_token,
                                    org=settings.sentry_org,
                                    project=settings.sentry_project),
    )
    return agent.triage(req.issue_id)
