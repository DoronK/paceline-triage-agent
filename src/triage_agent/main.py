"""FastAPI service: POST /triage {issue_id} -> structured triage report."""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .agent import TriageError
from .config import Settings, load_env
from .llm.base import LLMClient
from .service import build_llm, run_triage

__all__ = ["app", "build_llm"]

load_env()

app = FastAPI(title="Paceline Triage Agent", version="0.1.0")


class TriageRequest(BaseModel):
    issue_id: str


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return Settings.from_env()


@lru_cache(maxsize=1)
def _llm() -> LLMClient:
    return build_llm(_settings())


@app.post("/triage")
def triage(req: TriageRequest) -> dict:
    try:
        return run_triage(req.issue_id, settings=_settings(), llm=_llm())
    except TriageError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
