"""Environment-based configuration. Secrets come from env vars only."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def load_env(path: str | Path = ".env") -> None:
    """Load a .env file if present. Vars already set in the shell win."""
    load_dotenv(path, override=False)


@dataclass(frozen=True)
class Settings:
    sentry_token: str
    sentry_org: str
    sentry_project: str
    llm_provider: str = "anthropic"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5-5"
    # OpenAI-compatible providers (gemini, groq, openrouter, ollama,
    # openai_compat): LLM_BASE_URL overrides the preset's URL.
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    sentry_base_url: str = "https://sentry.io/api/0"
    # Report sink: unset = report only in the response (default).
    report_sink: str | None = None
    slack_webhook_url: str | None = None
    github_token: str | None = None
    github_repo: str | None = None

    @staticmethod
    def from_env(require_sentry: bool = True) -> "Settings":
        """Read settings from env. require_sentry=False suits offline evals."""
        token = os.environ.get("SENTRY_TOKEN", "")
        org = os.environ.get("SENTRY_ORG", "")
        project = os.environ.get("SENTRY_PROJECT", "")
        if require_sentry and not (token and org and project):
            raise RuntimeError(
                "SENTRY_TOKEN, SENTRY_ORG and SENTRY_PROJECT env vars are required"
            )
        return Settings(
            sentry_token=token,
            sentry_org=org,
            sentry_project=project,
            llm_provider=os.environ.get("LLM_PROVIDER", "anthropic"),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            anthropic_model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
            llm_base_url=os.environ.get("LLM_BASE_URL"),
            llm_api_key=os.environ.get("LLM_API_KEY"),
            llm_model=os.environ.get("LLM_MODEL"),
            sentry_base_url=os.environ.get("SENTRY_BASE_URL", "https://sentry.io/api/0"),
            report_sink=os.environ.get("REPORT_SINK"),
            slack_webhook_url=os.environ.get("SLACK_WEBHOOK_URL"),
            github_token=os.environ.get("GITHUB_TOKEN"),
            github_repo=os.environ.get("GITHUB_REPO"),
        )
