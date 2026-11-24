# Paceline Triage Agent

An agentic crash/performance triage system over Sentry data, built as a
portfolio project: given a Sentry issue, an LLM agent uses read-only tools to
investigate it and produces a structured triage report (severity, suspected
component, regression assessment, recommended action, confidence).

Built for [Paceline Run](https://www.pacelinerun.com), an Android running app.
All committed data is synthetic or scrubbed. No production secrets, no write
access to anything.

## Status

v1 under construction. Built with AI assistance (Claude) with human review of
every change. Metrics reported here come only from the eval harness in
`evals/` - nothing is invented.

## Architecture

- `src/triage_agent/llm/` - swappable LLM layer (Anthropic default; any
  provider implementing `LLMClient` can be dropped in)
- `src/triage_agent/sentry_client.py` - read-only Sentry REST API client
- `src/triage_agent/agent.py` - the tool-calling agent loop
- `src/triage_agent/main.py` - FastAPI service (`POST /triage`)
- `evals/` - golden dataset + runner (accuracy, LLM-as-judge, cost per triage)

## Setup

```bash
pip install -e ".[dev]"
export SENTRY_TOKEN=...        # read-only token (org:read, project:read, event:read)
export SENTRY_ORG=...
export SENTRY_PROJECT=...
export ANTHROPIC_API_KEY=...   # default provider
uvicorn triage_agent.main:app --reload
```

## Evals

```bash
python -m evals.run_evals --dataset evals/dataset/synthetic_sample.json
```
