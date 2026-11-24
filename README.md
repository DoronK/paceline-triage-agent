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

- `src/triage_agent/llm/` - swappable LLM layer: Anthropic (default) and
  one OpenAI-compatible client that covers Gemini, Groq, OpenRouter and
  Ollama; any provider implementing `LLMClient` can be dropped in
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

### Free LLM providers

Set `LLM_PROVIDER` to a preset and give it a model (and a key where the
provider needs one). `LLM_BASE_URL` overrides a preset's URL;
`LLM_PROVIDER=openai_compat` with `LLM_BASE_URL` reaches any other
OpenAI-compatible endpoint.

```bash
# Google Gemini: free key from https://aistudio.google.com (no billing on the
# project = free tier, rate-limited; free-tier prompts may be used by Google).
# Pick a current model id from what your key can see (gemini-2.5-flash was
# shut down Oct 2026 and now returns 404):
#   curl -s https://generativelanguage.googleapis.com/v1beta/openai/models \
#     -H "Authorization: Bearer $LLM_API_KEY"
export LLM_PROVIDER=gemini LLM_API_KEY=... LLM_MODEL=<model id>

# Ollama: fully local and free, no key (`ollama pull qwen3` first)
export LLM_PROVIDER=ollama LLM_MODEL=qwen3

# Groq / OpenRouter (free-tier or `:free` models)
export LLM_PROVIDER=groq LLM_API_KEY=... LLM_MODEL=...
export LLM_PROVIDER=openrouter LLM_API_KEY=... LLM_MODEL=...:free
```

Model names and free-tier limits change; check the provider's current
list. The Anthropic-only server-side refusal fallback does not apply to
these providers.

## Evals

```bash
python -m evals.run_evals --dataset evals/dataset/synthetic_sample.json
```
