# Paceline Triage Agent

An agentic crash/performance triage system over Sentry data, built as a
portfolio project: given a Sentry issue, an LLM agent uses read-only tools to
investigate it and produces a structured triage report (severity, suspected
component, regression assessment, recommended action, confidence).

Built for [Paceline Run](https://www.pacelinerun.com), an Android running app.
All committed data is synthetic or scrubbed. No production secrets, no write
access to anything.

## Status

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
```

Put config in a `.env` file in the repo root (git-ignored; vars already set
in your shell take precedence):

```bash
SENTRY_TOKEN=...        # read-only token (org:read, project:read, event:read)
SENTRY_ORG=...
SENTRY_PROJECT=...
ANTHROPIC_API_KEY=...   # default provider
```

### Run a triage

```bash
triage <sentry_issue_id>      # prints the report as JSON
```

Exit codes: `0` report printed, `1` triage failed, `2` configuration error.

Or run the HTTP service:

```bash
uvicorn triage_agent.main:app --reload
curl -s localhost:8000/triage -H 'content-type: application/json' \
  -d '{"issue_id":"<sentry_issue_id>"}'
```

### Free LLM providers

Set `LLM_PROVIDER` to a preset and give it a model (and a key where the
provider needs one). `LLM_BASE_URL` overrides a preset's URL;
`LLM_PROVIDER=openai_compat` with `LLM_BASE_URL` reaches any other
OpenAI-compatible endpoint.

```bash
# Google Gemini: free key from https://aistudio.google.com (no billing on the
# project = free tier, rate-limited; free-tier prompts may be used by Google).
# gemini-2.5-flash now returns 404 ("no longer available to new users").
# gemini-3.5-flash works with tool calls; list what your key can see with:
#   curl -s https://generativelanguage.googleapis.com/v1beta/openai/models \
#     -H "Authorization: Bearer $LLM_API_KEY"
export LLM_PROVIDER=gemini LLM_API_KEY=... LLM_MODEL=gemini-3.5-flash

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

The runner drives the real agent loop over a labelled dataset. Sentry is
replayed from each case's fixture data, so no Sentry credentials are needed;
the LLM is whichever provider `LLM_PROVIDER` configures.

```bash
# live run with the configured provider
python -m evals.run_evals --dataset evals/dataset/synthetic_sample.json

# + LLM-as-judge on summary/action, $ cost (per 1M tokens), free-tier throttle
python -m evals.run_evals --dataset evals/dataset/synthetic_sample.json \
  --judge --price-in 3 --price-out 15 --rpm 5

# offline pipeline check (what CI runs); scores are a fixed-answer floor
python -m evals.run_evals --dataset evals/dataset/synthetic_sample.json --llm stub
```

It reports accuracy per labelled field (`severity` and `is_regression`
exact; `suspected_component` matches when every word of the label appears
in the prediction), errors, average tokens, average cost when prices are
given, and the judge's 1-5 average. An errored case counts as wrong for
every field it labels. The full per-case report is written to
`evals/results/` (git-ignored).

`evals/dataset/synthetic_sample.json` holds 10 synthetic cases. Real golden
cases are labelled privately and never committed.

### Labelling real cases (owner, local only)

```bash
python -m evals.fetch_candidates --limit 25
```

This pulls live issues (plus latest event and releases) from the configured
Sentry project into `evals/dataset/golden_private.json` - git-ignored on
purpose, since it holds real issue data. Fill each case's `labels`
(`severity`, `suspected_component`, `is_regression` - any non-empty subset),
then point the runner at that file. `validate_dataset` refuses to score
cases whose labels are still empty.
