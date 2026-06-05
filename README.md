# AI Triage Agent

Customer support intent classification API.
- **Week 1** — keyword-based classifier, FastAPI skeleton, GitHub push
- **Week 2** — replaced with LLM reasoning via LiteLLM + DeepSeek, Langfuse observability
- **Week 3** — promptfoo eval suite, LiteLLM proxy with cost tracking and model fallback
- **Week 4** — Zero Trust security stack: input sanitizer, spotlighting, guard classifier, output filter

## Security Stack (Week 4)

Every request passes through three defense layers before reaching the LLM and three before returning to the client:

```
POST /triage
  │
  ├─ Phase 1 — InputSanitizer        (app/security/input_sanitizer.py)
  │     • strips control characters
  │     • enforces 4096 char limit
  │     • blocks known injection patterns (regex)  → 422
  │     • flags suspicious encoding (base64/hex)
  │
  ├─ Phase 2a — Guard Classifier     (app/security/guard_classifier.py)
  │     • lightweight LLM call (max_tokens=50) screens for semantic injection
  │     • confidence > 0.7 → block with 422
  │     • confidence ≤ 0.7 → flag in Langfuse, allow through
  │     • fails open on error — never blocks legitimate traffic
  │
  ├─ Phase 2b — Spotlighting
  │     • user message wrapped in <untrusted_input> tags
  │     • system prompt instructs LLM to treat tags as data boundary
  │     • reduces injection success rate ~50% → ~2% (Microsoft research)
  │
  ├─ LLM classifier (DeepSeek via LiteLLM)
  │
  └─ Phase 3 — OutputFilter          (app/security/output_filter.py)
        • Layer A: PII redaction — email, phone, API keys, credit cards, IPv4
          (allowlists example.com/support.example.com support addresses)
        • Layer B: schema validation — intent enum, confidence range, bool types
        • Schema violations → safe fallback response + Langfuse log
```

Set `GUARD_MODEL` in `.env` to use a cheaper model for the guard classifier.

## Architecture

The app follows a multi-agent chatbot pattern:

- **Agent state** (per-prompt, ephemeral): tracked via `@observe()` spans on `classify()`. Each LLM call creates a Langfuse span capturing input, output, and latency.
- **Conversation memory** (per-session, persistent): tracked via Langfuse sessions. Pass `session_id` in the request body and all traces sharing that ID are grouped into one Langfuse session.
- **Chatbot orchestrator**: the FastAPI `POST /triage` endpoint routes prompts to the classifier agent, optionally attached to a session context.
- **LiteLLM proxy** (optional): sits between the app and the LLM providers — logs cost per request, routes to cheap model first, falls back to expensive model if needed.

## Endpoints

- `GET /health` — verify the server is alive
- `POST /triage` — classify a customer message

**Request:**
```json
{
  "message": "I want to speak to a manager",
  "session_id": "user-123"
}
```

**Response:**
```json
{
  "intent": "escalation",
  "response": "I understand your frustration...",
  "confidence": 0.95,
  "needs_escalation": true
}
```

## Setup

**Step 1 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2 — Configure your API keys**
```bash
cp .env.example .env
# Edit .env and fill in your keys
```

Get a DeepSeek key at [platform.deepseek.com](https://platform.deepseek.com/) (~$0.14/1M tokens).
Get Langfuse keys at [cloud.langfuse.com](https://cloud.langfuse.com/) (free tier).

**Step 3 — Start the server**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Step 4 — Test it**
```bash
# Health check
curl http://localhost:8000/health
# → {"status":"ok"}

# Classify a message
curl -s -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to speak to a manager"}'
# → {"intent":"escalation","response":"...","confidence":0.95,"needs_escalation":true}

# With session tracking
curl -s -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"message": "I need a refund", "session_id": "user-123"}'

# Empty message guard
curl -s -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"message": ""}'
# → {"detail":"message cannot be empty"}
```

## Run the Eval Harness

**pytest — unit accuracy (calls LLM directly)**
```bash
pytest tests/ -v -s
```
Runs 12 labeled test cases and prints a per-intent accuracy table.

**promptfoo — integration eval (calls the running API)**
```bash
# Start the server first, then:
cd tests
promptfoo eval -c promptfooconfig.yaml --no-cache

# View HTML report
promptfoo view
```
Hits the live `/triage` endpoint, checks each response's `intent` field, and reports pass/fail per test case.

## LiteLLM Proxy (optional)

Adds cost tracking, model routing, and automatic fallback between providers.

```
Agent → LiteLLM Proxy (port 4000) → cheap-classifier (DeepSeek $0.14/M)
                                   → expensive-fallback (Claude Haiku, if needed)
```

**Start the proxy:**
```bash
cd proxy
litellm --config config.yaml --port 4000
```

**Enable in `.env`:**
```
LITELLM_PROXY_URL=http://localhost:4000
LITELLM_MASTER_KEY=your_proxy_master_key_here
```

Or run with Docker:
```bash
cd proxy
docker build -t triage-proxy .
docker run -p 4000:4000 --env-file ../.env triage-proxy
```

## Intents

| Intent | Example trigger |
|---|---|
| `password_reset` | "I forgot my password", "my account is locked" |
| `billing` | "I've been double charged", "I need a refund" |
| `technical_support` | "the app keeps crashing", "I'm getting a 500 error" |
| `escalation` | "get me your manager", "I want to file a complaint" |
| `unknown` | anything unrecognized |

## Switching Models

Change `LLM_MODEL` in your `.env`:
```
LLM_MODEL=cheap-classifier        # via proxy (default)
LLM_MODEL=deepseek/deepseek-chat  # direct, no proxy
LLM_MODEL=claude-haiku-4-5-20251001  # Claude Haiku direct
```
