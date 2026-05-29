# AI Triage Agent

Customer support intent classification API. Week 1 used keyword matching; Week 2 replaced it with LLM reasoning via LiteLLM — swap providers with a single env var.

## Endpoints

- `GET /health` — verify the server is alive
- `POST /triage` — classify a customer message

**Response schema:**
```json
{
  "intent": "billing",
  "response": "For billing questions...",
  "confidence": 0.97,
  "needs_escalation": false
}
```

## Setup

**Step 1 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2 — Configure your API key**
```bash
cp .env.example .env
# Edit .env and add your key
```

Get a DeepSeek key at [platform.deepseek.com](https://platform.deepseek.com/) (~$0.14/1M tokens). Or use Claude Haiku — see `.env.example` for options.

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

# Empty message guard
curl -s -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"message": ""}'
# → {"detail":"message cannot be empty"}
```

## Run the Eval Harness

```bash
pytest tests/ -v -s
```

Runs 12 labeled test cases and prints an accuracy table per intent.

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
LLM_MODEL=deepseek/deepseek-chat       # DeepSeek V3 (default)
LLM_MODEL=claude-haiku-4-5-20251001    # Claude Haiku
```
