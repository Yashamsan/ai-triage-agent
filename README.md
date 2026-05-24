# AI Triage Agent

Customer support intent classification API built with FastAPI.

## Endpoints

- `GET /health` — verify the server is alive
- `POST /triage` — classify a customer message

## How to Run

**Step 1 — Navigate to project folder**
```bash
cd ~/projects/ai-triage-agent
```

**Step 2 — Install dependencies (one time)**
```bash
pip install -r requirements.txt
```

**Step 3 — Start the server**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- `app.main:app` — file `app/main.py`, object `app`
- `--host 0.0.0.0` — accept connections from anywhere
- `--port 8000` — listen on port 8000
- `--reload` — auto-restart on code changes (skip in production)

You'll see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Step 4 — Test it (in a second terminal)**
```bash
# Health check
curl http://localhost:8000/health
# → {"status":"ok"}

# Classify a message
curl -s -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to speak to a manager"}'
# → {"intent":"escalation","response":"I understand your frustration..."}

# Empty message guard
curl -s -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"message": ""}'
# → {"detail":"message cannot be empty"}
```

## Intents

| Intent | Example trigger |
|---|---|
| `password_reset` | "I forgot my password" |
| `billing` | "I need a refund" |
| `technical_support` | "The app keeps crashing" |
| `escalation` | "I want to speak to a manager" |
| `unknown` | anything unrecognized |
