# AI Triage Agent

Customer support intent classification API built with FastAPI.

## Endpoints

- `GET /health` — health check
- `POST /triage` — classify customer message

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Example

```bash
curl -s -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to speak to a manager"}'
```

## Intents

| Intent | Example trigger |
|---|---|
| `password_reset` | "I forgot my password" |
| `billing` | "I need a refund" |
| `technical_support` | "The app keeps crashing" |
| `escalation` | "I want to speak to a manager" |
| `unknown` | anything unrecognized |
