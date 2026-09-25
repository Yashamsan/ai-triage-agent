# AI Triage Agent

A bilingual (English/Arabic) customer-support triage agent built on LangGraph, with a full decision-governance layer (**ProofLayer**), a multi-agent orchestrator, and SDAIA AI-compliance/risk-management modules layered on top.

Every decision the agent makes — classification, reflection, tool calls, the final response — is recorded as a queryable, replayable graph, not just a log line.

## Architecture

```
POST /triage  (English)          POST /triage  (Arabic, auto-detected)
      │                                  │
      ├─ Zero Trust security stack ──────┤   (see "Security Stack" below)
      │                                  │
      └─ LangGraph pipeline (app/agent_graph.py, app_ar/agent_graph.py)
           │
           ├─ classifier      → LLM intent classification (DeepSeek via LiteLLM)
           ├─ reflect         → LLM-as-judge second opinion; can override the
           │                    classifier when it's confidently wrong
           ├─ tool_runner     → dispatches to a DB tool by (possibly revised) intent:
           │                      • faq_lookup()  — generic FAQ table, pgvector search
           │                      • kb_lookup()   — private company knowledge base
           │                        (product_inquiry), pgvector search, bge-m3 embeddings
           │                      • ticket_lookup() — creates an escalation ticket
           ├─ store_memory    → per-session conversation memory + long-term precedents
           └─ responder       → LLM synthesizes a natural, grounded reply from the
                                 retrieved content (never pastes it verbatim)
```

Every node's Thought → Action → Observation is written to **ProofLayer** (`audit/`, `app/prooflayer_graph.py`) as a decision graph (`pl_nodes`/`pl_edges` in Postgres), queryable via `/api/v1/decisions/{id}/trace`, `/replay`, and `/blame` — full governance, not just an audit log.

On top of the core triage agent:

- **Multi-agent orchestrator** (`orchestrator/`) — a router plus 7 domain specialists (billing, technical, sales, retention, loyalty, complaints, fraud detection), reachable via `POST /triage/multi`.
- **SDAIA compliance module** (`app/sdaia_api.py`, `prooflayer-sdaia/`) — risk classification, incident tracking, safety reports, and ethics labels for agents operating under Saudi AI governance requirements.
- **SDAIA-P145 RMF** (`app/rmf_api.py`, `app/rmf_core.py`) — the National AI Risk Management Framework's five-stage risk cycle (context → risk register → assessment → treatment → review).
- **Agent Passport** (`app/passport_api.py`) — a single per-agent credential card assembled from the behavioral-contract registry and (when present) SDAIA compliance records.
- **Durable execution** (`app/temporal_routes.py`) — optional Temporal-backed `/triage/durable` for workflows that must survive a crash mid-escalation.

## Repo layout

```
app/            English triage agent, ProofLayer core, SDAIA/RMF/Passport APIs, admin UI backend
app_ar/         Arabic triage agent (mirrors app/, Arabic-first prompts and security)
shared/         Code genuinely shared between app/ and app_ar/ — embeddings, precedent memory
orchestrator/   Multi-agent router + specialist agents
audit/          ProofLayer's append-only decision ledger (hash-chained JSONL)
prooflayer-sdaia/  Standalone SDAIA compliance scaffold (its own package, own tests)
config/         Orchestrator agent/routing config (contact_center.yaml)
knowledge_base/ Gitignored raw KB spreadsheet + scripts/ingest_knowledge_base.py output
scripts/        Ingestion, retrieval eval harness, demo seeders
tests/          pytest suite (132+ tests, `-m "not llm"` skips real-LLM-call tests)
ui/             Static admin dashboard (ProofLayer trace viewer, RMF tab, etc.)
docker/         Full Docker Compose stack (this app + self-hosted LangFuse + Temporal)
proxy/          Optional LiteLLM proxy for cost tracking / model fallback
```

## Quick start — Docker Compose (recommended)

Brings up the triage agent, its Postgres (pgvector), and a full self-hosted LangFuse observability stack in one command.

```bash
cd docker
cp ../.env.example ../.env    # fill in DEEPSEEK_API_KEY at minimum
docker compose up -d triage-agent
```

Open `http://localhost:8000/ui/admin.html` for the admin dashboard, or hit the API directly:

```bash
curl -s -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"message": "I forgot my password", "session_id": "test-1"}'

# Arabic is auto-detected from the message text — same endpoint:
curl -s -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"message": "نسيت كلمة المرور", "session_id": "test-ar-1"}'
```

The container bakes the embedding model into the image at build time and runs CPU-only by default (see "Embeddings & RAG" below for GPU).

**First build is slow** (CUDA-capable torch + the embedding model, several GB) — subsequent builds reuse Docker's layer cache unless `requirements.txt`, `Dockerfile`, or the embedding model changes.

## Embeddings & RAG

Semantic search (both the generic FAQ table and the private company knowledge base) runs on **BAAI/bge-m3** — a real multilingual embedding model, chosen specifically because Arabic queries need genuine cross-lingual understanding, not just a fast English model. Loaded once and shared across every consumer via `shared/embeddings.py`.

**GPU (optional):** `docker-compose.yml` reserves a GPU for the `triage-agent` service, and `shared/embeddings.py` auto-detects CUDA and uses fp16 automatically. It's **forced to CPU by default** (`EMBEDDING_DEVICE: "cpu"` in `docker-compose.yml`) because on a resource-constrained host, the CUDA runtime's memory footprint (~2GB host RAM + ~3.4GB VRAM for this model) can leave too little headroom for Docker Desktop's own networking to stay reliable. Delete that line to re-enable GPU once you've confirmed the host has headroom (`docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` should succeed first).

**Ingesting the knowledge base:**
```bash
pip install -r requirements-kb.txt   # pandas + openpyxl, not needed at runtime
# Drop your spreadsheet(s) in knowledge_base/raw/ (gitignored), then:
python scripts/ingest_knowledge_base.py
```
Safe to re-run — each file's previously-ingested chunks are deleted before re-inserting, so editing the spreadsheet and re-running keeps the DB in sync.

**Measuring retrieval quality** — don't change the embedding model, the retrieval threshold, or the ranking logic without running this first:
```bash
python scripts/eval_kb_retrieval.py
```
Runs a golden query/expected-answer set against the real, production `find_kb_chunk()` and reports accuracy plus a misses list. This is what caught the swap from `all-MiniLM-L6-v2` to `bge-m3` taking Arabic retrieval accuracy from 38% to 90% on this KB — and what caught a leftover keyword-overlap heuristic (tuned for the old, weaker model) actively hurting results once the embedding model improved.

## Security Stack

Every request passes through defense layers before reaching the LLM and after leaving it:

```
POST /triage
  │
  ├─ Phase 1 — InputSanitizer
  │     • strips control characters, enforces a length limit
  │     • blocks known injection patterns (regex) → 422
  │     • flags suspicious encoding (base64/hex)
  │
  ├─ Phase 2a — Guard Classifier
  │     • lightweight LLM call screens for semantic prompt injection
  │     • confidence > 0.7 → block with 422
  │     • fails open on error — never blocks legitimate traffic on an LLM hiccup
  │
  ├─ Phase 2b — Spotlighting
  │     • user message wrapped in <untrusted_input> tags
  │     • system prompt instructs the LLM to treat tags as a data boundary
  │
  ├─ LangGraph pipeline (classifier → reflect → tool_runner → responder)
  │
  └─ Phase 3 — OutputFilter
        • PII redaction — email, phone, API keys, credit cards, IPv4
        • schema validation — intent enum, confidence range, bool types
        • violations → safe fallback response, logged
```

## Endpoints

| Area | Method | Path | Purpose |
|---|---|---|---|
| Core | GET | `/health` | Liveness check |
| Core | POST | `/triage` | Classify + respond (language auto-detected) |
| Core | POST | `/triage/resume` | Approve/decline an escalation that's pending human review |
| Multi-agent | POST | `/triage/multi` | Route through the 7-specialist orchestrator instead of the single agent |
| Durable | POST | `/triage/durable` | Temporal-backed workflow, survives a crash mid-run |
| ProofLayer | GET | `/api/v1/decisions/{id}/trace` | Full Thought→Action→Observation trace for one decision |
| ProofLayer | GET | `/api/v1/decisions/{id}/replay` | Replay a past decision against current policy |
| ProofLayer | GET | `/api/v1/decisions/{id}/blame` | Which policy/precedent drove this decision |
| ProofLayer | GET | `/api/v1/compliance/iso-42001` | ISO 42001 compliance summary |
| ProofLayer | GET | `/api/v1/compliance/report.pdf` | Downloadable compliance report |
| SDAIA | GET | `/api/v1/sdaia/overview` | Fleet-wide SDAIA risk overview |
| SDAIA | GET | `/api/v1/sdaia/agents/{id}/deployment-check` | Is this agent cleared to deploy? |
| RMF | POST | `/api/v1/rmf/risks` | Register a risk in the P145 five-stage cycle |
| RMF | GET | `/api/v1/rmf/matrix` | Current risk matrix |
| Passport | GET | `/api/v1/passport/{agent_name}` | Per-agent credential card |

See each router file (`app/prooflayer_api.py`, `app/sdaia_api.py`, `app/rmf_api.py`, `app/multi_agent_routes.py`, `app/passport_api.py`) for the complete list — this table covers the ones you'll reach for first.

**`POST /triage` request:**
```json
{ "message": "I want to speak to a manager", "session_id": "user-123" }
```

**Response:**
```json
{
  "intent": "escalation",
  "response": "I understand your frustration...",
  "confidence": 0.95,
  "needs_escalation": true,
  "interrupted": false,
  "thread_id": null
}
```

## Manual (non-Docker) setup

```bash
# 1. Dependencies
pip install -r requirements.txt

# 2. Configure
cp .env.example .env   # fill in DEEPSEEK_API_KEY at minimum

# 3. PostgreSQL 17 + pgvector
sudo apt install postgresql-17 postgresql-17-pgvector
sudo service postgresql start
sudo -u postgres psql -c "CREATE DATABASE triage_agent;"
sudo -u postgres psql -d triage_agent -c "CREATE EXTENSION vector;"

python -m app.seed_data          # 6 demo FAQ articles + 6 tickets
python scripts/ingest_knowledge_base.py   # your real KB, if you have one

# 4. Run
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Testing

```bash
# Full suite, skipping tests that make real LLM calls
pytest -q -m "not llm"

# Include real-LLM-call tests too (costs money, needs DEEPSEEK_API_KEY)
pytest -q

# Retrieval-quality regression (see "Embeddings & RAG" above)
python scripts/eval_kb_retrieval.py

# Lint
ruff check .
```

`prooflayer-sdaia/` has its own test suite (`prooflayer-sdaia/conftest.py` deliberately excludes `P145/` from the same collection run — both it and `prooflayer-sdaia/app` are packages literally named `app`, and pytest can only resolve one `app` per process). Run it separately:
```bash
cd prooflayer-sdaia && pytest -q
```

## Self-hosted LangFuse (observability)

`docker compose up -d` brings up a full self-hosted LangFuse v3 stack (`langfuse-web`, `langfuse-worker`, its own Postgres, ClickHouse, Redis, MinIO) alongside the triage agent.

1. Fill in `docker/.env.langfuse` (gitignored — secrets never leave your machine):
   ```bash
   openssl rand -base64 32   # → NEXTAUTH_SECRET
   openssl rand -hex 32      # → LANGFUSE_ENCRYPTION_KEY
   # then replace every CHANGEME_* value in docker/.env.langfuse
   ```
2. `docker compose --env-file docker/.env.langfuse up -d`
3. Open `http://localhost:3000`, log in with `LANGFUSE_INIT_USER_*`, generate API keys under Project Settings.
4. Put `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL=http://localhost:3000` in the root `.env`.

On a memory-constrained host, this stack (plus the optional Temporal stack) is the first thing worth stopping if Docker Desktop's networking gets unreliable — none of it is in `/triage`'s actual request path:
```bash
docker stop docker-langfuse-web-1 docker-langfuse-worker-1 docker-clickhouse-1 \
            docker-redis-1 docker-minio-1 docker-temporal-1 docker-temporal-ui-1 \
            docker-temporal-postgresql-1
```

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `DEEPSEEK_API_KEY` | Yes | Classifier/reflection/response LLM calls |
| `DATABASE_URL` | Yes | Postgres connection (Docker Compose sets this for you) |
| `OPENROUTER_API_KEY` | No | Only for call sites explicitly using an `openrouter/` model |
| `LLM_MODEL` / `AR_LLM_MODEL` | No | Override the default model per language |
| `GUARD_MODEL` | No | Cheaper model for the injection guard classifier |
| `EMBEDDING_DEVICE` | No | Force `cpu` or `cuda` for embeddings; auto-detects if unset |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` | No | Observability tracing |
| `LITELLM_PROXY_URL` / `LITELLM_MASTER_KEY` | No | Route through the optional LiteLLM proxy instead of calling providers directly |

See `.env.example` for the full, current list with inline comments.

## Intents

| Intent | Example trigger |
|---|---|
| `password_reset` | "I forgot my password", "my account is locked" |
| `billing` | "I've been double charged", "I need a refund" |
| `technical_support` | "the app keeps crashing", "I'm getting a 500 error" |
| `product_inquiry` | "what packages do you offer", "how do I activate call forwarding" |
| `escalation` | "get me your manager", "I want to file a complaint" |
| `greeting` | "hi", "السلام عليكم" |
| `unknown` | anything the classifier and reflection both can't place |
