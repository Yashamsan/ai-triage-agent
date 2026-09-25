# load_dotenv MUST run before any langfuse import — SDK reads env vars at import time
from dotenv import load_dotenv

load_dotenv()

import threading
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langfuse import observe, propagate_attributes
from langgraph.types import Command
from pydantic import BaseModel

from app.agent_graph import triage_agent
from app.multi_agent_routes import router as multi_agent_router
from app.passport_api import router as passport_router
from app.prooflayer_api import router as prooflayer_router
from app.prooflayer_graph import record_decision
from app.rmf_api import router as rmf_router
from app.sdaia_api import router as sdaia_router
from app.security.guard_classifier import guard_classify
from app.security.input_sanitizer import InputSanitizer
from app.security.output_filter import OutputFilter
from app_ar.agent_graph import triage_agent_ar
from app_ar.security.guard_classifier import guard_classify_ar
from app_ar.security.input_sanitizer import InputSanitizer as ArInputSanitizer
from app_ar.security.output_filter import OutputFilter as ArOutputFilter
from audit import audit_record

app = FastAPI(title="AI Triage Agent")
app.include_router(prooflayer_router)
app.include_router(multi_agent_router)
app.include_router(sdaia_router)
app.include_router(passport_router)
app.include_router(rmf_router)

try:
    # Additive, demo-scoped durable-execution path (POST /triage/durable) —
    # requires temporalio + a running Temporal server. Never let its absence
    # take down the core app (password_reset/billing/technical_support/etc.
    # via the existing /triage endpoint above must keep working regardless).
    from app.temporal_routes import router as temporal_router
    app.include_router(temporal_router)
except Exception as exc:
    print(f"[startup] /triage/durable (Temporal) route skipped: {exc}")


@app.on_event("startup")
def _startup() -> None:
    from app.database import apply_schema, apply_schema_rmf, apply_schema_v2, apply_schema_v3
    try:
        apply_schema()
        print("[startup] schema (knowledge_base_chunks search_vector) applied")
    except Exception as exc:
        print(f"[startup] schema migration skipped: {exc}")
    try:
        apply_schema_v2()
    except Exception as exc:
        print(f"[startup] schema_v2 migration skipped: {exc}")
    try:
        apply_schema_v3()
        print("[startup] schema_v3 applied")
    except Exception as exc:
        print(f"[startup] schema_v3 migration skipped: {exc}")
    try:
        apply_schema_rmf()
        print("[startup] schema_rmf (P145) applied")
    except Exception as exc:
        print(f"[startup] schema_rmf migration skipped: {exc}")
    try:
        from app.embeddings import embed
        embed("warmup")
        print("[startup] embedding model pre-warmed")
    except Exception as exc:
        print(f"[startup] embedding warmup skipped: {exc}")
    # Orchestrator registration calls ProofLayer over HTTP (localhost:8000),
    # i.e. this same app — which isn't accepting connections yet during its
    # own startup event. Defer a few seconds so uvicorn has bound the port.
    def _delayed_orchestrator_init() -> None:
        try:
            from app.multi_agent_routes import init_orchestrator
            init_orchestrator()
            print("[startup] multi-agent orchestrator pre-warmed")
        except Exception as exc:
            print(f"[startup] orchestrator eager init skipped: {exc}")

    threading.Timer(3.0, _delayed_orchestrator_init).start()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)
sanitizer = InputSanitizer()
output_filter = OutputFilter()
ar_sanitizer = ArInputSanitizer()
ar_output_filter = ArOutputFilter()

_ARABIC_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)


def _is_arabic(text: str) -> bool:
    return any(lo <= ord(c) <= hi for c in text for lo, hi in _ARABIC_RANGES)


# Sticky per-session language routing. _is_arabic() only looks at the
# current message's characters, so a numeric/English follow-up (e.g. "4569"
# in reply to "give me the last 4 digits of your card") inside an Arabic
# conversation would otherwise get routed to _triage_en — a different
# LangGraph instance with a different thread_id, losing all prior context.
# Remember which language a session started in and keep routing there.
_session_language: dict[str, str] = {}


def _resolve_language(session_id: str | None, message: str) -> str:
    if session_id and session_id in _session_language:
        return _session_language[session_id]
    language = "ar" if _is_arabic(message) else "en"
    if session_id:
        _session_language[session_id] = language
    return language


class TriageRequest(BaseModel):
    message: str
    session_id: str | None = None


class TriageResponse(BaseModel):
    intent: str
    response: str
    confidence: float
    needs_escalation: bool
    interrupted: bool = False      # True when waiting for human escalation approval
    thread_id: str | None = None   # Use with /triage/resume when interrupted=True


class ResumeRequest(BaseModel):
    thread_id: str
    approved: bool


# Decorator execution order (bottom-up):
#   triage() → @observe wraps it → @audit_record wraps that → @app.post registers
@app.post("/triage", response_model=TriageResponse)
@audit_record(agent_id="ai-triage-agent", model_id="deepseek/deepseek-chat")
@observe(name="triage")
async def triage(request: TriageRequest, background_tasks: BackgroundTasks):
    if not request.message.strip():
        raise HTTPException(status_code=422, detail="message cannot be empty")

    if _resolve_language(request.session_id, request.message) == "ar":
        result, trace_steps = await _triage_ar(request)
        agent_name = "triage-agent-ar"
    else:
        result, trace_steps = await _triage_en(request)
        agent_name = "triage-agent-en"

    background_tasks.add_task(
        record_decision,
        decision_value=result.intent,
        confidence=result.confidence,
        input_query=request.message,
        session_id=result.thread_id or request.session_id or "",
        agent_name=agent_name,
        model_id="deepseek/deepseek-chat",
        reasoning_summary="pending-escalation" if result.interrupted else result.response[:500],
        human_override=False,
        trace_steps=trace_steps,
    )

    return result


async def _triage_en(request: TriageRequest) -> tuple[TriageResponse, list[dict]]:
    sanitized = sanitizer.sanitize(request.message)
    if sanitized.blocked:
        raise HTTPException(status_code=422, detail=f"Message rejected: {sanitized.block_reason}")

    safe_message = sanitized.sanitized_message

    guard = guard_classify(safe_message)
    if guard.is_injection and guard.confidence > 0.7:
        raise HTTPException(
            status_code=422,
            detail=f"Message rejected: suspected prompt injection (confidence={guard.confidence:.2f})",
        )

    thread_id = request.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    with propagate_attributes(session_id=thread_id, user_id=thread_id, trace_name="triage"):
        await triage_agent.ainvoke(
            {
                "message": safe_message,
                "session_id": thread_id,
                "intent": "",
                "confidence": 0.0,
                "needs_escalation": False,
                "needs_revision": False,
                "revised_intent": None,
                "revised_confidence": 0.0,
                "critique": None,
                "tool_output": "",
                "resolved": False,
                "context_history": "",
                "precedent_context": "",
                "response_text": "",
                "trace_steps": [],
            },
            config=config,
        )

    snapshot = triage_agent.get_state(config)
    if snapshot.next:
        partial = snapshot.values
        return TriageResponse(
            intent=partial.get("intent", "escalation"),
            response=(
                "⏸ **Escalation Pending Approval**\n\n"
                "A senior agent review is required before this ticket is created. "
                f"Use `POST /triage/resume` with `thread_id={thread_id!r}` to approve or decline."
            ),
            confidence=partial.get("confidence", 0.0),
            needs_escalation=True,
            interrupted=True,
            thread_id=thread_id,
        ), partial.get("trace_steps", [])

    final_state = snapshot.values
    trace_steps = final_state.get("trace_steps", [])
    pii_result = output_filter.filter_pii(final_state["response_text"])
    triage_response = TriageResponse(
        intent=final_state["intent"],
        response=pii_result.filtered_text,
        confidence=final_state["confidence"],
        needs_escalation=final_state["needs_escalation"],
    )
    if output_filter.validate_triage_response(triage_response):
        return TriageResponse(
            intent="unknown",
            response="We're experiencing a technical issue. Please try again later.",
            confidence=0.0,
            needs_escalation=False,
        ), trace_steps
    return triage_response, trace_steps


async def _triage_ar(request: TriageRequest) -> tuple[TriageResponse, list[dict]]:
    sanitized = ar_sanitizer.sanitize(request.message)
    if sanitized.blocked:
        raise HTTPException(status_code=422, detail=f"تم رفض الرسالة: {sanitized.block_reason}")

    safe_message = sanitized.sanitized_message

    guard = guard_classify_ar(safe_message)
    if guard.is_injection and guard.confidence > 0.7:
        raise HTTPException(
            status_code=422,
            detail=f"تم رفض الرسالة: اشتباه في حقن تعليمات (الثقة={guard.confidence:.2f})",
        )

    # Prefix thread_id with "ar_" so /triage/resume knows which agent to resume
    base_id = request.session_id or str(uuid.uuid4())
    thread_id = base_id if base_id.startswith("ar_") else f"ar_{base_id}"
    config = {"configurable": {"thread_id": thread_id}}

    with propagate_attributes(session_id=thread_id, user_id=thread_id, trace_name="triage_ar"):
        await triage_agent_ar.ainvoke(
            {
                "message": safe_message,
                "session_id": thread_id,
                "intent": "",
                "confidence": 0.0,
                "needs_escalation": False,
                "needs_revision": False,
                "revised_intent": None,
                "revised_confidence": 0.0,
                "critique": None,
                "tool_output": "",
                "resolved": False,
                "context_history": "",
                "precedent_context": "",
                "response_text": "",
                "trace_steps": [],
            },
            config=config,
        )

    snapshot = triage_agent_ar.get_state(config)
    if snapshot.next:
        partial = snapshot.values
        return TriageResponse(
            intent=partial.get("intent", "escalation"),
            response=(
                "⏸ **بانتظار موافقة المشرف**\n\n"
                "يتطلب هذا الطلب مراجعة من وكيل أول قبل إنشاء التذكرة. "
                f"استخدم `POST /triage/resume` مع `thread_id={thread_id!r}` للموافقة أو الرفض."
            ),
            confidence=partial.get("confidence", 0.0),
            needs_escalation=True,
            interrupted=True,
            thread_id=thread_id,
        ), partial.get("trace_steps", [])

    final_state = snapshot.values
    trace_steps = final_state.get("trace_steps", [])
    pii_result = ar_output_filter.filter_pii(final_state["response_text"])
    triage_response = TriageResponse(
        intent=final_state["intent"],
        response=pii_result.filtered_text,
        confidence=final_state["confidence"],
        needs_escalation=final_state["needs_escalation"],
    )
    if ar_output_filter.validate_triage_response(triage_response):
        return TriageResponse(
            intent="unknown",
            response="نواجه مشكلة تقنية حالياً. الرجاء المحاولة لاحقاً.",
            confidence=0.0,
            needs_escalation=False,
        ), trace_steps
    return triage_response, trace_steps


@app.post("/triage/resume", response_model=TriageResponse)
@observe(name="triage-resume")
async def triage_resume(request: ResumeRequest, background_tasks: BackgroundTasks):
    config = {"configurable": {"thread_id": request.thread_id}}

    if request.thread_id.startswith("ar_"):
        agent = triage_agent_ar
        pii = ar_output_filter
        agent_name = "triage-agent-ar"
    else:
        agent = triage_agent
        pii = output_filter
        agent_name = "triage-agent-en"

    snapshot = agent.get_state(config)
    if not snapshot or not snapshot.next:
        raise HTTPException(status_code=404, detail="No pending escalation found for this thread_id")

    final_state = await agent.ainvoke(Command(resume=request.approved), config=config)
    pii_result = pii.filter_pii(final_state["response_text"])

    # A human is the one deciding here (approve/decline the escalation) — this
    # is the actual human-in-the-loop intervention point, so record it as
    # such rather than letting it go unlogged like the rest of the pipeline.
    background_tasks.add_task(
        record_decision,
        decision_value=final_state["intent"],
        confidence=final_state["confidence"],
        input_query=final_state.get("message", ""),
        session_id=request.thread_id,
        agent_name=agent_name,
        model_id="deepseek/deepseek-chat",
        reasoning_summary=(
            f"Human {'approved' if request.approved else 'declined'} escalation. "
            + pii_result.filtered_text[:400]
        ),
        human_override=True,
        trace_steps=final_state.get("trace_steps", []),
    )

    return TriageResponse(
        intent=final_state["intent"],
        response=pii_result.filtered_text,
        confidence=final_state["confidence"],
        needs_escalation=final_state["needs_escalation"],
    )


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Static UI ─────────────────────────────────────────────────────────
_UI_DIR = Path(__file__).parent.parent / "ui"

@app.get("/")
def serve_ui():
    return FileResponse(_UI_DIR / "index.html")


@app.get("/admin")
def serve_admin():
    return FileResponse(_UI_DIR / "admin.html")


app.mount("/ui", StaticFiles(directory=str(_UI_DIR)), name="ui")
