# load_dotenv MUST run before any langfuse import — SDK reads env vars at import time
from dotenv import load_dotenv

load_dotenv()

import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langfuse import observe, propagate_attributes
from langgraph.types import Command
from pydantic import BaseModel

from app.agent_graph import triage_agent
from app.prooflayer_api import router as prooflayer_router
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
async def triage(request: TriageRequest):
    if not request.message.strip():
        raise HTTPException(status_code=422, detail="message cannot be empty")

    if _is_arabic(request.message):
        return await _triage_ar(request)
    return await _triage_en(request)


async def _triage_en(request: TriageRequest) -> TriageResponse:
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
        )

    final_state = snapshot.values
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
        )
    return triage_response


async def _triage_ar(request: TriageRequest) -> TriageResponse:
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
            },
            config=config,
        )

    snapshot = triage_agent_ar.get_state(config)
    final_state = snapshot.values
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
        )
    return triage_response


@app.post("/triage/resume", response_model=TriageResponse)
@observe(name="triage-resume")
async def triage_resume(request: ResumeRequest):
    config = {"configurable": {"thread_id": request.thread_id}}

    if request.thread_id.startswith("ar_"):
        agent = triage_agent_ar
        pii = ar_output_filter
    else:
        agent = triage_agent
        pii = output_filter

    snapshot = agent.get_state(config)
    if not snapshot or not snapshot.next:
        raise HTTPException(status_code=404, detail="No pending escalation found for this thread_id")

    final_state = await agent.ainvoke(Command(resume=request.approved), config=config)
    pii_result = pii.filter_pii(final_state["response_text"])
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
