"""FastAPI app — Arabic triage agent."""

from dotenv import load_dotenv

load_dotenv()

import uuid
from pathlib import Path  # noqa: F401

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse  # noqa: F401
from langfuse import observe, propagate_attributes
from langgraph.types import Command
from pydantic import BaseModel

from app_ar.agent_graph import triage_agent_ar
from app_ar.security.guard_classifier import guard_classify_ar
from app_ar.security.input_sanitizer import InputSanitizer
from app_ar.security.output_filter import OutputFilter

app = FastAPI(title="AI Triage Agent — Arabic")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)
sanitizer = InputSanitizer()
output_filter = OutputFilter()


class TriageRequest(BaseModel):
    message: str
    session_id: str | None = None


class TriageResponse(BaseModel):
    intent: str
    response: str
    confidence: float
    needs_escalation: bool
    interrupted: bool = False
    thread_id: str | None = None


class ResumeRequest(BaseModel):
    thread_id: str
    approved: bool


@app.post("/triage", response_model=TriageResponse)
@observe(name="triage_ar")
def triage(request: TriageRequest):
    if not request.message.strip():
        raise HTTPException(status_code=422, detail="الرسالة لا يمكن أن تكون فارغة")

    sanitized = sanitizer.sanitize(request.message)
    if sanitized.blocked:
        raise HTTPException(
            status_code=422,
            detail=f"تم رفض الرسالة: {sanitized.block_reason}",
        )

    safe_message = sanitized.sanitized_message

    guard = guard_classify_ar(safe_message)
    if guard.is_injection and guard.confidence > 0.7:
        raise HTTPException(
            status_code=422,
            detail=f"تم رفض الرسالة: اشتباه في حقن تعليمات (الثقة={guard.confidence:.2f})",
        )

    thread_id = request.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "message": safe_message,
        "session_id": thread_id,
        "intent": "",
        "confidence": 0.0,
        "needs_escalation": False,
        "tool_output": "",
        "resolved": False,
        "response_text": "",
    }

    with propagate_attributes(
        session_id=thread_id,
        user_id=thread_id,
        trace_name="triage_ar",
    ):
        final_state = triage_agent_ar.invoke(initial_state, config=config)

    snapshot = triage_agent_ar.get_state(config)
    if snapshot.next:
        partial = snapshot.values
        return TriageResponse(
            intent=partial.get("intent", "escalation"),
            response=(
                "⏸ **التصعيد في انتظار الموافقة**\n\n"
                "مراجعة وكيل أول مطلوبة قبل إنشاء هذه التذكرة. "
                f"استخدم `POST /triage/resume` مع `thread_id={thread_id!r}` للموافقة أو الرفض."
            ),
            confidence=partial.get("confidence", 0.0),
            needs_escalation=True,
            interrupted=True,
            thread_id=thread_id,
        )

    pii_result = output_filter.filter_pii(final_state["response_text"])

    triage_response = TriageResponse(
        intent=final_state["intent"],
        response=pii_result.filtered_text,
        confidence=final_state["confidence"],
        needs_escalation=final_state["needs_escalation"],
    )
    schema_issues = output_filter.validate_triage_response(triage_response)

    if schema_issues:
        return TriageResponse(
            intent="unknown",
            response="نواجه مشكلة تقنية حالياً. الرجاء المحاولة لاحقاً.",
            confidence=0.0,
            needs_escalation=False,
        )

    return triage_response


@app.post("/triage/resume", response_model=TriageResponse)
@observe(name="triage-resume-ar")
def triage_resume(request: ResumeRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    snapshot = triage_agent_ar.get_state(config)
    if not snapshot or not snapshot.next:
        raise HTTPException(status_code=404, detail="لا يوجد طلب تصعيد معلق لهذا thread_id")

    final_state = triage_agent_ar.invoke(Command(resume=request.approved), config=config)

    pii_result = output_filter.filter_pii(final_state["response_text"])
    return TriageResponse(
        intent=final_state["intent"],
        response=pii_result.filtered_text,
        confidence=final_state["confidence"],
        needs_escalation=final_state["needs_escalation"],
    )


@app.get("/health")
def health():
    return {"status": "ok"}
