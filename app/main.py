# load_dotenv MUST run before any langfuse import — SDK reads env vars at import time
from dotenv import load_dotenv

load_dotenv()

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langfuse import observe, propagate_attributes
from pydantic import BaseModel

from app.agent_graph import triage_agent
from app.security.guard_classifier import guard_classify
from app.security.input_sanitizer import InputSanitizer
from app.security.output_filter import OutputFilter

app = FastAPI(title="AI Triage Agent")
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


# @observe must be BELOW @app.post so FastAPI registers the observed function
@app.post("/triage", response_model=TriageResponse)
@observe(name="triage")
def triage(request: TriageRequest):
    if not request.message.strip():
        raise HTTPException(status_code=422, detail="message cannot be empty")

    # ── Phase 1: Input Sanitization ──────────────
    sanitized = sanitizer.sanitize(request.message)
    if sanitized.blocked:
        raise HTTPException(
            status_code=422,
            detail=f"Message rejected: {sanitized.block_reason}",
        )

    safe_message = sanitized.sanitized_message

    # ── Phase 2: Guard Classifier ─────────────────
    guard = guard_classify(safe_message)
    if guard.is_injection and guard.confidence > 0.7:
        raise HTTPException(
            status_code=422,
            detail=f"Message rejected: suspected prompt injection (confidence={guard.confidence:.2f})",
        )

    # ── Route to Agent ───────────────────────────
    initial_state = {
        "message": safe_message,
        "session_id": request.session_id,
        "intent": "",
        "confidence": 0.0,
        "needs_escalation": False,
        "tool_output": "",
        "resolved": False,
        "response_text": "",
    }

    with propagate_attributes(
        session_id=request.session_id or "",
        user_id=request.session_id or "",
        trace_name="triage",
    ):
        final_state = triage_agent.invoke(initial_state)

    # ── Phase 3: Output Filtering ─────────────────
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
            response="We're experiencing a technical issue. Please try again later.",
            confidence=0.0,
            needs_escalation=False,
        )

    return triage_response


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Static UI ─────────────────────────────────────────────────────────
_UI_DIR = Path(__file__).parent.parent / "ui"

@app.get("/")
def serve_ui():
    return FileResponse(_UI_DIR / "index.html")

app.mount("/ui", StaticFiles(directory=str(_UI_DIR)), name="ui")
