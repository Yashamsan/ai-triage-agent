from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langfuse import observe, propagate_attributes
from pydantic import BaseModel, ValidationError

from app.classifier import classify, ClassifierOutput
from app.security.guard_classifier import guard_classify, GuardResult
from app.security.input_sanitizer import InputSanitizer, SanitizationResult
from app.security.output_filter import OutputFilter, PIIFilterResult

load_dotenv()

app = FastAPI(title="AI Triage Agent")
sanitizer = InputSanitizer()
output_filter = OutputFilter()

from app.agent_graph import triage_agent


class TriageRequest(BaseModel):
    message: str
    session_id: str | None = None


class TriageResponse(BaseModel):
    intent: str
    response: str
    confidence: float
    needs_escalation: bool


@observe()
@app.post("/triage", response_model=TriageResponse)
def triage(request: TriageRequest):
    # ── Input Validation ─────────────────────────
    if not request.message.strip():
        raise HTTPException(status_code=422, detail="message cannot be empty")

    # ── Phase 1: Input Sanitization (Zero Trust) ─
    sanitized = sanitizer.sanitize(request.message)
    if sanitized.blocked:
        raise HTTPException(
            status_code=422,
            detail=f"Message rejected: {sanitized.block_reason}",
        )
    if sanitized.risk_flags:
        try:
            from langfuse import get_current_trace
            trace = get_current_trace()
            if trace:
                trace.update(input=sanitized.sanitized_message)
        except Exception:
            pass

    safe_message = sanitized.sanitized_message

    # ── Phase 2: Guard Classifier (Enterprise) ───
    guard = guard_classify(safe_message)
    if guard.is_injection and guard.confidence > 0.7:
        raise HTTPException(
            status_code=422,
            detail=f"Message rejected: suspected prompt injection (confidence={guard.confidence:.2f})",
        )
    if guard.is_injection:
        try:
            from langfuse import get_current_trace
            trace = get_current_trace()
            if trace:
                trace.update(
                    metadata={
                        "guard_flag": True,
                        "guard_confidence": guard.confidence,
                        "guard_reason": guard.reason,
                    }
                )
        except Exception:
            pass

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

    if request.session_id:
        with propagate_attributes(session_id=request.session_id):
            final_state = triage_agent.invoke(initial_state)
    else:
        final_state = triage_agent.invoke(initial_state)

    # ── Phase 3: Output Filtering (Enterprise) ───
    response_text = final_state["response_text"]

    # Layer A: PII scan on response text
    pii_result = output_filter.filter_pii(response_text)

    # Layer B: Schema validation
    triage_response = TriageResponse(
        intent=final_state["intent"],
        response=pii_result.filtered_text,
        confidence=final_state["confidence"],
        needs_escalation=final_state["needs_escalation"],
    )
    schema_issues = output_filter.validate_triage_response(triage_response)

    if schema_issues:
        try:
            from langfuse import get_current_trace
            trace = get_current_trace()
            if trace:
                trace.update(metadata={"schema_issues": schema_issues})
        except Exception:
            pass
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
