import json
import os

import litellm
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langfuse import observe, propagate_attributes
from pydantic import BaseModel, ValidationError

from app.security.input_sanitizer import InputSanitizer, SanitizationResult

load_dotenv()

app = FastAPI(title="AI Triage Agent")
sanitizer = InputSanitizer()

SYSTEM_PROMPT = """You are a customer support triage agent. Classify the customer message into exactly one intent.

Intents:
- password_reset: login issues, forgotten password, account locked, can't sign in, credentials
- billing: payments, charges, invoices, refunds, subscriptions, pricing, double-charged
- technical_support: bugs, errors, crashes, features not working, slow performance
- escalation: wants manager or supervisor, filing a formal complaint, expressing strong anger
- unknown: does not fit any category above

Return ONLY valid JSON with these exact fields:
{
  "intent": "<one of the five intents above>",
  "confidence": <float between 0.0 and 1.0>,
  "needs_escalation": <true if message is urgent or emotionally charged, otherwise false>
}"""

INTENT_RESPONSES = {
    "password_reset": "To reset your password, visit the login page and click 'Forgot Password'. You'll receive an email with a reset link.",
    "billing": "For billing questions, please log in to your account and visit the Billing section, or contact billing@support.example.com.",
    "technical_support": "Our technical support team is here to help. Please describe the issue in detail and include any error messages you see.",
    "escalation": "I understand your frustration. I'm connecting you with a senior support agent who can address your concern directly. Please hold on.",
    "unknown": "I'm not sure how to help with that. Could you rephrase, or contact our support team at support@example.com?",
}

VALID_INTENTS = set(INTENT_RESPONSES.keys())


class TriageRequest(BaseModel):
    message: str
    session_id: str | None = None


class ClassifierOutput(BaseModel):
    intent: str
    confidence: float
    needs_escalation: bool


class TriageResponse(BaseModel):
    intent: str
    response: str
    confidence: float
    needs_escalation: bool


@observe()
def classify(message: str) -> ClassifierOutput:
    api_base = os.getenv("LITELLM_PROXY_URL", None)
    api_key = os.getenv("LITELLM_MASTER_KEY", None)
    # Use proxy model name when proxy is configured, direct model otherwise
    default_model = "cheap-classifier" if api_base else "deepseek/deepseek-chat"
    model = os.getenv("LLM_MODEL", default_model)
    llm_response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        temperature=0,
        **({"api_base": api_base} if api_base else {}),
        **({"api_key": api_key} if api_key else {}),
    )
    raw = llm_response.choices[0].message.content
    data = json.loads(raw)
    result = ClassifierOutput(**data)
    if result.intent not in VALID_INTENTS:
        result.intent = "unknown"
    return result


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

    # ── Route to Agent ───────────────────────────
    safe_message = sanitized.sanitized_message
    if request.session_id:
        with propagate_attributes(session_id=request.session_id):
            classification = classify(safe_message)
    else:
        classification = classify(safe_message)

    return TriageResponse(
        intent=classification.intent,
        response=INTENT_RESPONSES[classification.intent],
        confidence=classification.confidence,
        needs_escalation=classification.needs_escalation,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
