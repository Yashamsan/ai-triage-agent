from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="AI Triage Agent")

INTENTS = {
    "password_reset": {
        "keywords": ["forgot", "password", "reset", "can't login", "cannot login", "locked out"],
        "response": "To reset your password, visit the login page and click 'Forgot Password'. You'll receive an email with a reset link.",
    },
    "billing": {
        "keywords": ["invoice", "charge", "bill", "payment", "refund", "subscription", "pricing"],
        "response": "For billing questions, please log in to your account and visit the Billing section, or contact billing@support.example.com.",
    },
    "technical_support": {
        "keywords": ["error", "bug", "crash", "not working", "broken", "issue", "problem", "fails"],
        "response": "Our technical support team is here to help. Please describe the issue in detail and include any error messages you see.",
    },
    "escalation": {
        "keywords": ["manager", "supervisor", "complaint", "unacceptable", "demand", "escalate", "speak to someone"],
        "response": "I understand your frustration. I'm connecting you with a senior support agent who can address your concern directly. Please hold on.",
    },
}

FALLBACK_RESPONSE = "I'm not sure how to help with that. Could you rephrase, or contact our support team at support@example.com?"


class TriageRequest(BaseModel):
    message: str


class TriageResponse(BaseModel):
    intent: str
    response: str


def classify(message: str) -> str:
    lower = message.lower()
    for intent, data in INTENTS.items():
        if any(kw in lower for kw in data["keywords"]):
            return intent
    return "unknown"


@app.post("/triage", response_model=TriageResponse)
def triage(request: TriageRequest):
    if not request.message.strip():
        raise HTTPException(status_code=422, detail="message cannot be empty")
    intent = classify(request.message)
    response = INTENTS[intent]["response"] if intent != "unknown" else FALLBACK_RESPONSE
    return TriageResponse(intent=intent, response=response)


@app.get("/health")
def health():
    return {"status": "ok"}
