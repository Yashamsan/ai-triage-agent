"""
Output Filter — Zero Trust Layer 3.
Extended for Saudi PII: National ID, Iqama, SA-IBAN, Saudi mobile.
"""

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PIIFilterResult:
    filtered_text: str
    was_modified: bool = False
    redacted_fields: list[str] = field(default_factory=list)


class OutputFilter:
    """Filters Arabic responses for PII — US + Saudi patterns."""

    # Universal PII
    EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    PHONE_RE = re.compile(
        r"(?:\b|\+)(?:\+\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"
    )
    API_KEY_RE = re.compile(
        r"\b(?:"
        r"sk-[A-Za-z0-9-]{20,}|"
        r"pk-[A-Za-z0-9]{20,}|"
        r"AKIA[A-Z0-9]{16}|"
        r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"
        r")\b"
    )
    CC_RE = re.compile(r"\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b")
    IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

    # Saudi-specific PII
    SAUDI_NATIONAL_ID_RE = re.compile(r"\b1\d{9}\b")    # Saudi National ID (starts with 1)
    IQAMA_RE = re.compile(r"\b2\d{9}\b")                 # Iqama number (starts with 2)
    SAUDI_MOBILE_RE = re.compile(r"\b(?:05|5)\d{8}\b")  # Saudi mobile: 05XXXXXXXX
    SAUDI_IBAN_RE = re.compile(r"\bSA\d{22}\b")          # Saudi IBAN: SA + 22 digits

    DEFAULT_ALLOWED_DOMAINS: frozenset[str] = frozenset({
        "example.com",
        "support.example.com",
    })

    VALID_INTENTS: frozenset[str] = frozenset({
        "greeting", "password_reset", "billing", "technical_support",
        "escalation", "unknown",
    })

    def __init__(self, allowed_domains: set[str] | None = None):
        self._allowed_domains = allowed_domains or set(self.DEFAULT_ALLOWED_DOMAINS)

    def filter_pii(self, text: str) -> PIIFilterResult:
        """Scan and redact PII (Arabic + English patterns)."""
        redacted: list[str] = []

        def _redact_email(m: re.Match) -> str:
            domain = m.group(0).split("@")[-1].lower()
            if domain in self._allowed_domains:
                return m.group(0)
            redacted.append("email")
            return "[REDACTED email]"

        text = self.EMAIL_RE.sub(_redact_email, text)

        new_text = self.API_KEY_RE.sub("[REDACTED api_key]", text)
        if new_text != text:
            redacted.append("api_key")
        text = new_text

        # CC before phone to avoid partial matches
        new_text = self.CC_RE.sub("[REDACTED credit_card]", text)
        if new_text != text:
            redacted.append("credit_card")
        text = new_text

        new_text = self.PHONE_RE.sub("[REDACTED phone]", text)
        if new_text != text:
            redacted.append("phone")
        text = new_text

        new_text = self.IP_RE.sub("[REDACTED ip]", text)
        if new_text != text:
            redacted.append("ip")
        text = new_text

        new_text = self.SAUDI_NATIONAL_ID_RE.sub("[REDACTED saudi_national_id]", text)
        if new_text != text:
            redacted.append("saudi_national_id")
        text = new_text

        new_text = self.IQAMA_RE.sub("[REDACTED iqama]", text)
        if new_text != text:
            redacted.append("iqama")
        text = new_text

        new_text = self.SAUDI_MOBILE_RE.sub("[REDACTED saudi_mobile]", text)
        if new_text != text:
            redacted.append("saudi_mobile")
        text = new_text

        new_text = self.SAUDI_IBAN_RE.sub("[REDACTED saudi_iban]", text)
        if new_text != text:
            redacted.append("saudi_iban")
        text = new_text

        return PIIFilterResult(
            filtered_text=text,
            was_modified=len(redacted) > 0,
            redacted_fields=redacted,
        )

    def validate_classifier_output(self, output: Any) -> list[str]:
        issues: list[str] = []
        if output.intent not in self.VALID_INTENTS:
            issues.append(f"invalid_intent: '{output.intent}'")
        if not isinstance(output.confidence, (int, float)):
            issues.append("confidence_not_numeric")
        elif not 0.0 <= output.confidence <= 1.0:
            issues.append(f"confidence_out_of_range: {output.confidence}")
        if not isinstance(output.needs_escalation, bool):
            issues.append("needs_escalation_not_bool")
        return issues

    def validate_triage_response(self, response: Any) -> list[str]:
        issues: list[str] = []
        if not response.response or not response.response.strip():
            issues.append("empty_response")

        class _AsClassifierOutput:
            def __init__(self, r: Any) -> None:
                self.intent = r.intent
                self.confidence = r.confidence
                self.needs_escalation = r.needs_escalation

        issues.extend(self.validate_classifier_output(_AsClassifierOutput(response)))
        return issues
