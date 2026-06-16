"""
Output Filter — Zero Trust defense Layer 3.

Filters LLM responses for PII leakage and validates response schema
before returning to the client. Enterprise-tier control.

Two layers:
  Layer A — PII Scanner: redacts emails, phones, API keys, credit cards, IPs
  Layer B — Schema Validator: validates TriageResponse shape
"""

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PIIFilterResult:
    """Result of PII filtering on a response string."""
    filtered_text: str
    was_modified: bool = False
    redacted_fields: list[str] = field(default_factory=list)


class OutputFilter:
    """
    Filters responses for PII and validates output schema.

    PII patterns are redacted with '[REDACTED <type>]' markers.
    Static example domains (e.g. example.com) are allowlisted to
    avoid breaking legitimate support response templates.
    """

    # ─── PII Pattern Matchers ─────────────────────────────────────
    EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

    PHONE_RE = re.compile(
        r'(?:\b|\+)(?:\+\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b'
    )

    # Includes sk-proj- style keys (modern OpenAI format uses hyphens)
    API_KEY_RE = re.compile(
        r'\b(?:'
        r'sk-[A-Za-z0-9-]{20,}|'
        r'pk-[A-Za-z0-9]{20,}|'
        r'AKIA[A-Z0-9]{16}|'
        r'eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+'
        r')\b'
    )

    CC_RE = re.compile(r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b')

    IP_RE = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')

    # ─── Allowlist ────────────────────────────────────────────────
    DEFAULT_ALLOWED_DOMAINS: frozenset[str] = frozenset({
        "example.com",
        "support.example.com",
    })

    VALID_INTENTS: frozenset[str] = frozenset({
        "password_reset", "billing", "technical_support",
        "escalation", "unknown",
    })

    def __init__(self, allowed_domains: set[str] | None = None):
        self._allowed_domains = allowed_domains or set(self.DEFAULT_ALLOWED_DOMAINS)

    # ─── Public API ───────────────────────────────────────────────

    def filter_pii(self, text: str) -> PIIFilterResult:
        """Scan and redact PII from a response string, respecting allowlists."""
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

        # CC before PHONE — phone pattern can partially match 16-digit card numbers
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

        return PIIFilterResult(
            filtered_text=text,
            was_modified=len(redacted) > 0,
            redacted_fields=redacted,
        )

    def validate_classifier_output(self, output: Any) -> list[str]:
        """Validate classifier output schema. Returns list of issues (empty = valid)."""
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
        """Validate the full triage response before returning."""
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
