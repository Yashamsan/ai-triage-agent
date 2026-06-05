"""
Input Sanitizer — Zero Trust defense Layer 1.

Strips/rejects dangerous content before it reaches the LLM.
Foundation-tier control per Anthropic Zero Trust framework.
"""

import re
from dataclasses import dataclass, field


@dataclass
class SanitizationResult:
    """Result of sanitizing a user message."""
    sanitized_message: str
    was_modified: bool = False
    risk_flags: list[str] = field(default_factory=list)
    blocked: bool = False
    block_reason: str | None = None


class InputSanitizer:
    """
    Sanitizes user input before it reaches the LLM classifier.

    Guards:
    - Null bytes & control characters → strip
    - Max message length → enforce (4096 chars)
    - Suspicious encoding (Base64/hex) → flag
    - Known injection patterns → block
    - Empty/whitespace → block
    """

    MAX_MESSAGE_LENGTH = 4096

    # ─── Guard 1: Control Characters ────────────────────────────
    # Strip null bytes (\x00), backspace (\x08), escape (\x1b),
    # and other non-printable ASCII except \n \r \t
    CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

    # ─── Guard 2: Injection Patterns ────────────────────────────
    # Patterns that indicate attempts to override system instructions
    INJECTION_PATTERNS: list[re.Pattern] = [
        re.compile(
            r'ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|directives)',
            re.I,
        ),
        re.compile(
            r'disregard\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts)',
            re.I,
        ),
        re.compile(r'forget\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts)', re.I),
        re.compile(r'system\s+(prompt|instruction|message)\s*[:=]', re.I),
        re.compile(r'<\s*system\s*>', re.I),
        re.compile(r'you\s+are\s+(now\s+)?an?\s+\w+', re.I),
    ]

    # ─── Guard 3: Suspicious Encoding ────────────────────────────
    # Long Base64 or hex strings are unusual in support messages
    ENCODING_PATTERNS: list[re.Pattern] = [
        re.compile(r'^[A-Za-z0-9+/=]{40,}$'),   # Base64-like (40+ chars)
        re.compile(r'^[0-9a-fA-F]{40,}$'),       # Hex-like (40+ chars)
    ]

    def sanitize(self, message: str) -> SanitizationResult:
        """Run all sanitization guards on a message."""
        if not message or not message.strip():
            return SanitizationResult(
                sanitized_message="",
                blocked=True,
                block_reason="empty_message",
            )

        flags: list[str] = []
        modified = False
        original = message

        # ── Guard: Control characters ──
        cleaned = self.CONTROL_CHAR_RE.sub("", message)
        if cleaned != original:
            flags.append("control_chars_stripped")
            modified = True
            message = cleaned

        # ── Guard: Max length ──
        if len(message) > self.MAX_MESSAGE_LENGTH:
            flags.append(f"truncated_from_{len(message)}")
            message = message[:self.MAX_MESSAGE_LENGTH]
            modified = True

        # ── Guard: Injection patterns ──
        for pattern in self.INJECTION_PATTERNS:
            if pattern.search(message):
                flags.append("injection_pattern_detected")
                return SanitizationResult(
                    sanitized_message=message,
                    was_modified=modified,
                    risk_flags=flags,
                    blocked=True,
                    block_reason="injection_pattern_detected",
                )

        # ── Guard: Suspicious encoding ──
        for pattern in self.ENCODING_PATTERNS:
            if pattern.match(message.strip()):
                flags.append("encoded_payload_detected")
                # Flag but don't block — legitimate base64 support tickets exist
                # but still note it for monitoring

        return SanitizationResult(
            sanitized_message=message,
            was_modified=modified,
            risk_flags=flags,
            blocked=False,
        )
