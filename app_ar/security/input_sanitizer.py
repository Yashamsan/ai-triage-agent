"""
Input Sanitizer — Zero Trust Layer 1.
Extended for Arabic: blocks RTL override characters and combining marks used in attacks.
"""

import re
from dataclasses import dataclass, field


@dataclass
class SanitizationResult:
    sanitized_message: str
    was_modified: bool = False
    risk_flags: list[str] = field(default_factory=list)
    blocked: bool = False
    block_reason: str | None = None


class InputSanitizer:
    """
    Sanitizes user input before it reaches the LLM classifier.
    Arabic-aware: strips RTL override (U+202E), Arabic-specific control chars.
    """

    MAX_MESSAGE_LENGTH = 4096

    CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    # U+202E = RIGHT-TO-LEFT OVERRIDE (can hide malicious text)
    # U+202D = LEFT-TO-RIGHT OVERRIDE
    # U+202B = RIGHT-TO-LEFT EMBEDDING
    # U+202A = LEFT-TO-RIGHT EMBEDDING
    # U+2066-2069 = DIRECTIONAL ISOLATES
    UNICODE_CONTROL_RE = re.compile(
        "[\u202A\u202B\u202C\u202D\u202E\u2066\u2067\u2068\u2069]"
    )

    INJECTION_PATTERNS: list[re.Pattern] = [
        re.compile(
            r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|directives)",
            re.I,
        ),
        re.compile(
            r"disregard\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts)",
            re.I,
        ),
        re.compile(r"forget\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts)", re.I),
        re.compile(r"system\s+(prompt|instruction|message)\s*[:=]", re.I),
        re.compile(r"<\s*system\s*>", re.I),
        re.compile(r"you\s+are\s+(now\s+)?an?\s+\w+", re.I),
        # Arabic injection patterns
        re.compile(r"تجاهل\s+(جميع\s+)?(التعليمات|الأوامر|الرسائل)\s+(السابقة|أعلاه)", re.I),
        re.compile(r"أنت\s+(الآن\s+)?(مساعد|مطور|نظام)", re.I),
        re.compile(r"تظاهر\s+بأنك", re.I),
    ]

    ENCODING_PATTERNS: list[re.Pattern] = [
        re.compile(r"^[A-Za-z0-9+/=]{40,}$"),
        re.compile(r"^[0-9a-fA-F]{40,}$"),
    ]

    def sanitize(self, message: str) -> SanitizationResult:
        if not message or not message.strip():
            return SanitizationResult(
                sanitized_message="",
                blocked=True,
                block_reason="empty_message",
            )

        flags: list[str] = []
        modified = False
        original = message

        cleaned = self.CONTROL_CHAR_RE.sub("", message)
        if cleaned != original:
            flags.append("control_chars_stripped")
            modified = True
            message = cleaned

        cleaned = self.UNICODE_CONTROL_RE.sub("", message)
        if cleaned != message:
            flags.append("unicode_directionals_stripped")
            modified = True
            message = cleaned

        if len(message) > self.MAX_MESSAGE_LENGTH:
            flags.append(f"truncated_from_{len(message)}")
            message = message[: self.MAX_MESSAGE_LENGTH]
            modified = True

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

        for pattern in self.ENCODING_PATTERNS:
            if pattern.match(message.strip()):
                flags.append("encoded_payload_detected")

        return SanitizationResult(
            sanitized_message=message,
            was_modified=modified,
            risk_flags=flags,
            blocked=False,
        )
