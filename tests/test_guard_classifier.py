"""
Tests for guard classifier — unit tests focusing on prompt structure and error handling.

Note: Full integration tests require LLM API access and live in tests/integration/.
These unit tests validate the module structure, prompt format, and error paths.
"""

import pytest

from app.security.guard_classifier import (
    GUARD_SYSTEM_PROMPT,
    GuardResult,
    guard_classify,
)


class TestGuardPrompt:
    """Verify the guard prompt focuses on injection detection."""

    def test_prompt_mentions_injection(self):
        assert "injection" in GUARD_SYSTEM_PROMPT.lower()

    def test_prompt_requests_json_output(self):
        assert "JSON" in GUARD_SYSTEM_PROMPT or "json" in GUARD_SYSTEM_PROMPT


class TestGuardResult:
    """Verify GuardResult dataclass works correctly."""

    def test_clean_message(self):
        result = GuardResult(is_injection=False, confidence=0.05)
        assert not result.is_injection
        assert result.confidence == 0.05

    def test_injection_flagged(self):
        result = GuardResult(is_injection=True, confidence=0.92, reason="known injection pattern")
        assert result.is_injection
        assert result.reason == "known injection pattern"

    def test_default_reason_is_none(self):
        result = GuardResult(is_injection=False, confidence=0.0)
        assert result.reason is None


class TestGuardClassifyFallback:
    """Verify guard_classify handles errors gracefully (fail-open)."""

    @pytest.mark.llm
    def test_fails_open_on_error(self):
        """Passing an empty string should fail open, not crash the endpoint."""
        result = guard_classify("")
        assert isinstance(result, GuardResult)
