"""Tests for output filter — PII redaction + schema validation."""

import pytest
from app.security.output_filter import OutputFilter, PIIFilterResult
from app.main import ClassifierOutput, TriageResponse

filter = OutputFilter()


class TestPIIRedaction:
    """Verify PII patterns are detected and redacted."""

    def test_redacts_email(self):
        result = filter.filter_pii("Contact me at john@evil.com")
        assert result.was_modified
        assert "[REDACTED email]" in result.filtered_text
        assert "john@evil.com" not in result.filtered_text

    def test_allows_example_domain(self):
        result = filter.filter_pii("Email support@example.com for help")
        assert not result.was_modified
        assert "support@example.com" in result.filtered_text

    def test_allows_billing_example(self):
        result = filter.filter_pii("Contact billing@support.example.com")
        assert not result.was_modified

    def test_redacts_phone(self):
        result = filter.filter_pii("Call +966 55 123 4567")
        assert result.was_modified
        assert "[REDACTED phone]" in result.filtered_text

    def test_redacts_api_key(self):
        result = filter.filter_pii("Key: sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        assert result.was_modified
        assert "[REDACTED api_key]" in result.filtered_text

    def test_redacts_credit_card(self):
        result = filter.filter_pii("Card: 4111-1111-1111-1111")
        assert result.was_modified
        assert "[REDACTED credit_card]" in result.filtered_text

    def test_redacts_ip(self):
        result = filter.filter_pii("Server: 192.168.1.1")
        assert result.was_modified
        assert "[REDACTED ip]" in result.filtered_text

    def test_passes_clean_text(self):
        result = filter.filter_pii("I need help resetting my password")
        assert not result.was_modified
        assert result.filtered_text == "I need help resetting my password"

    def test_tracks_multiple_redactions(self):
        result = filter.filter_pii("Email me at hacker@bad.com or call 555-123-4567")
        assert len(result.redacted_fields) >= 2
        assert "email" in result.redacted_fields
        assert "phone" in result.redacted_fields


class TestSchemaValidation:
    """Verify classifier output and triage response validation."""

    def test_valid_classifier_passes(self):
        output = ClassifierOutput(
            intent="billing", confidence=0.95, needs_escalation=False
        )
        assert filter.validate_classifier_output(output) == []

    def test_invalid_intent_flagged(self):
        output = ClassifierOutput(
            intent="hack_the_planet", confidence=0.9, needs_escalation=False
        )
        issues = filter.validate_classifier_output(output)
        assert any("invalid_intent" in i for i in issues)

    def test_confidence_out_of_range(self):
        output = ClassifierOutput(
            intent="billing", confidence=1.5, needs_escalation=False
        )
        issues = filter.validate_classifier_output(output)
        assert any("confidence_out_of_range" in i for i in issues)

    def test_confidence_not_numeric(self):
        # model_construct bypasses Pydantic validation to test the filter logic
        output = ClassifierOutput.model_construct(
            intent="billing", confidence="high", needs_escalation=False
        )
        issues = filter.validate_classifier_output(output)
        assert any("confidence_not_numeric" in i for i in issues)

    def test_escalation_not_bool(self):
        output = ClassifierOutput.model_construct(
            intent="billing", confidence=0.9, needs_escalation="yes"
        )
        issues = filter.validate_classifier_output(output)
        assert any("needs_escalation_not_bool" in i for i in issues)

    def test_valid_triage_response_passes(self):
        response = TriageResponse(
            intent="password_reset",
            response="Click forgot password",
            confidence=0.95,
            needs_escalation=False,
        )
        assert filter.validate_triage_response(response) == []

    def test_empty_response_flagged(self):
        response = TriageResponse(
            intent="unknown",
            response="",
            confidence=0.0,
            needs_escalation=False,
        )
        issues = filter.validate_triage_response(response)
        assert any("empty_response" in i for i in issues)
