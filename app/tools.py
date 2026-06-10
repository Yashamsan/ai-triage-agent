"""Mock tools that the agent can call based on intent.

Each tool simulates what a real database or MCP server would return.
Phase 1: hardcoded data. Phase 2: real database. Phase 3: MCP server.
"""

from dataclasses import dataclass


@dataclass
class ToolResult:
    success: bool
    data: str
    resolved: bool  # True = fully answered, False = needs escalation


def faq_lookup(intent: str, user_message: str) -> ToolResult:
    """Simulate an FAQ database lookup."""
    faqs = {
        "password_reset": (
            "To reset your password:\n"
            "1. Go to https://example.com/forgot-password\n"
            "2. Enter your email address\n"
            "3. Check your inbox for a reset link (expires in 30 minutes)\n"
            "4. Click the link and set a new password\n\n"
            "If you don't receive the email within 5 minutes, check your spam folder "
            "or contact IT support."
        ),
        "billing": (
            "Your current billing info:\n"
            "• Plan: Premium Monthly ($29.99/mo)\n"
            "• Last payment: June 1, 2026 — Successful\n"
            "• Next payment: July 1, 2026\n"
            "• Invoice history: Available in Billing Settings\n\n"
            "To update payment method, visit Account → Billing → Payment Methods."
        ),
        "technical_support": (
            "Common troubleshooting steps:\n"
            "1. Clear your browser cache and cookies\n"
            "2. Try a different browser or incognito mode\n"
            "3. Restart your device\n"
            "4. Check our status page at status.example.com\n\n"
            "If the issue persists, our engineering team can investigate."
        ),
    }
    article = faqs.get(intent)
    if article:
        return ToolResult(success=True, data=article, resolved=True)
    return ToolResult(
        success=False,
        data="No matching FAQ article found.",
        resolved=False,
    )


def ticket_lookup(user_message: str) -> ToolResult:
    """Simulate looking up an existing support ticket."""
    return ToolResult(
        success=True,
        data="No existing tickets found matching your description. "
        "A new ticket will be created.",
        resolved=True,
    )


def run_tool(intent: str, user_message: str) -> ToolResult:
    """Dispatch to the right tool based on classified intent."""
    tool_map = {
        "password_reset": faq_lookup,
        "billing": faq_lookup,
        "technical_support": faq_lookup,
        "escalation": lambda i, m: ticket_lookup(m),
        "unknown": lambda i, m: ToolResult(
            success=False,
            data="I couldn't determine what you need help with.",
            resolved=False,
        ),
    }
    handler = tool_map.get(intent, tool_map["unknown"])
    return handler(intent, user_message)
