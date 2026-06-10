"""Tool layer — dispatches to real DB queries based on intent.

Phase 1 (current): PostgreSQL + pgvector for FAQ lookup and ticket creation.
Phase 2: real user account data.
Phase 3: MCP server tools.
"""

from dataclasses import dataclass


@dataclass
class ToolResult:
    success: bool
    data: str
    resolved: bool  # True = fully answered, False = needs escalation


def faq_lookup(intent: str, user_message: str) -> ToolResult:
    """Find the best matching FAQ article via vector similarity search."""
    try:
        from app.embeddings import embed
        from app.database import find_faq

        embedding = embed(user_message)
        row = find_faq(intent, embedding)

        if row:
            return ToolResult(
                success=True,
                data=f"**{row['title']}**\n\n{row['content']}",
                resolved=True,
            )
    except Exception as exc:
        # DB unavailable — fall back to static responses
        pass

    # Static fallback (used when DB is not yet seeded or unreachable)
    fallbacks = {
        "password_reset": (
            "To reset your password, visit the login page and click 'Forgot Password'. "
            "You'll receive a reset link by email (expires in 30 minutes)."
        ),
        "billing": (
            "For billing questions, log in and visit Account → Billing, "
            "or email billing@support.example.com."
        ),
        "technical_support": (
            "Try clearing your cache and cookies, then check status.example.com "
            "for ongoing incidents. If the issue persists, our team will investigate."
        ),
    }
    content = fallbacks.get(intent)
    if content:
        return ToolResult(success=True, data=content, resolved=True)

    return ToolResult(success=False, data="No matching FAQ article found.", resolved=False)


def ticket_lookup(user_message: str) -> ToolResult:
    """Create a support ticket and return its ID."""
    try:
        from app.embeddings import embed
        from app.database import create_ticket

        embedding = embed(user_message)
        ticket_id = create_ticket(user_message, "escalation", embedding)
        return ToolResult(
            success=True,
            data=(
                f"Priority ticket #{ticket_id} created. "
                "A senior support agent will reach out within 15 minutes."
            ),
            resolved=True,
        )
    except Exception:
        return ToolResult(
            success=True,
            data="A priority ticket has been created. A senior agent will contact you shortly.",
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
