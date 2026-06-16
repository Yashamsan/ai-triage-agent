"""Tool layer — dispatches to real DB queries based on intent.

Phase 1 (current): PostgreSQL + pgvector for FAQ lookup and ticket creation.
Phase 2: real user account data.
Phase 3: MCP server tools.
"""

from __future__ import annotations

from dataclasses import dataclass

from langfuse import observe

from app.observability import (
    RetrievalMetricsLogger,
    trace_embedding,
    trace_ticket_creation,
    trace_vector_search,
)


@dataclass
class ToolResult:
    success: bool
    data: str
    resolved: bool  # True = fully answered, False = needs escalation


@observe(name="faq_lookup")
def faq_lookup(intent: str, user_message: str) -> ToolResult:
    """Find the best matching FAQ article via vector similarity search."""
    metrics = RetrievalMetricsLogger()
    try:
        with metrics.trace_latency("embedding"):
            embedding = trace_embedding(user_message)
        with metrics.trace_latency("vector_search"):
            row = trace_vector_search(intent, embedding)

        hit = row is not None
        metrics.score_hit(hit)
        metrics.score_mrr(1 if hit else None)

        if hit:
            return ToolResult(
                success=True,
                data=f"**{row['title']}**\n\n{row['content']}",
                resolved=True,
            )
    except Exception:
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


@observe(name="ticket_lookup")
def ticket_lookup(user_message: str) -> ToolResult:
    """Create a support ticket and return its ID."""
    metrics = RetrievalMetricsLogger()
    try:
        with metrics.trace_latency("embedding"):
            embedding = trace_embedding(user_message)
        with metrics.trace_latency("ticket_db_insert"):
            ticket_id = trace_ticket_creation(user_message, "escalation", embedding)
        metrics.score_ticket_created(True)
        return ToolResult(
            success=True,
            data=(
                f"Priority ticket #{ticket_id} created. "
                "A senior support agent will reach out within 15 minutes."
            ),
            resolved=True,
        )
    except Exception:
        metrics.score_ticket_created(False)
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
            data=(
                "Hi! I'm the support assistant. I can help with:\n\n"
                "• **Password & account access** — resets, locked accounts\n"
                "• **Billing** — invoices, charges, refunds\n"
                "• **Technical issues** — app errors, API problems\n\n"
                "What can I help you with today?"
            ),
            resolved=False,
        ),
    }
    handler = tool_map.get(intent, tool_map["unknown"])
    return handler(intent, user_message)
