"""MCP server exposing DB-backed tools over stdio transport."""

from mcp.server.fastmcp import FastMCP

from app.tools import ToolResult, faq_lookup, ticket_lookup

mcp = FastMCP("AI Triage Agent Tools", log_level="WARNING")


@mcp.tool()
def faq_search(intent: str, user_message: str) -> str:
    """Search FAQ articles for the given intent and return the best match."""
    result: ToolResult = faq_lookup(intent, user_message)
    return result.data


@mcp.tool()
def create_support_ticket(user_message: str) -> str:
    """Create a priority support ticket and return the ticket ID."""
    result: ToolResult = ticket_lookup(user_message)
    return result.data


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
