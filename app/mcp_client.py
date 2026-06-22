"""MCP client — connects to the server over stdio and calls tools."""

from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


@asynccontextmanager
async def mcp_tool_context():
    """Start the MCP server subprocess and yield an async call_tool function."""
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "app.mcp_server"],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            async def call_tool(name: str, arguments: dict[str, Any]) -> str:
                result = await session.call_tool(name, arguments)
                if result.content:
                    return result.content[0].text
                return ""

            yield call_tool
