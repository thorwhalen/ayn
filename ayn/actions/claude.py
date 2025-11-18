"""Claude Desktop MCP configuration generation."""

from __future__ import annotations

from ..mcp.server import MCPServer


def generate_claude_mcp_config(
    mcp_server: MCPServer, server_url: str, transport: str = "sse"
) -> dict:
    """Generate Claude Desktop MCP configuration

    Args:
        mcp_server: MCP server instance
        server_url: URL where MCP server is hosted
        transport: Transport type (sse, stdio)

    Returns:
        Claude Desktop MCP config

    Example:
        >>> from ayn.mcp import MCPServer
        >>> server = MCPServer("test")
        >>> config = generate_claude_mcp_config(server, "http://localhost:8000")
        >>> config['mcpServers']['test']['url']
        'http://localhost:8000/sse'
    """
    if transport == "sse":
        return {
            "mcpServers": {
                mcp_server.name: {
                    "url": f"{server_url}/sse",
                    "transport": {"type": "sse"},
                }
            }
        }
    elif transport == "stdio":
        return {
            "mcpServers": {
                mcp_server.name: {
                    "command": "python",
                    "args": ["-m", "mcp_server"],
                    "transport": {"type": "stdio"},
                }
            }
        }
    else:
        raise ValueError(f"Unsupported transport: {transport}")
