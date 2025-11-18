"""Example of MCP (Model Context Protocol) integration."""

from ayn import (
    AgentMetadata,
    AgentFramework,
    GenericController,
    ControllerConfig,
    MCPTool,
    MCPResource,
    MCPPrompt,
    controller_to_mcp_server,
    generate_claude_mcp_config,
)


def main():
    print("AYN - MCP Integration Example")
    print("=" * 60)

    # 1. Create a custom agent
    metadata = AgentMetadata(
        name="research-assistant",
        description="Helps with research tasks",
        framework=AgentFramework.CUSTOM,
        tags=["research", "assistant"],
    )

    controller = GenericController(metadata, ControllerConfig())

    # 2. Define custom MCP tools
    print("\n1. Creating MCP tools...")

    def search_papers(query: str, limit: int = 10) -> str:
        """Search for academic papers"""
        return f"Found {limit} papers for query: {query}"

    def summarize_text(text: str) -> str:
        """Summarize text"""
        return f"Summary of: {text[:50]}..."

    tools = [
        MCPTool(
            name="search_papers",
            description="Search for academic papers",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results"},
                },
                "required": ["query"],
            },
            handler=search_papers,
        ),
        MCPTool(
            name="summarize",
            description="Summarize text",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to summarize"}
                },
                "required": ["text"],
            },
            handler=summarize_text,
        ),
    ]

    # 3. Define resources
    print("2. Creating MCP resources...")
    resources = [
        MCPResource(
            uri="file:///data/research_papers.csv",
            name="Research Papers Dataset",
            mimeType="text/csv",
            description="Collection of research papers",
        )
    ]

    # 4. Define prompts
    print("3. Creating MCP prompts...")
    prompts = [
        MCPPrompt(
            name="research_query",
            description="Generate a research query",
            arguments=[
                {
                    "name": "topic",
                    "description": "Research topic",
                    "required": True,
                }
            ],
        )
    ]

    # 5. Create MCP server
    print("\n4. Creating MCP server...")
    mcp_server = controller_to_mcp_server(
        controller, metadata, tools=tools, resources=resources, prompts=prompts
    )

    print(f"   Server name: {mcp_server.name}")
    print(f"   Tools: {len(mcp_server.tools)}")
    print(f"   Resources: {len(mcp_server.resources)}")
    print(f"   Prompts: {len(mcp_server.prompts)}")

    # 6. Test JSON-RPC handler
    print("\n5. Testing JSON-RPC handler...")
    handler = mcp_server.to_json_rpc_handler()

    # List tools
    response = handler({"jsonrpc": "2.0", "method": "tools/list", "id": 1})
    print(f"   Tools available: {len(response['result']['tools'])}")

    # Call a tool
    response = handler(
        {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "search_papers", "arguments": {"query": "AI agents"}},
            "id": 2,
        }
    )
    print(f"   Tool result: {response['result']['content'][0]['text']}")

    # 7. Generate Claude Desktop config
    print("\n6. Generating Claude Desktop config...")
    claude_config = generate_claude_mcp_config(mcp_server, "http://localhost:8000")

    import json

    print(json.dumps(claude_config, indent=2))

    print("\n✓ MCP integration example completed!")
    print("\nTo use with Claude Desktop:")
    print("1. Save the config to ~/.config/claude-desktop/config.json (Linux/Mac)")
    print("   or %APPDATA%\\Claude\\config.json (Windows)")
    print("2. Restart Claude Desktop")
    print("3. The tools will be available in your conversations")


if __name__ == "__main__":
    main()
