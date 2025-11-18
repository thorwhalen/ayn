"""MCP server implementation."""

from __future__ import annotations

from typing import Callable, List, Optional

from ..controllers.base import AgentController
from ..core.models import AgentMetadata
from .models import MCPTool, MCPResource, MCPPrompt


class MCPServer:
    """MCP Server implementation

    Exposes agent capabilities via Model Context Protocol.

    Example:
        >>> server = MCPServer(name="test-server")
        >>> tool = MCPTool(
        ...     name="greet",
        ...     description="Greet user",
        ...     input_schema={"type": "object", "properties": {}}
        ... )
        >>> server.add_tool(tool)
        >>> len(server.tools)
        1
    """

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        controller: Optional[AgentController] = None,
    ):
        self.name = name
        self.version = version
        self.controller = controller
        self.tools: List[MCPTool] = []
        self.resources: List[MCPResource] = []
        self.prompts: List[MCPPrompt] = []

    def add_tool(self, tool: MCPTool):
        """Add a tool to the server

        Example:
            >>> server = MCPServer("test")
            >>> tool = MCPTool("test", "test tool", {})
            >>> server.add_tool(tool)
            >>> server.tools[0].name
            'test'
        """
        self.tools.append(tool)

    def add_resource(self, resource: MCPResource):
        """Add a resource to the server"""
        self.resources.append(resource)

    def add_prompt(self, prompt: MCPPrompt):
        """Add a prompt to the server"""
        self.prompts.append(prompt)

    def handle_list_tools(self) -> dict:
        """Handle tools/list request"""
        return {"tools": [tool.to_dict() for tool in self.tools]}

    def handle_list_resources(self) -> dict:
        """Handle resources/list request"""
        return {"resources": [res.to_dict() for res in self.resources]}

    def handle_list_prompts(self) -> dict:
        """Handle prompts/list request"""
        return {"prompts": [prompt.to_dict() for prompt in self.prompts]}

    def handle_call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Handle tools/call request"""
        for tool in self.tools:
            if tool.name == tool_name:
                if tool.handler:
                    result = tool.handler(**arguments)
                    return {"content": [{"type": "text", "text": str(result)}]}
                elif self.controller:
                    result = self.controller.invoke(arguments)
                    return {"content": [{"type": "text", "text": str(result)}]}

        raise ValueError(f"Tool not found: {tool_name}")

    def to_json_rpc_handler(self) -> Callable:
        """Create a JSON-RPC handler for MCP protocol

        Returns:
            Handler function that processes JSON-RPC requests
        """

        def handler(request: dict) -> dict:
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")

            try:
                if method == "tools/list":
                    result = self.handle_list_tools()
                elif method == "resources/list":
                    result = self.handle_list_resources()
                elif method == "prompts/list":
                    result = self.handle_list_prompts()
                elif method == "tools/call":
                    result = self.handle_call_tool(
                        params.get("name"), params.get("arguments", {})
                    )
                else:
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {method}",
                        },
                    }

                return {"jsonrpc": "2.0", "id": request_id, "result": result}

            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32603, "message": str(e)},
                }

        return handler


def controller_to_mcp_server(
    controller: AgentController,
    metadata: AgentMetadata,
    *,
    tools: Optional[List[MCPTool]] = None,
    resources: Optional[List[MCPResource]] = None,
    prompts: Optional[List[MCPPrompt]] = None,
) -> MCPServer:
    """Convert an agent controller to MCP server

    Args:
        controller: Agent controller
        metadata: Agent metadata
        tools: Optional tools to expose
        resources: Optional resources to expose
        prompts: Optional prompts to expose

    Returns:
        MCP server instance

    Example:
        >>> from ayn.controllers import GenericController
        >>> from ayn.core import AgentMetadata, AgentFramework, ControllerConfig
        >>> meta = AgentMetadata(name="test", description="test", framework=AgentFramework.CUSTOM)
        >>> controller = GenericController(meta, ControllerConfig())
        >>> server = controller_to_mcp_server(controller, meta)
        >>> server.name
        'test'
    """
    server = MCPServer(name=metadata.name, version=metadata.version, controller=controller)

    # Add default invoke tool
    default_tool = MCPTool(
        name=f"{metadata.name}_invoke",
        description=metadata.description,
        input_schema={
            "type": "object",
            "properties": {"input_data": {"type": "object", "description": "Input data"}},
            "required": ["input_data"],
        },
        handler=lambda input_data: controller.invoke(input_data),
    )
    server.add_tool(default_tool)

    # Add custom tools
    if tools:
        for tool in tools:
            server.add_tool(tool)

    # Add resources
    if resources:
        for resource in resources:
            server.add_resource(resource)

    # Add prompts
    if prompts:
        for prompt in prompts:
            server.add_prompt(prompt)

    return server
