"""Convenience utility functions."""

from __future__ import annotations

from typing import Optional, List

from .registry import AgentRegistry
from .controllers.base import BaseAgentController
from .controllers.base import AgentController
from .core import ControllerConfig, AgentMetadata
from .mcp import MCPTool


def create_agent_from_registry(
    registry: AgentRegistry, query: str, config: Optional[ControllerConfig] = None
) -> BaseAgentController:
    """Search registry and create agent controller

    Args:
        registry: Agent registry
        query: Search query
        config: Optional controller config

    Returns:
        Agent controller

    Example:
        >>> from ayn import AgentRegistry
        >>> from ayn.core import AgentMetadata, AgentFramework
        >>> registry = AgentRegistry(searchers=[])
        >>> meta = AgentMetadata(name="test-agent", description="test", framework=AgentFramework.CUSTOM)
        >>> registry['test-agent'] = meta
        >>> controller = create_agent_from_registry(registry, 'test')
        >>> isinstance(controller, BaseAgentController)
        True
    """
    results = registry.search(query, limit=1)
    if not results:
        raise ValueError(f"No agents found for query: {query}")

    metadata = results[0]
    return BaseAgentController.from_metadata(metadata, config)


def export_agent_full_stack(
    controller: AgentController,
    metadata: AgentMetadata,
    *,
    api_host: str = "0.0.0.0",
    api_port: int = 8000,
    export_fastapi: bool = True,
    export_mcp: bool = True,
    export_chatgpt: bool = True,
    mcp_tools: Optional[List[MCPTool]] = None,
) -> dict:
    """Export agent with all integrations

    Args:
        controller: Agent controller
        metadata: Agent metadata
        api_host: API host
        api_port: API port
        export_fastapi: Whether to export as FastAPI
        export_mcp: Whether to export as MCP server
        export_chatgpt: Whether to generate ChatGPT action
        mcp_tools: Optional MCP tools

    Returns:
        Dict with export artifacts

    Example:
        >>> from ayn.controllers import GenericController
        >>> from ayn.core import AgentMetadata, AgentFramework, ControllerConfig
        >>> meta = AgentMetadata(name="test", description="test", framework=AgentFramework.CUSTOM)
        >>> controller = GenericController(meta, ControllerConfig())
        >>> artifacts = export_agent_full_stack(
        ...     controller, meta,
        ...     export_fastapi=False, export_mcp=False, export_chatgpt=True
        ... )
        >>> 'chatgpt_action' in artifacts
        True
    """
    from .export.fastapi_export import _create_fastapi_app
    from .mcp.server import controller_to_mcp_server
    from .actions import generate_chatgpt_action, generate_claude_mcp_config

    artifacts = {}
    api_url = f"http://{api_host}:{api_port}"

    if export_fastapi:
        app = _create_fastapi_app(controller, metadata)
        artifacts["fastapi_app"] = app
        artifacts["api_url"] = api_url

    if export_mcp:
        mcp_server = controller_to_mcp_server(
            controller, metadata, tools=mcp_tools
        )
        artifacts["mcp_server"] = mcp_server
        artifacts["claude_config"] = generate_claude_mcp_config(
            mcp_server, api_url
        )

    if export_chatgpt:
        artifacts["chatgpt_action"] = generate_chatgpt_action(
            controller, metadata, api_url
        )

    return artifacts
