"""ayn - Agents You Need

A meta-framework for discovering, installing, and integrating reusable AI agents.
Acts as the 'pip + Docker Hub + Postman' for AI agents.

Core Features:
1. Search & Discovery: Find agents across GitHub, HuggingFace, awesome lists
2. Standard Controller Interface: Parametrizable agent controllers with action methods
3. HTTP/REST Export: FastAPI and LangServe wrappers
4. MCP Support: Model Context Protocol server integration
5. Custom Actions: ChatGPT and Claude action generation

Architecture:
- Search layer: Unified registry across multiple sources
- Controller protocol: Standard interface for invoking agents
- Export layer: HTTP/REST service generation
- MCP layer: Tools, resources, and prompts
- Actions: OpenAPI specs for ChatGPT/Claude

Example:
    >>> from ayn import AgentRegistry, BaseAgentController
    >>> registry = AgentRegistry()  # doctest: +SKIP
    >>> results = registry.search('data preparation agents')  # doctest: +SKIP
    >>> agent = BaseAgentController.from_metadata(results[0])  # doctest: +SKIP
    >>> result = agent.invoke(input_data)  # doctest: +SKIP
"""

from __future__ import annotations

__version__ = "0.0.1"

# Core models and config
from .core import (
    AgentFramework,
    AgentModality,
    AgentMetadata,
    ControllerConfig,
)

# Registry and search
from .registry import (
    AgentSearcher,
    GitHubAgentSearcher,
    HuggingFaceAgentSearcher,
    AwesomeListSearcher,
    AgentRegistry,
)

# Controllers
from .controllers import (
    AgentController,
    BaseAgentController,
    GenericController,
    CrewAIController,
    LangChainController,
    AutoGenController,
)

# Export
from .export import (
    export_as_fastapi,
    export_as_langserve,
)

# MCP
from .mcp import (
    MCPTool,
    MCPResource,
    MCPPrompt,
    MCPServer,
    controller_to_mcp_server,
)

# Actions
from .actions import (
    generate_chatgpt_action,
    generate_claude_mcp_config,
)

# Utilities
from .utils import (
    create_agent_from_registry,
    export_agent_full_stack,
)

__all__ = [
    # Version
    "__version__",
    # Data models
    "AgentFramework",
    "AgentModality",
    "AgentMetadata",
    "ControllerConfig",
    # Search
    "AgentSearcher",
    "GitHubAgentSearcher",
    "HuggingFaceAgentSearcher",
    "AwesomeListSearcher",
    "AgentRegistry",
    # Controller
    "AgentController",
    "BaseAgentController",
    "GenericController",
    "CrewAIController",
    "LangChainController",
    "AutoGenController",
    # Export
    "export_as_fastapi",
    "export_as_langserve",
    # MCP
    "MCPTool",
    "MCPResource",
    "MCPPrompt",
    "MCPServer",
    "controller_to_mcp_server",
    # Actions
    "generate_chatgpt_action",
    "generate_claude_mcp_config",
    # Convenience
    "create_agent_from_registry",
    "export_agent_full_stack",
]
