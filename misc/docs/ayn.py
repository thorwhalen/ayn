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
    >>> from ayn import AgentRegistry, AgentController
    >>> registry = AgentRegistry()  # doctest: +SKIP
    >>> results = registry.search('data preparation agents')  # doctest: +SKIP
    >>> agent = AgentController.from_registry(results[0])  # doctest: +SKIP
    >>> result = agent.invoke(input_data)  # doctest: +SKIP
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Union,
)
from collections.abc import MutableMapping
from functools import lru_cache


# ============================================================================
# Core Data Models
# ============================================================================


class AgentFramework(Enum):
    """Supported agent frameworks"""

    CREWAI = "crewai"
    LANGCHAIN = "langchain"
    LANGGRAPH = "langgraph"
    AUTOGEN = "autogen"
    SWARMS = "swarms"
    SMOLAGENTS = "smolagents"
    SUPERAGI = "superagi"
    CUSTOM = "custom"


class AgentModality(Enum):
    """Agent modalities"""

    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


@dataclass
class AgentMetadata:
    """Metadata for an agent

    Example:
        >>> meta = AgentMetadata(
        ...     name="data-prep-agent",
        ...     description="Prepares data for visualization",
        ...     framework=AgentFramework.CREWAI
        ... )
        >>> meta.name
        'data-prep-agent'
    """

    name: str
    description: str
    framework: AgentFramework
    source: str = ""  # GitHub URL, HF model ID, etc.
    tags: List[str] = field(default_factory=list)
    modality: AgentModality = AgentModality.TEXT
    version: str = "0.1.0"
    author: str = ""
    license: str = ""
    dependencies: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary

        Example:
            >>> meta = AgentMetadata(name="test", description="test agent", framework=AgentFramework.CREWAI)
            >>> d = meta.to_dict()
            >>> d['name']
            'test'
        """
        result = asdict(self)
        result["framework"] = self.framework.value
        result["modality"] = self.modality.value
        return result

    @classmethod
    def from_dict(cls, data: dict) -> AgentMetadata:
        """Create from dictionary"""
        data = data.copy()
        if isinstance(data.get("framework"), str):
            data["framework"] = AgentFramework(data["framework"])
        if isinstance(data.get("modality"), str):
            data["modality"] = AgentModality(data["modality"])
        return cls(**data)


# ============================================================================
# Search Interface
# ============================================================================


class AgentSearcher(Protocol):
    """Protocol for agent search implementations"""

    def search(
        self, query: str, *, limit: int = 20, **filters
    ) -> Iterable[AgentMetadata]:
        """Search for agents

        Args:
            query: Search query string
            limit: Maximum results to return
            **filters: Additional filters (framework, tags, etc.)

        Returns:
            Iterable of agent metadata
        """
        ...


class GitHubAgentSearcher:
    """Search for agents in GitHub repositories

    Example:
        >>> searcher = GitHubAgentSearcher()  # doctest: +SKIP
        >>> results = searcher.search('crewai agent')  # doctest: +SKIP
    """

    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")

    def search(
        self, query: str, *, limit: int = 20, **filters
    ) -> Iterable[AgentMetadata]:
        """Search GitHub for agent repos"""
        # Simplified implementation - real version would use GitHub API
        # TODO: Implement actual GitHub API search
        framework_filter = filters.get("framework")
        search_terms = f"{query} agent"
        if framework_filter:
            search_terms += f" {framework_filter}"

        # Placeholder - would normally make API calls
        return []


class HuggingFaceAgentSearcher:
    """Search for agents on Hugging Face

    Uses the Hugging Face MCP tool if available.

    Example:
        >>> searcher = HuggingFaceAgentSearcher()  # doctest: +SKIP
        >>> results = searcher.search('code generation')  # doctest: +SKIP
    """

    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token or os.getenv("HF_TOKEN")

    def search(
        self, query: str, *, limit: int = 20, **filters
    ) -> Iterable[AgentMetadata]:
        """Search Hugging Face for agents"""
        try:
            # Try to use HuggingFace MCP tools if available
            # This would integrate with the HF tools we have access to
            pass
        except Exception:
            pass

        return []


class AwesomeListSearcher:
    """Parse and search awesome lists for agents

    Example:
        >>> searcher = AwesomeListSearcher()  # doctest: +SKIP
        >>> results = searcher.search('autogen')  # doctest: +SKIP
    """

    AWESOME_LISTS = [
        "https://github.com/e2b-dev/awesome-ai-agents",
        "https://github.com/kyrolabs/awesome-agentic-ai",
        "https://github.com/Exponential-ML/awesome-open-source-ai-agents",
    ]

    def search(
        self, query: str, *, limit: int = 20, **filters
    ) -> Iterable[AgentMetadata]:
        """Search awesome lists"""
        # TODO: Implement markdown parsing and search
        # Would use graze to download and cache the markdown files
        return []


class AgentRegistry(MutableMapping[str, AgentMetadata]):
    """Unified registry for searching and managing agents

    Aggregates results from multiple sources and provides a dict-like interface.

    Example:
        >>> registry = AgentRegistry()
        >>> # Add a local agent
        >>> meta = AgentMetadata(name="test", description="test", framework=AgentFramework.CUSTOM)
        >>> registry['test'] = meta
        >>> registry['test'].name
        'test'
        >>> 'test' in registry
        True
    """

    def __init__(
        self,
        searchers: Optional[List[AgentSearcher]] = None,
        cache_dir: Optional[str] = None,
    ):
        self.searchers = searchers or [
            GitHubAgentSearcher(),
            HuggingFaceAgentSearcher(),
            AwesomeListSearcher(),
        ]
        self.cache_dir = cache_dir or os.path.expanduser("~/.ayn/registry")
        self._local_store: Dict[str, AgentMetadata] = {}
        self._load_cache()

    def _load_cache(self):
        """Load cached registry from disk"""
        if os.path.exists(self.cache_dir):
            cache_file = os.path.join(self.cache_dir, "registry.json")
            if os.path.exists(cache_file):
                with open(cache_file) as f:
                    data = json.load(f)
                    for key, meta_dict in data.items():
                        self._local_store[key] = AgentMetadata.from_dict(meta_dict)

    def _save_cache(self):
        """Save registry cache to disk"""
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, "registry.json")
        data = {k: v.to_dict() for k, v in self._local_store.items()}
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        source: Optional[str] = None,
        **filters,
    ) -> List[AgentMetadata]:
        """Search across all sources

        Args:
            query: Search query
            limit: Max results
            source: Specific source to search (github, huggingface, awesome)
            **filters: Additional filters

        Returns:
            List of agent metadata

        Example:
            >>> registry = AgentRegistry(searchers=[])
            >>> # Add some test data
            >>> registry['agent1'] = AgentMetadata(
            ...     name="agent1", description="test agent",
            ...     framework=AgentFramework.CREWAI, tags=['data']
            ... )
            >>> results = registry.search('agent')
            >>> len(results) >= 1
            True
        """
        results = []

        # Search local store first
        for name, meta in self._local_store.items():
            if (
                query.lower() in name.lower()
                or query.lower() in meta.description.lower()
                or any(query.lower() in tag.lower() for tag in meta.tags)
            ):
                results.append(meta)

        # Search remote sources if not enough results
        if len(results) < limit:
            for searcher in self.searchers:
                if source and source not in searcher.__class__.__name__.lower():
                    continue
                try:
                    remote_results = list(
                        searcher.search(query, limit=limit - len(results), **filters)
                    )
                    results.extend(remote_results)
                    if len(results) >= limit:
                        break
                except Exception as e:
                    # Continue on error
                    continue

        return results[:limit]

    # MutableMapping interface
    def __getitem__(self, key: str) -> AgentMetadata:
        return self._local_store[key]

    def __setitem__(self, key: str, value: AgentMetadata):
        self._local_store[key] = value
        self._save_cache()

    def __delitem__(self, key: str):
        del self._local_store[key]
        self._save_cache()

    def __iter__(self):
        return iter(self._local_store)

    def __len__(self):
        return len(self._local_store)


# ============================================================================
# Controller Interface
# ============================================================================


class AgentController(Protocol):
    """Standard protocol for agent controllers

    Controllers provide a uniform interface for invoking agents regardless
    of their underlying framework.

    Example:
        >>> class MyController:
        ...     def __init__(self, config):
        ...         self.config = config
        ...     def invoke(self, input_data, **kwargs):
        ...         return f"Processed: {input_data}"
        >>> controller = MyController({'model': 'gpt-4'})
        >>> controller.invoke('test')
        'Processed: test'
    """

    def invoke(self, input_data: Any, **kwargs) -> Any:
        """Invoke the agent synchronously

        Args:
            input_data: Input data for the agent
            **kwargs: Additional parameters

        Returns:
            Agent output
        """
        ...

    async def ainvoke(self, input_data: Any, **kwargs) -> Any:
        """Invoke the agent asynchronously"""
        ...

    def stream(self, input_data: Any, **kwargs) -> Iterable[Any]:
        """Stream agent outputs"""
        ...


@dataclass
class ControllerConfig:
    """Configuration for agent controllers

    Example:
        >>> config = ControllerConfig(
        ...     model="gpt-4",
        ...     temperature=0.7,
        ...     max_tokens=1000
        ... )
        >>> config.model
        'gpt-4'
    """

    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 60
    retry_count: int = 3
    api_key: Optional[str] = None
    additional_params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


class BaseAgentController(ABC):
    """Base class for agent controllers

    Provides common functionality and enforces the controller protocol.

    Example:
        >>> class SimpleController(BaseAgentController):
        ...     def invoke(self, input_data, **kwargs):
        ...         return f"Result: {input_data}"
        >>> controller = SimpleController(ControllerConfig())
        >>> controller.invoke('test')
        'Result: test'
    """

    def __init__(self, config: ControllerConfig):
        self.config = config

    @abstractmethod
    def invoke(self, input_data: Any, **kwargs) -> Any:
        """Invoke the agent - must be implemented by subclasses"""
        pass

    async def ainvoke(self, input_data: Any, **kwargs) -> Any:
        """Default async implementation wraps synchronous invoke"""
        return self.invoke(input_data, **kwargs)

    def stream(self, input_data: Any, **kwargs) -> Iterable[Any]:
        """Default streaming yields single result"""
        yield self.invoke(input_data, **kwargs)

    @classmethod
    def from_metadata(
        cls, metadata: AgentMetadata, config: Optional[ControllerConfig] = None
    ) -> BaseAgentController:
        """Factory method to create controller from metadata"""
        config = config or ControllerConfig()
        # Route to appropriate framework-specific controller
        if metadata.framework == AgentFramework.CREWAI:
            return CrewAIController(metadata, config)
        elif metadata.framework == AgentFramework.LANGCHAIN:
            return LangChainController(metadata, config)
        elif metadata.framework == AgentFramework.AUTOGEN:
            return AutoGenController(metadata, config)
        else:
            return GenericController(metadata, config)


class GenericController(BaseAgentController):
    """Generic controller for custom agents

    Example:
        >>> meta = AgentMetadata(name="test", description="test", framework=AgentFramework.CUSTOM)
        >>> controller = GenericController(meta, ControllerConfig())
        >>> result = controller.invoke({'input': 'test'})
    """

    def __init__(self, metadata: AgentMetadata, config: ControllerConfig):
        super().__init__(config)
        self.metadata = metadata

    def invoke(self, input_data: Any, **kwargs) -> Any:
        """Generic invoke - subclasses override for specific behavior"""
        return {"status": "success", "input": input_data, "agent": self.metadata.name}


class CrewAIController(GenericController):
    """Controller for CrewAI agents"""

    pass


class LangChainController(GenericController):
    """Controller for LangChain agents"""

    pass


class AutoGenController(GenericController):
    """Controller for AutoGen agents"""

    pass


# ============================================================================
# Export Layer - FastAPI and LangServe
# ============================================================================


def _create_fastapi_app(
    controller: AgentController, metadata: Optional[AgentMetadata] = None
) -> Any:
    """Create a FastAPI app wrapping an agent controller

    Args:
        controller: Agent controller instance
        metadata: Optional agent metadata for docs

    Returns:
        FastAPI app instance

    Example:
        >>> from ayn import GenericController, ControllerConfig, AgentMetadata, AgentFramework
        >>> meta = AgentMetadata(name="test", description="test", framework=AgentFramework.CUSTOM)
        >>> controller = GenericController(meta, ControllerConfig())
        >>> app = _create_fastapi_app(controller, meta)  # doctest: +SKIP
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
    except ImportError:
        raise ImportError("FastAPI is required: pip install fastapi uvicorn")

    app = FastAPI(
        title=metadata.name if metadata else "Agent API",
        description=metadata.description if metadata else "AI Agent API",
        version=metadata.version if metadata else "0.1.0",
    )

    class InvokeRequest(BaseModel):
        input_data: Any
        kwargs: dict = {}

    @app.post("/invoke")
    async def invoke_endpoint(request: InvokeRequest):
        """Invoke the agent"""
        result = await controller.ainvoke(request.input_data, **request.kwargs)
        return JSONResponse({"result": result})

    @app.post("/stream")
    async def stream_endpoint(request: InvokeRequest):
        """Stream agent outputs"""
        from fastapi.responses import StreamingResponse

        async def generate():
            for chunk in controller.stream(request.input_data, **request.kwargs):
                yield json.dumps({"chunk": chunk}) + "\n"

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    @app.get("/health")
    async def health():
        """Health check"""
        return {"status": "healthy"}

    @app.get("/metadata")
    async def get_metadata():
        """Get agent metadata"""
        if metadata:
            return metadata.to_dict()
        return {"message": "No metadata available"}

    return app


def export_as_fastapi(
    controller: AgentController,
    metadata: Optional[AgentMetadata] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """Export agent controller as FastAPI service

    Args:
        controller: Agent controller
        metadata: Optional agent metadata
        host: Host to bind to
        port: Port to bind to

    Example:
        >>> # This would start a server - doctest: +SKIP
        >>> controller = GenericController(...)  # doctest: +SKIP
        >>> export_as_fastapi(controller, port=8080)  # doctest: +SKIP
    """
    app = _create_fastapi_app(controller, metadata)

    try:
        import uvicorn

        uvicorn.run(app, host=host, port=port)
    except ImportError:
        raise ImportError("uvicorn is required: pip install uvicorn")


def export_as_langserve(
    controller: AgentController, metadata: Optional[AgentMetadata] = None
) -> Any:
    """Export agent as LangServe compatible service

    Args:
        controller: Agent controller
        metadata: Optional agent metadata

    Returns:
        LangServe app

    Example:
        >>> # Requires langserve - doctest: +SKIP
        >>> controller = GenericController(...)  # doctest: +SKIP
        >>> app = export_as_langserve(controller)  # doctest: +SKIP
    """
    try:
        from langserve import add_routes
        from fastapi import FastAPI
    except ImportError:
        raise ImportError("LangServe is required: pip install langserve")

    app = FastAPI()

    # Create a LangChain-compatible runnable wrapper
    class RunnableController:
        def __init__(self, controller):
            self.controller = controller

        def invoke(self, input_data, config=None):
            return self.controller.invoke(input_data)

        async def ainvoke(self, input_data, config=None):
            return await self.controller.ainvoke(input_data)

        def stream(self, input_data, config=None):
            return self.controller.stream(input_data)

    runnable = RunnableController(controller)
    add_routes(app, runnable, path="/agent")

    return app


# ============================================================================
# MCP (Model Context Protocol) Support
# ============================================================================


@dataclass
class MCPTool:
    """MCP tool definition

    Example:
        >>> tool = MCPTool(
        ...     name="search",
        ...     description="Search for information",
        ...     input_schema={"type": "object", "properties": {"query": {"type": "string"}}}
        ... )
        >>> tool.name
        'search'
    """

    name: str
    description: str
    input_schema: dict
    handler: Optional[Callable] = None

    def to_dict(self) -> dict:
        """Convert to MCP tool format"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass
class MCPResource:
    """MCP resource definition

    Example:
        >>> resource = MCPResource(
        ...     uri="file:///data.csv",
        ...     name="Dataset",
        ...     mimeType="text/csv"
        ... )
        >>> resource.name
        'Dataset'
    """

    uri: str
    name: str
    mimeType: str = "text/plain"
    description: str = ""

    def to_dict(self) -> dict:
        """Convert to MCP resource format"""
        return {
            "uri": self.uri,
            "name": self.name,
            "mimeType": self.mimeType,
            "description": self.description,
        }


@dataclass
class MCPPrompt:
    """MCP prompt template

    Example:
        >>> prompt = MCPPrompt(
        ...     name="summarize",
        ...     description="Summarize text",
        ...     arguments=[{"name": "text", "description": "Text to summarize", "required": True}]
        ... )
        >>> prompt.name
        'summarize'
    """

    name: str
    description: str
    arguments: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to MCP prompt format"""
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments,
        }


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


# ============================================================================
# Custom Actions - ChatGPT and Claude
# ============================================================================


def _generate_openapi_spec(
    controller: AgentController, metadata: AgentMetadata
) -> dict:
    """Generate OpenAPI spec for agent

    Uses oa package if available for better spec generation.

    Args:
        controller: Agent controller
        metadata: Agent metadata

    Returns:
        OpenAPI 3.0 specification dict
    """
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": metadata.name,
            "description": metadata.description,
            "version": metadata.version,
        },
        "servers": [{"url": "https://api.example.com"}],
        "paths": {
            "/invoke": {
                "post": {
                    "summary": "Invoke the agent",
                    "operationId": "invokeAgent",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "input_data": {"type": "object"},
                                        "kwargs": {"type": "object"},
                                    },
                                    "required": ["input_data"],
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {"result": {"type": "object"}},
                                    }
                                }
                            },
                        }
                    },
                }
            }
        },
    }

    return spec


def generate_chatgpt_action(
    controller: AgentController, metadata: AgentMetadata, api_url: str
) -> dict:
    """Generate ChatGPT custom action configuration

    Args:
        controller: Agent controller
        metadata: Agent metadata
        api_url: Base URL for the API

    Returns:
        ChatGPT action configuration

    Example:
        >>> meta = AgentMetadata(name="test", description="test", framework=AgentFramework.CUSTOM)
        >>> controller = GenericController(meta, ControllerConfig())
        >>> action = generate_chatgpt_action(controller, meta, "https://api.example.com")
        >>> action['schema']['info']['title']
        'test'
    """
    openapi_spec = _generate_openapi_spec(controller, metadata)
    openapi_spec["servers"][0]["url"] = api_url

    return {
        "schema": openapi_spec,
        "authentication": {"type": "none"},  # Configure as needed
        "instructions": f"Use this action to {metadata.description.lower()}",
    }


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


# ============================================================================
# Convenience Functions
# ============================================================================


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
        >>> meta = AgentMetadata(name="test", description="test", framework=AgentFramework.CUSTOM)
        >>> controller = GenericController(meta, ControllerConfig())
        >>> artifacts = export_agent_full_stack(
        ...     controller, meta,
        ...     export_fastapi=False, export_mcp=False, export_chatgpt=True
        ... )
        >>> 'chatgpt_action' in artifacts
        True
    """
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


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Data models
    "AgentFramework",
    "AgentModality",
    "AgentMetadata",
    # Search
    "AgentSearcher",
    "GitHubAgentSearcher",
    "HuggingFaceAgentSearcher",
    "AwesomeListSearcher",
    "AgentRegistry",
    # Controller
    "AgentController",
    "ControllerConfig",
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


if __name__ == "__main__":
    # Quick example
    print("AYN - Agents You Need")
    print("=" * 50)

    # Create registry
    registry = AgentRegistry(searchers=[])  # Empty for demo

    # Add a test agent
    test_agent = AgentMetadata(
        name="data-prep-agent",
        description="Prepares data for visualization",
        framework=AgentFramework.CREWAI,
        tags=["data", "preparation", "visualization"],
    )
    registry["data-prep-agent"] = test_agent

    # Search
    results = registry.search("data")
    print(f"\nFound {len(results)} agents for 'data'")
    for result in results:
        print(f"  - {result.name}: {result.description}")

    # Create controller
    controller = BaseAgentController.from_metadata(test_agent)
    print(f"\nCreated controller: {controller.__class__.__name__}")

    # Test invoke
    result = controller.invoke({"data": [1, 2, 3]})
    print(f"Result: {result}")

    # Export
    print("\nExporting agent with all integrations...")
    artifacts = export_agent_full_stack(
        controller,
        test_agent,
        export_fastapi=False,  # Don't actually start server
        export_mcp=True,
        export_chatgpt=True,
    )

    if "chatgpt_action" in artifacts:
        print("\nChatGPT Action Config:")
        print(json.dumps(artifacts["chatgpt_action"]["schema"]["info"], indent=2))

    if "claude_config" in artifacts:
        print("\nClaude MCP Config:")
        print(json.dumps(artifacts["claude_config"], indent=2))

    print("\n✓ AYN initialized successfully!")
