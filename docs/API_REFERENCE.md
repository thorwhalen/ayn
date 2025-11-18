# AYN API Reference

This document provides detailed API reference for all AYN modules.

## Core Models (`ayn.core`)

### AgentFramework
```python
class AgentFramework(Enum):
    CREWAI = "crewai"
    LANGCHAIN = "langchain"
    LANGGRAPH = "langgraph"
    AUTOGEN = "autogen"
    SWARMS = "swarms"
    SMOLAGENTS = "smolagents"
    SUPERAGI = "superagi"
    CUSTOM = "custom"
```

### AgentMetadata
```python
@dataclass
class AgentMetadata:
    name: str
    description: str
    framework: AgentFramework
    source: str = ""
    tags: List[str] = []
    modality: AgentModality = AgentModality.TEXT
    version: str = "0.1.0"
    author: str = ""
    license: str = ""
    dependencies: List[str] = []
    capabilities: List[str] = []
```

### ControllerConfig
```python
@dataclass
class ControllerConfig:
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 60
    retry_count: int = 3
    api_key: Optional[str] = None
    additional_params: dict = {}
```

## Registry (`ayn.registry`)

### AgentRegistry
```python
class AgentRegistry(MutableMapping[str, AgentMetadata]):
    def __init__(
        self,
        searchers: Optional[List[AgentSearcher]] = None,
        cache_dir: Optional[str] = None,
    )

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        source: Optional[str] = None,
        **filters,
    ) -> List[AgentMetadata]
```

## Controllers (`ayn.controllers`)

### BaseAgentController
```python
class BaseAgentController(ABC):
    def __init__(self, config: ControllerConfig)

    @abstractmethod
    def invoke(self, input_data: Any, **kwargs) -> Any

    async def ainvoke(self, input_data: Any, **kwargs) -> Any

    def stream(self, input_data: Any, **kwargs) -> Iterable[Any]

    @classmethod
    def from_metadata(
        cls, metadata: AgentMetadata, config: Optional[ControllerConfig] = None
    ) -> BaseAgentController
```

## Export (`ayn.export`)

### export_as_fastapi
```python
def export_as_fastapi(
    controller: AgentController,
    metadata: Optional[AgentMetadata] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
)
```

### export_as_langserve
```python
def export_as_langserve(
    controller: AgentController,
    metadata: Optional[AgentMetadata] = None
) -> Any
```

## MCP (`ayn.mcp`)

### MCPServer
```python
class MCPServer:
    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        controller: Optional[AgentController] = None,
    )

    def add_tool(self, tool: MCPTool)
    def add_resource(self, resource: MCPResource)
    def add_prompt(self, prompt: MCPPrompt)
    def to_json_rpc_handler(self) -> Callable
```

### controller_to_mcp_server
```python
def controller_to_mcp_server(
    controller: AgentController,
    metadata: AgentMetadata,
    *,
    tools: Optional[List[MCPTool]] = None,
    resources: Optional[List[MCPResource]] = None,
    prompts: Optional[List[MCPPrompt]] = None,
) -> MCPServer
```

## Actions (`ayn.actions`)

### generate_chatgpt_action
```python
def generate_chatgpt_action(
    controller: AgentController,
    metadata: AgentMetadata,
    api_url: str
) -> dict
```

### generate_claude_mcp_config
```python
def generate_claude_mcp_config(
    mcp_server: MCPServer,
    server_url: str,
    transport: str = "sse"
) -> dict
```

## Utilities (`ayn.utils`)

### create_agent_from_registry
```python
def create_agent_from_registry(
    registry: AgentRegistry,
    query: str,
    config: Optional[ControllerConfig] = None
) -> BaseAgentController
```

### export_agent_full_stack
```python
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
) -> dict
```

See individual module documentation for detailed usage examples.
