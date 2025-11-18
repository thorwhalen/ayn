# AYN (Agents You Need)

**The meta-framework for AI agent interoperability**

Think "pip + Docker Hub + Postman" for AI agents.

---

## 🎯 Vision

AYN solves the agent discovery and integration problem by providing:

1. **Unified Search** across GitHub, HuggingFace, and awesome lists
2. **Standard Controller Interface** for invoking any agent
3. **HTTP/REST Export** via FastAPI and LangServe
4. **MCP Support** for Model Context Protocol integration
5. **Custom Actions** for ChatGPT and Claude

## 🚀 Quick Start

```python
from ayn import AgentRegistry, BaseAgentController, export_agent_full_stack

# 1. Search for agents
registry = AgentRegistry()
results = registry.search('data preparation')

# 2. Create controller
controller = BaseAgentController.from_metadata(results[0])

# 3. Use the agent
result = controller.invoke({'data': [1, 2, 3]})

# 4. Export with all integrations
artifacts = export_agent_full_stack(
    controller, 
    results[0],
    export_fastapi=True,
    export_mcp=True,
    export_chatgpt=True
)
```

## 📦 Installation

```bash
# Basic installation
pip install ayn

# With FastAPI support
pip install ayn[fastapi]

# With LangServe support
pip install ayn[langserve]

# Full installation
pip install ayn[all]
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AYN Framework                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Search     │  │  Controller  │  │    Export    │    │
│  │   Registry   │  │  Interface   │  │    Layer     │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│         │                  │                  │            │
│  ┌──────▼──────┐  ┌────────▼────────┐  ┌─────▼─────┐    │
│  │  GitHub     │  │  BaseAgent      │  │  FastAPI  │    │
│  │  HuggingFace│  │  Controller     │  │  LangServe│    │
│  │  AwesomeLists│ └─────────────────┘  └───────────┘    │
│  └─────────────┘                                          │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  MCP Layer   │  │   Actions    │  │  Frameworks  │    │
│  │  Tools       │  │   ChatGPT    │  │  CrewAI      │    │
│  │  Resources   │  │   Claude     │  │  LangChain   │    │
│  │  Prompts     │  │   Gemini     │  │  AutoGen     │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 🔑 Core Concepts

### 1. AgentMetadata

Describes an agent with standard attributes:

```python
from ayn import AgentMetadata, AgentFramework

metadata = AgentMetadata(
    name="my-agent",
    description="Does something useful",
    framework=AgentFramework.CREWAI,
    tags=["data", "analysis"],
    capabilities=["pandas", "numpy"],
    source="https://github.com/user/repo"
)
```

### 2. AgentRegistry

Dict-like interface for managing agents:

```python
from ayn import AgentRegistry

registry = AgentRegistry()

# Add agent
registry['my-agent'] = metadata

# Search
results = registry.search('data analysis')

# Dict operations
len(registry)  # Count
list(registry)  # List names
'my-agent' in registry  # Check existence
```

### 3. AgentController

Standard interface for invoking agents:

```python
from ayn import BaseAgentController, ControllerConfig

config = ControllerConfig(
    model="gpt-4",
    temperature=0.7,
    max_tokens=2000
)

controller = BaseAgentController.from_metadata(metadata, config)

# Sync
result = controller.invoke(input_data)

# Async
result = await controller.ainvoke(input_data)

# Stream
for chunk in controller.stream(input_data):
    print(chunk)
```

### 4. Export as HTTP API

```python
from ayn import export_as_fastapi

# Creates FastAPI app with:
# - POST /invoke
# - POST /stream
# - GET /health
# - GET /metadata

export_as_fastapi(
    controller, 
    metadata,
    host="0.0.0.0",
    port=8000
)
```

### 5. MCP (Model Context Protocol)

```python
from ayn import controller_to_mcp_server, MCPTool

# Create MCP server
mcp_server = controller_to_mcp_server(
    controller,
    metadata,
    tools=[
        MCPTool(
            name="my_tool",
            description="Does something",
            input_schema={"type": "object", ...},
            handler=lambda x: process(x)
        )
    ]
)

# JSON-RPC handler
handler = mcp_server.to_json_rpc_handler()
response = handler({
    "jsonrpc": "2.0",
    "method": "tools/list",
    "id": 1
})
```

### 6. Custom Actions

```python
from ayn import generate_chatgpt_action, generate_claude_mcp_config

# ChatGPT Custom GPT
chatgpt_config = generate_chatgpt_action(
    controller,
    metadata,
    api_url="https://api.myagent.com"
)

# Claude Desktop MCP
claude_config = generate_claude_mcp_config(
    mcp_server,
    "http://localhost:8000"
)
```

## 📚 Examples

### Example 1: Custom Agent Implementation

```python
from ayn import GenericController, AgentMetadata, AgentFramework

class MyAgent(GenericController):
    def invoke(self, input_data, **kwargs):
        # Your custom logic here
        result = self.process(input_data)
        return {
            "status": "success",
            "result": result,
            "agent": self.metadata.name
        }
    
    def process(self, data):
        # Implementation
        return data

# Create and use
metadata = AgentMetadata(
    name="my-custom-agent",
    description="My custom agent",
    framework=AgentFramework.CUSTOM
)

agent = MyAgent(metadata, ControllerConfig())
result = agent.invoke({"query": "Hello"})
```

### Example 2: Full Stack Export

```python
from ayn import export_agent_full_stack

# Export everything at once
artifacts = export_agent_full_stack(
    controller,
    metadata,
    api_host="0.0.0.0",
    api_port=8000,
    export_fastapi=True,
    export_mcp=True,
    export_chatgpt=True
)

# Access artifacts
fastapi_app = artifacts['fastapi_app']
mcp_server = artifacts['mcp_server']
chatgpt_action = artifacts['chatgpt_action']
claude_config = artifacts['claude_config']
```

### Example 3: Framework Integration

```python
# Integrating CrewAI
from crewai import Crew, Agent, Task

class CrewAIController(BaseAgentController):
    def __init__(self, metadata, config):
        super().__init__(config)
        self.metadata = metadata
        
        # Setup CrewAI components
        self.agent = Agent(
            role=metadata.description,
            goal="Complete the task",
            backstory="An AI agent"
        )
        
        self.crew = Crew(
            agents=[self.agent],
            tasks=[]
        )
    
    def invoke(self, input_data, **kwargs):
        task = Task(
            description=input_data.get('description'),
            agent=self.agent
        )
        return self.crew.kickoff(inputs=input_data)

# Use it
controller = CrewAIController(metadata, config)
result = controller.invoke({"description": "Analyze data"})
```

## 🔧 Advanced Features

### Custom Searchers

```python
from ayn import AgentSearcher, AgentRegistry

class MyCustomSearcher:
    def search(self, query, *, limit=20, **filters):
        # Your search logic
        results = []
        # ... search implementation
        return results

registry = AgentRegistry(searchers=[MyCustomSearcher()])
```

### MCP Resources and Prompts

```python
from ayn import MCPResource, MCPPrompt

# Add resources
resource = MCPResource(
    uri="file:///data/dataset.csv",
    name="Training Dataset",
    mimeType="text/csv",
    description="Dataset for training"
)

# Add prompts
prompt = MCPPrompt(
    name="analyze",
    description="Analyze data",
    arguments=[
        {
            "name": "data",
            "description": "Data to analyze",
            "required": True
        }
    ]
)

mcp_server = controller_to_mcp_server(
    controller,
    metadata,
    resources=[resource],
    prompts=[prompt]
)
```

### Streaming Responses

```python
class StreamingAgent(GenericController):
    def stream(self, input_data, **kwargs):
        # Yield chunks as they become available
        for i in range(10):
            yield {"chunk": i, "data": f"Processing {i}"}
            time.sleep(0.1)

# Use
for chunk in agent.stream(input_data):
    print(chunk)
```

## 🛠️ Development

### Running Tests

```python
# Run doctests
python -m doctest ayn.py -v

# Run tutorial
python ayn_tutorial.py
```

### Project Structure

```
ayn/
├── ayn.py              # Core module
├── ayn_tutorial.py     # Comprehensive tutorial
├── README.md           # This file
└── test_ayn.py         # Tests (TODO)
```

## 🗺️ Roadmap

### Phase 1: Core (✅ Complete)
- ✅ Agent metadata and registry
- ✅ Standard controller interface
- ✅ FastAPI export
- ✅ MCP server support
- ✅ ChatGPT/Claude actions

### Phase 2: Search (In Progress)
- ⏳ GitHub API integration
- ⏳ HuggingFace search
- ⏳ Awesome list parsing
- ⏳ Caching and indexing

### Phase 3: Installation (TODO)
- ⬜ Dependency management (uv/pipx)
- ⬜ Docker support
- ⬜ Environment isolation
- ⬜ Version management

### Phase 4: Frameworks (TODO)
- ⬜ CrewAI integration
- ⬜ LangChain/LangGraph integration
- ⬜ AutoGen integration
- ⬜ SmolAgents integration

### Phase 5: Advanced (TODO)
- ⬜ Agent monitoring
- ⬜ Logging and observability
- ⬜ Performance benchmarking
- ⬜ Cost tracking

## 🤝 Contributing

Contributions welcome! Key areas:

1. **Searchers**: Implement real API integrations
2. **Controllers**: Add framework-specific controllers
3. **Export**: Additional export formats
4. **Tools**: MCP tools and resources
5. **Documentation**: Examples and guides

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

Built on top of:
- **FastAPI** for HTTP APIs
- **LangServe** for LangChain integration
- **Model Context Protocol** (Anthropic)
- **thorwhalen1's packages**: dol, i2, graze, meshed, oa

Inspired by:
- Python's pip for package management
- Docker Hub for container discovery
- Postman for API testing
- OpenAPI for API specifications

## 📞 Contact

- Issues: [GitHub Issues](https://github.com/yourusername/ayn/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/ayn/discussions)

---

**AYN** - Making AI agents discoverable, interoperable, and deployable. 🚀
