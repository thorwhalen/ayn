# AYN Quick Reference

## Common Patterns

### Pattern 1: Discover and Use Agent
```python
from ayn import AgentRegistry, create_agent_from_registry

registry = AgentRegistry()
controller = create_agent_from_registry(registry, 'search query')
result = controller.invoke(input_data)
```

### Pattern 2: Register Custom Agent
```python
from ayn import AgentRegistry, AgentMetadata, AgentFramework

metadata = AgentMetadata(
    name="my-agent",
    description="What it does",
    framework=AgentFramework.CUSTOM,
    tags=["tag1", "tag2"]
)

registry = AgentRegistry()
registry['my-agent'] = metadata
```

### Pattern 3: Create Custom Controller
```python
from ayn import GenericController

class MyController(GenericController):
    def invoke(self, input_data, **kwargs):
        # Your logic
        return {"result": "processed"}
```

### Pattern 4: Export as API
```python
from ayn import export_as_fastapi

export_as_fastapi(controller, metadata, host="0.0.0.0", port=8000)
# Creates /invoke, /stream, /health, /metadata endpoints
```

### Pattern 5: Create MCP Server
```python
from ayn import controller_to_mcp_server, MCPTool

mcp_server = controller_to_mcp_server(
    controller,
    metadata,
    tools=[
        MCPTool(
            name="my_tool",
            description="Does X",
            input_schema={"type": "object", ...},
            handler=my_function
        )
    ]
)
```

### Pattern 6: Full Stack Export
```python
from ayn import export_agent_full_stack

artifacts = export_agent_full_stack(
    controller, metadata,
    export_fastapi=True,
    export_mcp=True,
    export_chatgpt=True
)

# Access:
fastapi_app = artifacts['fastapi_app']
mcp_server = artifacts['mcp_server']
chatgpt_config = artifacts['chatgpt_action']
claude_config = artifacts['claude_config']
```

### Pattern 7: Framework Integration

#### CrewAI
```python
from crewai import Crew, Agent
from ayn import BaseAgentController

class CrewAIController(BaseAgentController):
    def __init__(self, metadata, config):
        super().__init__(config)
        self.agent = Agent(role=metadata.description, ...)
        self.crew = Crew(agents=[self.agent], ...)
    
    def invoke(self, input_data, **kwargs):
        return self.crew.kickoff(inputs=input_data)
```

#### LangChain
```python
from langchain.agents import AgentExecutor
from ayn import BaseAgentController

class LangChainController(BaseAgentController):
    def __init__(self, metadata, config):
        super().__init__(config)
        self.executor = AgentExecutor(...)
    
    def invoke(self, input_data, **kwargs):
        return self.executor.invoke(input_data)
```

### Pattern 8: Async Operations
```python
# Async invoke
result = await controller.ainvoke(input_data)

# Streaming
for chunk in controller.stream(input_data):
    print(chunk)
```

## Common Use Cases

### Use Case 1: Data Analysis Agent
```python
metadata = AgentMetadata(
    name="data-analyst",
    description="Analyzes data and provides insights",
    framework=AgentFramework.CUSTOM,
    capabilities=["pandas", "statistics", "visualization"]
)

class DataAnalyst(GenericController):
    def invoke(self, input_data, **kwargs):
        data = input_data['data']
        return {
            "mean": sum(data)/len(data),
            "max": max(data),
            "min": min(data)
        }
```

### Use Case 2: RAG Agent
```python
metadata = AgentMetadata(
    name="rag-agent",
    description="Retrieval-augmented generation for Q&A",
    framework=AgentFramework.LANGCHAIN,
    capabilities=["vector-search", "embeddings", "llm"]
)

class RAGController(LangChainController):
    def invoke(self, input_data, **kwargs):
        query = input_data['query']
        # Retrieve relevant docs
        docs = self.retriever.get_relevant_documents(query)
        # Generate answer
        return self.llm.generate(query, context=docs)
```

### Use Case 3: Code Generation Agent
```python
metadata = AgentMetadata(
    name="code-gen",
    description="Generates code from natural language",
    framework=AgentFramework.AUTOGEN,
    capabilities=["code-generation", "python", "testing"]
)
```

### Use Case 4: Multi-Agent System
```python
# Register multiple agents
registry['analyst'] = analyst_metadata
registry['coder'] = coder_metadata
registry['reviewer'] = reviewer_metadata

# Create controllers
analyst = create_agent_from_registry(registry, 'analyst')
coder = create_agent_from_registry(registry, 'coder')
reviewer = create_agent_from_registry(registry, 'reviewer')

# Use in sequence
analysis = analyst.invoke(data)
code = coder.invoke(analysis)
review = reviewer.invoke(code)
```

## API Endpoints (FastAPI Export)

### POST /invoke
```bash
curl -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{"input_data": {"query": "test"}, "kwargs": {}}'
```

### POST /stream
```bash
curl -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"input_data": {"query": "test"}}'
```

### GET /health
```bash
curl http://localhost:8000/health
```

### GET /metadata
```bash
curl http://localhost:8000/metadata
```

## MCP JSON-RPC Methods

### tools/list
```json
{
  "jsonrpc": "2.0",
  "method": "tools/list",
  "id": 1
}
```

### tools/call
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "my_tool",
    "arguments": {"arg1": "value1"}
  },
  "id": 2
}
```

### resources/list
```json
{
  "jsonrpc": "2.0",
  "method": "resources/list",
  "id": 3
}
```

### prompts/list
```json
{
  "jsonrpc": "2.0",
  "method": "prompts/list",
  "id": 4
}
```

## Configuration Files

### Claude Desktop MCP Config
Location: `~/.config/claude-desktop/config.json` (Linux/Mac)
or `%APPDATA%\Claude\config.json` (Windows)

```json
{
  "mcpServers": {
    "my-agent": {
      "url": "http://localhost:8000/sse",
      "transport": {"type": "sse"}
    }
  }
}
```

### ChatGPT Custom Action
1. Go to ChatGPT → Configure GPT → Actions
2. Click "Create new action"
3. Paste OpenAPI schema from `generate_chatgpt_action()`
4. Configure authentication if needed
5. Test and deploy

## Environment Variables

```bash
# API keys
export GITHUB_TOKEN="your_github_token"
export HF_TOKEN="your_huggingface_token"
export OPENAI_API_KEY="your_openai_key"

# Configuration
export AYN_CACHE_DIR="~/.ayn/cache"
export AYN_REGISTRY_DIR="~/.ayn/registry"
```

## Troubleshooting

### Issue: Agent not found
```python
# Check registry
print(list(registry.keys()))

# Search with different query
results = registry.search('alternative query')
```

### Issue: Controller error
```python
# Check config
print(controller.config.to_dict())

# Try with different config
new_config = ControllerConfig(model="gpt-3.5-turbo")
controller = BaseAgentController.from_metadata(metadata, new_config)
```

### Issue: MCP connection failed
```python
# Check server is running
handler = mcp_server.to_json_rpc_handler()
response = handler({"jsonrpc": "2.0", "method": "tools/list", "id": 1})
print(response)
```

## Performance Tips

1. **Cache Registry**: Save registry to disk
2. **Batch Operations**: Process multiple inputs together
3. **Async When Possible**: Use `ainvoke()` for concurrent requests
4. **Stream Large Outputs**: Use `stream()` instead of `invoke()`
5. **Limit Tool Count**: Keep MCP tools focused

## Best Practices

1. **Descriptive Names**: Use clear agent names
2. **Rich Metadata**: Add tags, capabilities, descriptions
3. **Error Handling**: Implement retry logic in controllers
4. **Validation**: Validate input schemas
5. **Documentation**: Document custom tools and resources
6. **Versioning**: Track agent versions
7. **Testing**: Test agents before exporting
8. **Security**: Validate inputs, sanitize outputs

## Integration Checklist

- [ ] Define AgentMetadata
- [ ] Implement controller (or use GenericController)
- [ ] Test invoke() method
- [ ] Register in AgentRegistry
- [ ] Add MCP tools if needed
- [ ] Generate OpenAPI spec
- [ ] Test FastAPI endpoints
- [ ] Configure authentication
- [ ] Create ChatGPT action
- [ ] Setup Claude MCP config
- [ ] Deploy and monitor

## Resources

- **Documentation**: README.md
- **Tutorial**: ayn_tutorial.py
- **Examples**: Search for "Example:" in ayn.py
- **Doctests**: Run `python -m doctest ayn.py`
