"""AYN Tutorial - Agents You Need

This tutorial demonstrates all major features of AYN:
1. Search & Discovery
2. Controller Interface
3. FastAPI Export
4. MCP Server Integration
5. ChatGPT/Claude Custom Actions

AYN is the meta-framework for agent interoperability - think "pip + Docker Hub + Postman" for AI agents.
"""

import json
from ayn import (
    AgentRegistry,
    AgentMetadata,
    AgentFramework,
    BaseAgentController,
    ControllerConfig,
    GenericController,
    MCPTool,
    MCPServer,
    controller_to_mcp_server,
    generate_chatgpt_action,
    generate_claude_mcp_config,
    export_agent_full_stack,
    create_agent_from_registry,
)


# ============================================================================
# Example 1: Agent Registry and Search
# ============================================================================

def example_registry_search():
    """Demonstrates agent registry and search functionality"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Agent Registry and Search")
    print("=" * 70)
    
    # Create a registry (with empty searchers for this demo)
    registry = AgentRegistry(searchers=[])
    
    # Add some sample agents
    agents = [
        AgentMetadata(
            name="crewai-data-analyst",
            description="Multi-agent crew for data analysis and visualization",
            framework=AgentFramework.CREWAI,
            tags=["data", "analysis", "visualization"],
            capabilities=["pandas", "matplotlib", "statistical-analysis"],
            source="https://github.com/example/crewai-data-analyst",
        ),
        AgentMetadata(
            name="langchain-rag-agent",
            description="RAG agent for document Q&A",
            framework=AgentFramework.LANGCHAIN,
            tags=["rag", "qa", "documents"],
            capabilities=["vector-search", "embeddings", "llm-qa"],
            source="https://github.com/example/langchain-rag",
        ),
        AgentMetadata(
            name="autogen-coder",
            description="Code generation and debugging agent",
            framework=AgentFramework.AUTOGEN,
            tags=["code", "debugging", "generation"],
            capabilities=["python", "javascript", "debugging"],
            source="https://github.com/example/autogen-coder",
        ),
    ]
    
    for agent in agents:
        registry[agent.name] = agent
        print(f"✓ Registered: {agent.name}")
    
    # Search the registry
    print("\nSearching for 'data' agents:")
    results = registry.search("data")
    for result in results:
        print(f"  - {result.name} ({result.framework.value})")
        print(f"    Tags: {', '.join(result.tags)}")
    
    # Dict-like interface
    print(f"\nTotal agents in registry: {len(registry)}")
    print(f"Registry keys: {list(registry.keys())}")
    
    return registry


# ============================================================================
# Example 2: Controller Interface with Configuration
# ============================================================================

def example_controller_interface():
    """Demonstrates the standard controller interface"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Controller Interface")
    print("=" * 70)
    
    # Create agent metadata
    metadata = AgentMetadata(
        name="my-custom-agent",
        description="Custom agent for processing data",
        framework=AgentFramework.CUSTOM,
    )
    
    # Create controller with configuration
    config = ControllerConfig(
        model="gpt-4",
        temperature=0.7,
        max_tokens=2000,
        timeout=120,
    )
    
    controller = GenericController(metadata, config)
    print(f"Created controller: {controller.__class__.__name__}")
    print(f"Config: {controller.config.to_dict()}")
    
    # Synchronous invocation
    print("\nSynchronous invoke:")
    input_data = {"query": "Analyze sales data", "data": [100, 200, 150]}
    result = controller.invoke(input_data)
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Streaming (default implementation yields single result)
    print("\nStreaming:")
    for chunk in controller.stream(input_data):
        print(f"Chunk: {chunk}")
    
    return controller


# ============================================================================
# Example 3: Custom Agent Implementation
# ============================================================================

class DataProcessingAgent(GenericController):
    """Example custom agent that processes data"""
    
    def invoke(self, input_data, **kwargs):
        """Process data and return insights"""
        data = input_data.get("data", [])
        
        if not data:
            return {"error": "No data provided"}
        
        # Simulate some processing
        total = sum(data)
        avg = total / len(data)
        max_val = max(data)
        min_val = min(data)
        
        return {
            "status": "success",
            "agent": self.metadata.name,
            "insights": {
                "total": total,
                "average": avg,
                "max": max_val,
                "min": min_val,
                "count": len(data),
            },
            "raw_data": data,
        }


def example_custom_agent():
    """Demonstrates creating a custom agent"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Custom Agent Implementation")
    print("=" * 70)
    
    metadata = AgentMetadata(
        name="data-processor",
        description="Processes numerical data and provides insights",
        framework=AgentFramework.CUSTOM,
        capabilities=["statistics", "data-analysis"],
    )
    
    agent = DataProcessingAgent(metadata, ControllerConfig())
    
    # Test with sample data
    test_data = {"data": [10, 25, 15, 30, 20, 35, 18, 22]}
    result = agent.invoke(test_data)
    
    print("Input data:", test_data["data"])
    print("\nResults:")
    print(json.dumps(result, indent=2))
    
    return agent


# ============================================================================
# Example 4: MCP Server Integration
# ============================================================================

def example_mcp_server():
    """Demonstrates MCP server creation and usage"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: MCP (Model Context Protocol) Server")
    print("=" * 70)
    
    # Create a controller
    metadata = AgentMetadata(
        name="analysis-agent",
        description="Data analysis agent with MCP support",
        framework=AgentFramework.CUSTOM,
    )
    controller = GenericController(metadata, ControllerConfig())
    
    # Define custom MCP tools
    custom_tools = [
        MCPTool(
            name="calculate_stats",
            description="Calculate statistics for a dataset",
            input_schema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Numerical data to analyze",
                    }
                },
                "required": ["data"],
            },
            handler=lambda data: {
                "mean": sum(data) / len(data) if data else 0,
                "max": max(data) if data else None,
                "min": min(data) if data else None,
            },
        ),
    ]
    
    # Create MCP server
    mcp_server = controller_to_mcp_server(
        controller,
        metadata,
        tools=custom_tools,
    )
    
    print(f"MCP Server: {mcp_server.name} v{mcp_server.version}")
    print(f"Tools available: {len(mcp_server.tools)}")
    for tool in mcp_server.tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # Test JSON-RPC handler
    handler = mcp_server.to_json_rpc_handler()
    
    # List tools
    request = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}
    response = handler(request)
    print(f"\nJSON-RPC tools/list response:")
    print(json.dumps(response["result"], indent=2))
    
    # Call a tool
    request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "calculate_stats",
            "arguments": {"data": [10, 20, 30, 40, 50]},
        },
        "id": 2,
    }
    response = handler(request)
    print(f"\nJSON-RPC tools/call response:")
    print(json.dumps(response["result"], indent=2))
    
    # Generate Claude Desktop config
    claude_config = generate_claude_mcp_config(
        mcp_server, "http://localhost:8000"
    )
    print(f"\nClaude Desktop MCP Configuration:")
    print(json.dumps(claude_config, indent=2))
    
    return mcp_server


# ============================================================================
# Example 5: ChatGPT Custom Action
# ============================================================================

def example_chatgpt_action():
    """Demonstrates generating ChatGPT custom actions"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: ChatGPT Custom Action Generation")
    print("=" * 70)
    
    metadata = AgentMetadata(
        name="text-summarizer",
        description="Summarizes long text documents into concise summaries",
        framework=AgentFramework.CUSTOM,
        version="1.0.0",
    )
    
    controller = GenericController(metadata, ControllerConfig())
    
    # Generate ChatGPT action config
    action_config = generate_chatgpt_action(
        controller,
        metadata,
        api_url="https://api.myagent.com",
    )
    
    print("ChatGPT Custom Action Configuration:")
    print(json.dumps(action_config, indent=2))
    
    print("\n📝 To use this in ChatGPT:")
    print("1. Go to ChatGPT → Settings → Actions")
    print("2. Click 'Create new action'")
    print("3. Paste the schema above")
    print("4. Save and test!")
    
    return action_config


# ============================================================================
# Example 6: Full Stack Export
# ============================================================================

def example_full_stack_export():
    """Demonstrates exporting agent with all integrations"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Full Stack Export (FastAPI + MCP + ChatGPT)")
    print("=" * 70)
    
    # Create agent
    metadata = AgentMetadata(
        name="universal-agent",
        description="Universal agent with all export options",
        framework=AgentFramework.CUSTOM,
        version="1.0.0",
    )
    
    controller = GenericController(metadata, ControllerConfig())
    
    # Export with all integrations (but don't actually start server)
    artifacts = export_agent_full_stack(
        controller,
        metadata,
        api_host="0.0.0.0",
        api_port=8000,
        export_fastapi=False,  # Set True to actually start server
        export_mcp=True,
        export_chatgpt=True,
    )
    
    print("Export artifacts generated:")
    for key in artifacts.keys():
        print(f"  ✓ {key}")
    
    if "chatgpt_action" in artifacts:
        print("\nChatGPT Action OpenAPI Info:")
        print(json.dumps(artifacts["chatgpt_action"]["schema"]["info"], indent=2))
    
    if "claude_config" in artifacts:
        print("\nClaude MCP Config:")
        print(json.dumps(artifacts["claude_config"], indent=2))
    
    print("\n💡 Deployment instructions:")
    print("1. Deploy FastAPI app: `uvicorn app:app --host 0.0.0.0 --port 8000`")
    print("2. Add Claude config to ~/.config/claude-desktop/config.json")
    print("3. Import ChatGPT action schema into ChatGPT settings")
    
    return artifacts


# ============================================================================
# Example 7: Complete Workflow
# ============================================================================

def example_complete_workflow():
    """Demonstrates a complete workflow from discovery to deployment"""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Complete Workflow (Discovery → Deploy)")
    print("=" * 70)
    
    # Step 1: Create registry and add agents
    print("\n[Step 1] Creating registry...")
    registry = AgentRegistry(searchers=[])
    
    agent_meta = AgentMetadata(
        name="workflow-agent",
        description="Agent for complete workflow demo",
        framework=AgentFramework.CREWAI,
        tags=["workflow", "demo"],
    )
    registry[agent_meta.name] = agent_meta
    print(f"✓ Registered {agent_meta.name}")
    
    # Step 2: Search and retrieve
    print("\n[Step 2] Searching registry...")
    results = registry.search("workflow")
    print(f"✓ Found {len(results)} agents")
    
    # Step 3: Create controller
    print("\n[Step 3] Creating controller...")
    controller = create_agent_from_registry(registry, "workflow")
    print(f"✓ Controller created: {controller.__class__.__name__}")
    
    # Step 4: Test invocation
    print("\n[Step 4] Testing agent...")
    result = controller.invoke({"task": "process", "data": [1, 2, 3]})
    print(f"✓ Result: {result}")
    
    # Step 5: Export everything
    print("\n[Step 5] Exporting all integrations...")
    artifacts = export_agent_full_stack(
        controller,
        agent_meta,
        export_fastapi=False,
        export_mcp=True,
        export_chatgpt=True,
    )
    print(f"✓ Generated {len(artifacts)} artifacts")
    
    print("\n✅ Complete workflow finished successfully!")
    
    return {
        "registry": registry,
        "controller": controller,
        "artifacts": artifacts,
    }


# ============================================================================
# Example 8: Integration with External Frameworks
# ============================================================================

def example_framework_integration():
    """Shows how to integrate existing agent frameworks"""
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Framework Integration Pattern")
    print("=" * 70)
    
    print("""
The pattern for integrating existing frameworks (CrewAI, LangChain, AutoGen):

1. Create a controller subclass:

class CrewAIController(BaseAgentController):
    def __init__(self, metadata, config):
        super().__init__(config)
        self.metadata = metadata
        # Initialize CrewAI crew here
        # self.crew = Crew(agents=[...], tasks=[...])
    
    def invoke(self, input_data, **kwargs):
        # Run the crew
        # return self.crew.kickoff(inputs=input_data)
        pass

2. Register in AgentRegistry:

metadata = AgentMetadata(
    name="my-crew",
    framework=AgentFramework.CREWAI,
    ...
)
registry['my-crew'] = metadata

3. Use standard interface:

controller = BaseAgentController.from_metadata(metadata)
result = controller.invoke(data)

4. Export with all integrations:

artifacts = export_agent_full_stack(controller, metadata)
    """)
    
    print("See the ayn.py source code for more integration examples.")


# ============================================================================
# Main Tutorial Runner
# ============================================================================

def run_all_examples():
    """Run all tutorial examples"""
    print("\n" + "=" * 70)
    print("AYN - AGENTS YOU NEED: Complete Tutorial")
    print("Meta-framework for Agent Interoperability")
    print("=" * 70)
    
    examples = [
        ("Registry & Search", example_registry_search),
        ("Controller Interface", example_controller_interface),
        ("Custom Agent", example_custom_agent),
        ("MCP Server", example_mcp_server),
        ("ChatGPT Action", example_chatgpt_action),
        ("Full Stack Export", example_full_stack_export),
        ("Complete Workflow", example_complete_workflow),
        ("Framework Integration", example_framework_integration),
    ]
    
    results = {}
    
    for name, example_func in examples:
        try:
            result = example_func()
            results[name] = result
            print(f"\n✅ {name} completed successfully")
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")
            results[name] = None
    
    print("\n" + "=" * 70)
    print("TUTORIAL COMPLETE!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. ✓ AgentRegistry provides unified search across sources")
    print("2. ✓ BaseAgentController standardizes agent invocation")
    print("3. ✓ FastAPI export enables HTTP/REST APIs")
    print("4. ✓ MCP support integrates with Claude/OpenAI")
    print("5. ✓ ChatGPT actions for custom GPT integration")
    print("\nNext Steps:")
    print("- Implement GitHub/HuggingFace searchers with real APIs")
    print("- Add authentication to FastAPI exports")
    print("- Create framework-specific controllers (CrewAI, LangChain)")
    print("- Build agent installation/dependency management")
    print("- Add monitoring and logging capabilities")
    
    return results


if __name__ == "__main__":
    results = run_all_examples()
    
    # Save results summary
    summary = {
        "tutorial": "AYN - Agents You Need",
        "examples_run": len(results),
        "examples_successful": sum(1 for v in results.values() if v is not None),
    }
    
    print(f"\n📊 Summary: {summary}")
