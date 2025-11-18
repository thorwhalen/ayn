"""Basic usage example for AYN."""

from ayn import (
    AgentRegistry,
    AgentMetadata,
    AgentFramework,
    BaseAgentController,
    ControllerConfig,
)


def main():
    print("AYN - Agents You Need - Basic Usage Example")
    print("=" * 60)

    # 1. Create a registry
    print("\n1. Creating agent registry...")
    registry = AgentRegistry(searchers=[])  # Empty for local-only demo

    # 2. Add a custom agent
    print("\n2. Registering a custom agent...")
    metadata = AgentMetadata(
        name="data-prep-agent",
        description="Prepares data for visualization",
        framework=AgentFramework.CUSTOM,
        tags=["data", "preparation", "visualization"],
        capabilities=["pandas", "numpy"],
    )
    registry["data-prep-agent"] = metadata
    print(f"   Registered: {metadata.name}")

    # 3. Search for agents
    print("\n3. Searching for 'data' agents...")
    results = registry.search("data")
    print(f"   Found {len(results)} agent(s):")
    for agent in results:
        print(f"   - {agent.name}: {agent.description}")

    # 4. Create controller
    print("\n4. Creating controller from metadata...")
    config = ControllerConfig(model="gpt-4", temperature=0.7)
    controller = BaseAgentController.from_metadata(metadata, config)
    print(f"   Controller type: {controller.__class__.__name__}")

    # 5. Invoke the agent
    print("\n5. Invoking the agent...")
    input_data = {"data": [1, 2, 3, 4, 5], "operation": "analyze"}
    result = controller.invoke(input_data)
    print(f"   Result: {result}")

    print("\n✓ Basic usage example completed!")


if __name__ == "__main__":
    main()
