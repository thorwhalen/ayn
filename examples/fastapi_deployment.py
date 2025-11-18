"""Example of deploying an agent as a FastAPI service."""

from ayn import (
    AgentMetadata,
    AgentFramework,
    GenericController,
    ControllerConfig,
    export_agent_full_stack,
)


def main():
    print("AYN - FastAPI Deployment Example")
    print("=" * 60)

    # 1. Create agent metadata
    metadata = AgentMetadata(
        name="sentiment-analyzer",
        description="Analyzes sentiment of text",
        framework=AgentFramework.CUSTOM,
        tags=["nlp", "sentiment", "analysis"],
        version="1.0.0",
    )

    # 2. Create controller with custom logic
    class SentimentController(GenericController):
        def invoke(self, input_data, **kwargs):
            text = input_data.get("text", "")
            # Simplified sentiment analysis (replace with real implementation)
            sentiment = "positive" if "good" in text.lower() else "negative"
            return {
                "text": text,
                "sentiment": sentiment,
                "confidence": 0.85,
                "agent": self.metadata.name,
            }

    controller = SentimentController(metadata, ControllerConfig())

    # 3. Export with all integrations
    print("\n1. Exporting agent with all integrations...")
    artifacts = export_agent_full_stack(
        controller,
        metadata,
        api_host="0.0.0.0",
        api_port=8000,
        export_fastapi=True,
        export_mcp=True,
        export_chatgpt=True,
    )

    print("\n2. Generated artifacts:")
    for key in artifacts.keys():
        print(f"   - {key}")

    # 4. Display configurations
    if "chatgpt_action" in artifacts:
        print("\n3. ChatGPT Action OpenAPI Info:")
        print(f"   Title: {artifacts['chatgpt_action']['schema']['info']['title']}")
        print(f"   Version: {artifacts['chatgpt_action']['schema']['info']['version']}")

    if "claude_config" in artifacts:
        print("\n4. Claude MCP Configuration:")
        import json

        print(json.dumps(artifacts["claude_config"], indent=2))

    print("\n✓ To run the FastAPI server, uncomment the line below:")
    print("#   uvicorn run:app --host 0.0.0.0 --port 8000")
    print("\n   Then visit: http://localhost:8000/docs")


if __name__ == "__main__":
    main()
