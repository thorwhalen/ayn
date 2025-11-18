"""ChatGPT custom action generation."""

from __future__ import annotations

from ..controllers.base import AgentController
from ..core.models import AgentMetadata


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
        >>> from ayn.controllers import GenericController
        >>> from ayn.core import AgentMetadata, AgentFramework, ControllerConfig
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
