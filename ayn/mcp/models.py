"""MCP data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional


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
