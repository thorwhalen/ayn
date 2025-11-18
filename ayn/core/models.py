"""Core data models for agents."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List


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
