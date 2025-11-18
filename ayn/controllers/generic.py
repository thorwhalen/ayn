"""Generic controller for custom agents."""

from __future__ import annotations

from typing import Any

from ..core.config import ControllerConfig
from ..core.models import AgentMetadata
from .base import BaseAgentController


class GenericController(BaseAgentController):
    """Generic controller for custom agents

    Example:
        >>> from ayn.core import AgentMetadata, AgentFramework, ControllerConfig
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
