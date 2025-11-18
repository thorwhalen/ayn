"""Base controller protocol and abstract class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional, Protocol

from ..core.config import ControllerConfig
from ..core.models import AgentMetadata, AgentFramework


class AgentController(Protocol):
    """Standard protocol for agent controllers

    Controllers provide a uniform interface for invoking agents regardless
    of their underlying framework.

    Example:
        >>> class MyController:
        ...     def __init__(self, config):
        ...         self.config = config
        ...     def invoke(self, input_data, **kwargs):
        ...         return f"Processed: {input_data}"
        >>> controller = MyController({'model': 'gpt-4'})
        >>> controller.invoke('test')
        'Processed: test'
    """

    def invoke(self, input_data: Any, **kwargs) -> Any:
        """Invoke the agent synchronously

        Args:
            input_data: Input data for the agent
            **kwargs: Additional parameters

        Returns:
            Agent output
        """
        ...

    async def ainvoke(self, input_data: Any, **kwargs) -> Any:
        """Invoke the agent asynchronously"""
        ...

    def stream(self, input_data: Any, **kwargs) -> Iterable[Any]:
        """Stream agent outputs"""
        ...


class BaseAgentController(ABC):
    """Base class for agent controllers

    Provides common functionality and enforces the controller protocol.

    Example:
        >>> class SimpleController(BaseAgentController):
        ...     def invoke(self, input_data, **kwargs):
        ...         return f"Result: {input_data}"
        >>> controller = SimpleController(ControllerConfig())
        >>> controller.invoke('test')
        'Result: test'
    """

    def __init__(self, config: ControllerConfig):
        self.config = config

    @abstractmethod
    def invoke(self, input_data: Any, **kwargs) -> Any:
        """Invoke the agent - must be implemented by subclasses"""
        pass

    async def ainvoke(self, input_data: Any, **kwargs) -> Any:
        """Default async implementation wraps synchronous invoke"""
        return self.invoke(input_data, **kwargs)

    def stream(self, input_data: Any, **kwargs) -> Iterable[Any]:
        """Default streaming yields single result"""
        yield self.invoke(input_data, **kwargs)

    @classmethod
    def from_metadata(
        cls, metadata: AgentMetadata, config: Optional[ControllerConfig] = None
    ) -> BaseAgentController:
        """Factory method to create controller from metadata"""
        from .generic import GenericController
        from .framework import CrewAIController, LangChainController, AutoGenController

        config = config or ControllerConfig()
        # Route to appropriate framework-specific controller
        if metadata.framework == AgentFramework.CREWAI:
            return CrewAIController(metadata, config)
        elif metadata.framework == AgentFramework.LANGCHAIN:
            return LangChainController(metadata, config)
        elif metadata.framework == AgentFramework.AUTOGEN:
            return AutoGenController(metadata, config)
        else:
            return GenericController(metadata, config)
