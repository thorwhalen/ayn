"""Framework-specific controller implementations."""

from __future__ import annotations

from .generic import GenericController


class CrewAIController(GenericController):
    """Controller for CrewAI agents

    TODO: Implement full CrewAI integration
    See AYN_IMPLEMENTATION_ROADMAP.md for details
    """

    pass


class LangChainController(GenericController):
    """Controller for LangChain agents

    TODO: Implement full LangChain integration
    See AYN_IMPLEMENTATION_ROADMAP.md for details
    """

    pass


class AutoGenController(GenericController):
    """Controller for AutoGen agents

    TODO: Implement full AutoGen integration
    See AYN_IMPLEMENTATION_ROADMAP.md for details
    """

    pass
