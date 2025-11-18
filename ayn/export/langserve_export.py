"""LangServe export functionality."""

from __future__ import annotations

from typing import Any, Optional

from ..controllers.base import AgentController
from ..core.models import AgentMetadata


def export_as_langserve(
    controller: AgentController, metadata: Optional[AgentMetadata] = None
) -> Any:
    """Export agent as LangServe compatible service

    Args:
        controller: Agent controller
        metadata: Optional agent metadata

    Returns:
        LangServe app

    Example:
        >>> # Requires langserve - doctest: +SKIP
        >>> controller = GenericController(...)  # doctest: +SKIP
        >>> app = export_as_langserve(controller)  # doctest: +SKIP
    """
    try:
        from langserve import add_routes
        from fastapi import FastAPI
    except ImportError:
        raise ImportError("LangServe is required: pip install langserve")

    app = FastAPI()

    # Create a LangChain-compatible runnable wrapper
    class RunnableController:
        def __init__(self, controller):
            self.controller = controller

        def invoke(self, input_data, config=None):
            return self.controller.invoke(input_data)

        async def ainvoke(self, input_data, config=None):
            return await self.controller.ainvoke(input_data)

        def stream(self, input_data, config=None):
            return self.controller.stream(input_data)

    runnable = RunnableController(controller)
    add_routes(app, runnable, path="/agent")

    return app
