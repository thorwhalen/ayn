"""FastAPI export functionality."""

from __future__ import annotations

import json
from typing import Any, Optional

from ..controllers.base import AgentController
from ..core.models import AgentMetadata


def _create_fastapi_app(
    controller: AgentController, metadata: Optional[AgentMetadata] = None
) -> Any:
    """Create a FastAPI app wrapping an agent controller

    Args:
        controller: Agent controller instance
        metadata: Optional agent metadata for docs

    Returns:
        FastAPI app instance

    Example:
        >>> from ayn.controllers import GenericController
        >>> from ayn.core import ControllerConfig, AgentMetadata, AgentFramework
        >>> meta = AgentMetadata(name="test", description="test", framework=AgentFramework.CUSTOM)
        >>> controller = GenericController(meta, ControllerConfig())
        >>> app = _create_fastapi_app(controller, meta)  # doctest: +SKIP
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
    except ImportError:
        raise ImportError("FastAPI is required: pip install fastapi uvicorn")

    app = FastAPI(
        title=metadata.name if metadata else "Agent API",
        description=metadata.description if metadata else "AI Agent API",
        version=metadata.version if metadata else "0.1.0",
    )

    class InvokeRequest(BaseModel):
        input_data: Any
        kwargs: dict = {}

    @app.post("/invoke")
    async def invoke_endpoint(request: InvokeRequest):
        """Invoke the agent"""
        result = await controller.ainvoke(request.input_data, **request.kwargs)
        return JSONResponse({"result": result})

    @app.post("/stream")
    async def stream_endpoint(request: InvokeRequest):
        """Stream agent outputs"""
        from fastapi.responses import StreamingResponse

        async def generate():
            for chunk in controller.stream(request.input_data, **request.kwargs):
                yield json.dumps({"chunk": chunk}) + "\n"

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    @app.get("/health")
    async def health():
        """Health check"""
        return {"status": "healthy"}

    @app.get("/metadata")
    async def get_metadata():
        """Get agent metadata"""
        if metadata:
            return metadata.to_dict()
        return {"message": "No metadata available"}

    return app


def export_as_fastapi(
    controller: AgentController,
    metadata: Optional[AgentMetadata] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """Export agent controller as FastAPI service

    Args:
        controller: Agent controller
        metadata: Optional agent metadata
        host: Host to bind to
        port: Port to bind to

    Example:
        >>> # This would start a server - doctest: +SKIP
        >>> controller = GenericController(...)  # doctest: +SKIP
        >>> export_as_fastapi(controller, port=8080)  # doctest: +SKIP
    """
    app = _create_fastapi_app(controller, metadata)

    try:
        import uvicorn

        uvicorn.run(app, host=host, port=port)
    except ImportError:
        raise ImportError("uvicorn is required: pip install uvicorn")
