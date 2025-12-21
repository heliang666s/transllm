"""Executable entrypoint for the TransLLM streaming proxy."""

from __future__ import annotations

import os

import uvicorn

from src.platform.api import anthropic_router
from src.platform.app import app


def run() -> None:
    """Start the FastAPI app via uvicorn."""
    # Include anthropic router routes
    app.include_router(anthropic_router)

    host = os.getenv("TRANSLLM_HOST", "0.0.0.0")
    port = int(os.getenv("TRANSLLM_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
