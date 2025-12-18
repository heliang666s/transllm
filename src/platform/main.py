"""Executable entrypoint for the TransLLM streaming proxy."""

from __future__ import annotations

import os

import uvicorn


def run() -> None:
    """Start the FastAPI app via uvicorn."""
    host = os.getenv("TRANSLLM_HOST", "0.0.0.0")
    port = int(os.getenv("TRANSLLM_PORT", "8000"))
    uvicorn.run("src.platform.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
