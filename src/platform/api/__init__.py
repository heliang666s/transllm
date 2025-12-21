"""API route groupings for platform entrypoints."""

from .anthropic import router as anthropic_router

__all__ = ["anthropic_router"]
