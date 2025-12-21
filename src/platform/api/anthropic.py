from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

router = APIRouter()


@router.get("/stream/v1/models")
async def list_models_stream() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": "gpt-5-nano",
                "object": "model",
            }
        ],
    }


@router.post("/stream/v1/messages/count_tokens")
async def count_tokens_stream_v1(
    request: Request,
    url: str | None = Query(None, description="Upstream URL override"),
    apikey: str | None = Query(None, description="Upstream API key override"),
    source: str | None = Query(None, description="Upstream provider name override (ignored for local counting)"),
) -> dict[str, Any]:
    """Anthropic 风格路径别名：/stream/v1/messages/count_tokens -> /messages/count_tokens。"""
    from src.platform.app import count_tokens

    return await count_tokens(request, url=url, apikey=apikey, source=source)


@router.post("/stream/v1/messages")
async def messages_stream_v1(
    request: Request,
    url: str | None = Query(None, description="Upstream URL override"),
    apikey: str | None = Query(None, description="Upstream API key override"),
    source: str | None = Query(None, description="Upstream provider name override"),
    provider: str | None = Query(None, description="Client provider override"),
) -> StreamingResponse:
    from src.platform.app import stream

    return await stream(request, url=url, apikey=apikey, source=source, provider=provider)
