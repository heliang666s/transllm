"""FastAPI SSE bridge that normalizes provider-specific requests to IR, forwards
them to an upstream provider, and streams converted events back to the caller.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field
from urllib.parse import urlsplit, urlunsplit

import src.transllm  # noqa: F401 - ensures adapters register with ProviderRegistry
from src.platform.token_utils import (
    _extract_content_text,
    _safe_json,
    _estimate_tokens,
    _estimate_request_tokens,
)
from src.transllm.converters.response_converter import ResponseConverter
from src.transllm.converters.stream_converter import StreamConverter
from src.transllm.core.exceptions import ConversionError, UnsupportedProviderError
from src.transllm.core.schema import ContentBlock, Provider
from src.transllm.utils.provider_registry import ProviderRegistry

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


class UpstreamConfig(BaseModel):
    """Upstream endpoint configuration."""

    url: AnyHttpUrl | str
    apikey: str = Field(..., alias="apikey")
    source: Provider
    timeout_seconds: float | None = Field(
        default=60.0, description="Optional timeout applied to the upstream request"
    )

    model_config = ConfigDict(populate_by_name=True)


class StreamEnvelope(BaseModel):
    """Wrapper allowing optional config/hints alongside a raw provider request."""

    payload: dict[str, Any] | None = None
    provider: Provider | None = None
    config: UpstreamConfig | None = None
    stream: bool | None = True

    model_config = ConfigDict(extra="allow")


app = FastAPI(title="TransLLM Streaming Proxy")
CONFIG_PATH = Path(__file__).with_name("config.yaml")

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _truncate_text(text: str, limit: int = 2000) -> str:
    """Guardrail to avoid logging extremely large bodies."""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _augment_message_start(
    message: dict[str, Any], model_name: str | None
) -> dict[str, Any]:
    """Ensure message_start carries model/usage fields."""
    message = dict(message)
    if model_name and "model" not in message:
        message["model"] = model_name
    message.setdefault("stop_reason", None)
    message.setdefault("stop_sequence", None)
    usage = message.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    usage.setdefault("input_tokens", 0)
    usage.setdefault("output_tokens", 0)
    message["usage"] = usage
    return message


def _sse_frame(data: Any, event: str | None = None) -> str:
    """Format a strict SSE frame with optional event name."""
    data_str = data if isinstance(data, str) else _safe_json(data)
    lines = []
    if event:
        lines.append(f"event: {event}")
    if data_str == "":
        lines.append("data:")
    else:
        for line in str(data_str).splitlines():
            lines.append(f"data: {line}")
    return "\n".join(lines) + "\n\n"


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """Return a minimal model list for compatibility."""
    return {
        "object": "list",
        "data": [
            {
                "id": "gpt-5-nano",
                "object": "model",
            }
        ],
    }


@app.get("/stream/v1/models")
async def list_models_stream() -> dict[str, Any]:
    """Alias for model listing with Anthropic-style prefix."""
    return await list_models()


def _normalize_provider(raw_provider: Any) -> Provider | None:
    if isinstance(raw_provider, Provider):
        return raw_provider
    if isinstance(raw_provider, str):
        try:
            return Provider(raw_provider.lower())
        except ValueError:
            return None
    return None


def _detect_provider(request_body: dict[str, Any]) -> Provider | None:
    """Lightweight detection based on known request shapes."""

    if not isinstance(request_body, dict):
        raise HTTPException(status_code=400, detail="request_body must be a dict")

    keys = set(request_body.keys())
    model = str(request_body.get("model", "") or "").strip().lower()

    # gemini
    if (
        "contents" in keys
        or "systemInstruction" in keys
        or "generationConfig" in keys
        or "safetySettings" in keys
    ):
        return Provider.gemini
    if "gemini" in model or "models/gemini" in model:
        return Provider.gemini

    # anthropic
    if "anthropic_version" in keys or "stop_sequences" in keys:
        return Provider.anthropic
    if "claude" in model or "anthropic" in model:
        return Provider.anthropic

    if "system" in keys and "messages" in keys:
        openai_knobs = {
            "frequency_penalty",
            "presence_penalty",
            "logit_bias",
            "response_format",
            "stream_options",
            "seed",
            "n",
            "logprobs",
            "top_logprobs",
        }
        if not (keys & openai_knobs):
            return Provider.anthropic

    # openai
    if "input" in keys:
        return Provider.openai
    if "messages" in keys:
        return Provider.openai
    if "gpt" in model or model.startswith(("o1", "o3", "o4", "o5")) or "openai" in model:
        return Provider.openai

    # -------- Fail --------
    sample = {k: request_body.get(k) for k in ("model", "messages", "system", "contents") if k in request_body}
    raise HTTPException(
        status_code=400,
        detail=(
            "Unable to detect provider. "
            f"keys={sorted(keys)} model={model!r} sample={sample}"
        ),
    )


def _build_config(
    body_config: dict[str, Any] | None,
    url_param: str | None,
    apikey_param: str | None,
    source_param: str | None,
) -> UpstreamConfig:
    merged: dict[str, Any] = {}

    def _load_default_config() -> dict[str, Any]:
        if not CONFIG_PATH.exists():
            return {}
        if yaml is None:
            raise RuntimeError("PyYAML is required to read config.yaml")
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError("config.yaml must contain a mapping")
        return loaded

    try:
        merged.update(_load_default_config())
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to load default config: {exc}"
        ) from exc
    if body_config:
        merged.update(body_config)
    if url_param:
        merged.setdefault("url", url_param)
    if apikey_param:
        merged.setdefault("apikey", apikey_param)
    if source_param:
        merged.setdefault("source", source_param)

    required_fields = {"url", "apikey", "source"}
    if not required_fields.issubset(merged):
        missing = ", ".join(sorted(required_fields - set(merged.keys())))
        raise HTTPException(
            status_code=400, detail=f"Missing upstream configuration: {missing}"
        )

    return UpstreamConfig(**merged)


def _resolve_messages_url(base_url: AnyHttpUrl | str, *, count_tokens: bool = False) -> str:
    """Ensure the upstream URL points at the Messages or count_tokens endpoint with /stream/v1."""
    base = str(base_url).rstrip("/")
    parsed = urlsplit(base)
    path_parts = [part for part in parsed.path.split("/") if part]

    # Drop any existing endpoint/version markers to avoid duplication
    while path_parts and path_parts[-1] in {"count_tokens", "messages"}:
        path_parts.pop()
    while path_parts and path_parts[-1] in {"v1", "stream"}:
        path_parts.pop()

    final_parts = path_parts + ["stream", "v1", "messages"]
    if count_tokens:
        final_parts.append("count_tokens")

    final_path = "/" + "/".join(final_parts)
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            final_path,
            parsed.query,
            parsed.fragment,
        )
    )


def _convert_response_payload(
    data: dict[str, Any], from_provider: Provider, to_provider: Provider
) -> dict[str, Any]:
    try:
        return ResponseConverter.convert(data, from_provider, to_provider)
    except (UnsupportedProviderError, ConversionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net
        raise HTTPException(
            status_code=500, detail=f"Failed to convert response: {exc}"
        ) from exc


@app.post("/messages")
async def messages(
    request: Request,
    url: str | None = Query(None, description="Upstream URL override"),
    apikey: str | None = Query(None, description="Upstream API key override"),
    source: str | None = Query(None, description="Upstream provider name override"),
    provider: str | None = Query(
        None, description="Client provider override (e.g., anthropic/openai/gemini)"
    ),
) -> dict[str, Any]:
    """Forward a single Messages request to the upstream provider (non-streaming)."""
    raw_body = await request.json()

    envelope = StreamEnvelope.model_validate(raw_body)
    payload = envelope.payload or {}
    if not payload and isinstance(raw_body, dict):
        payload = {
            k: v
            for k, v in raw_body.items()
            if k not in {"config", "provider", "stream"}
        }

    client_provider = _normalize_provider(provider) or _normalize_provider(
        envelope.provider
    ) or _detect_provider(payload)
    if client_provider is None:
        raise HTTPException(
            status_code=400,
            detail="Unable to detect provider; pass provider explicitly or use OpenAI/Anthropic/Gemini shapes.",
        )

    target_provider = config.source
    if not ProviderRegistry.is_supported(client_provider):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported client provider: {client_provider.value}",
        )
    if not ProviderRegistry.is_supported(target_provider):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported upstream provider: {target_provider.value}",
        )

    client_adapter = ProviderRegistry.get_adapter(client_provider)
    upstream_adapter = ProviderRegistry.get_adapter(target_provider)

    try:
        logger.info("Incoming payload (%s): %s", client_provider.value, payload)
        unified_request = client_adapter.to_unified_request(payload)
        logger.info("Unified request: %s", unified_request.model_dump())
        upstream_body = upstream_adapter.from_unified_request(unified_request)
        if (
            target_provider == Provider.openai
            and "max_tokens" in upstream_body
            and "max_completion_tokens" not in upstream_body
        ):
            upstream_body["max_completion_tokens"] = upstream_body.pop("max_tokens")
        if target_provider == Provider.openai and "system_instruction" in upstream_body:
            upstream_body.pop("system_instruction", None)
        logger.info(
            "Upstream request body (%s -> %s): %s",
            client_provider.value,
            target_provider.value,
            upstream_body,
        )
    except (ConversionError, UnsupportedProviderError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net
        raise HTTPException(
            status_code=400,
            detail=f"Failed to normalize request: {exc}",
        ) from exc

    headers = {
        "Authorization": f"Bearer {config.apikey}",
        "x-api-key": config.apikey,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    upstream_url = _resolve_messages_url(config.url, count_tokens=False)
    timeout = config.timeout_seconds if config.timeout_seconds else None

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                upstream_url, headers=headers, json=upstream_body
            )
            response.raise_for_status()
            upstream_payload = response.json()
    except httpx.HTTPStatusError as exc:
        err_text = _truncate_text(exc.response.text or exc.response.reason_phrase or "")
        logger.error(
            "Upstream /messages error %s: %s",
            exc.response.status_code,
            err_text,
        )
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=exc.response.text or exc.response.reason_phrase,
        ) from exc
    except httpx.RequestError as exc:
        logger.error("Upstream /messages request error: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Upstream request failed: {exc}",
        ) from exc

    logger.info(
        "Upstream response payload (%s): %s",
        target_provider.value,
        upstream_payload,
    )
    converted = _convert_response_payload(
        upstream_payload, target_provider, client_provider
    )
    logger.info(
        "Converted response payload (%s): %s",
        client_provider.value,
        converted,
    )
    return converted


@app.post("/stream/api/event_logging/batch")
async def event_logging_batch() -> dict[str, str]:
    """Stub to avoid noisy 404s when clients send telemetry batches."""
    return {"status": "ok"}


@app.post("/stream/v1/messages/count_tokens")
async def count_tokens_stream_v1(
    request: Request,
    url: str | None = Query(None, description="Upstream URL override"),
    apikey: str | None = Query(None, description="Upstream API key override"),
    source: str | None = Query(
        None, description="Upstream provider name override (ignored for local counting)"
    ),
) -> dict[str, Any]:
    """Alias for /messages/count_tokens using Anthropic-style pathing."""
    return await count_tokens(request, url=url, apikey=apikey, source=source)


@app.post("/stream/v1/messages")
async def messages_stream_v1(
    request: Request,
    url: str | None = Query(None, description="Upstream URL override"),
    apikey: str | None = Query(None, description="Upstream API key override"),
    source: str | None = Query(None, description="Upstream provider name override"),
    provider: str | None = Query(
        None, description="Client provider override (e.g., anthropic/openai/gemini)"
    ),
) -> StreamingResponse:
    """Alias for streaming endpoint using Anthropic-style pathing."""
    return await stream(
        request, url=url, apikey=apikey, source=source, provider=provider
    )


@app.post("/messages/count_tokens")
async def count_tokens(
    request: Request,
    url: str | None = Query(None, description="Upstream URL override"),
    apikey: str | None = Query(None, description="Upstream API key override"),
    source: str | None = Query(
        None, description="Upstream provider name override (ignored for local counting)"
    ),
) -> dict[str, Any]:
    """Estimate tokens locally after normalizing the request (no upstream call)."""
    raw_body = await request.json()

    envelope = StreamEnvelope.model_validate(raw_body)
    payload = envelope.payload or {}
    if not payload and isinstance(raw_body, dict):
        payload = {
            k: v
            for k, v in raw_body.items()
            if k not in {"config", "provider", "stream"}
        }

    config = _build_config(
        envelope.config.model_dump(by_alias=True) if envelope.config else None,
        url,
        apikey,
        source,
    )

    client_provider = (
        _normalize_provider(envelope.provider) or _detect_provider(payload)
    )
    if client_provider is None:
        raise HTTPException(
            status_code=400,
            detail="Unable to detect provider; pass provider explicitly or use OpenAI/Anthropic/Gemini shapes.",
        )

    if not ProviderRegistry.is_supported(client_provider):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported client provider: {client_provider.value}",
        )

    client_adapter = ProviderRegistry.get_adapter(client_provider)

    try:
        unified_request = client_adapter.to_unified_request(payload)
        tokens = _estimate_request_tokens(unified_request)
        logger.info(
            "Local token count (%s): %s", client_provider.value, tokens
        )
    except (ConversionError, UnsupportedProviderError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net
        raise HTTPException(
            status_code=400,
            detail=f"Failed to normalize request: {exc}",
        ) from exc
    return {
        "input_tokens": tokens,
        "output_tokens": 0,
        "total_tokens": tokens,
    }


@app.post("/stream")
async def stream(
    request: Request,
    url: str | None = Query(None, description="Upstream URL override"),
    apikey: str | None = Query(None, description="Upstream API key override"),
    source: str | None = Query(None, description="Upstream provider name override"),
    provider: str | None = Query(
        None, description="Client provider override (e.g., anthropic/openai/gemini)"
    ),
) -> StreamingResponse:
    """Accept provider-specific streaming requests, proxy upstream, and emit SSE."""
    raw_body = await request.json()

    # Parse optional wrapper while allowing raw provider bodies.
    envelope = StreamEnvelope.model_validate(raw_body)
    payload = envelope.payload or {}
    if not payload and isinstance(raw_body, dict):
        payload = {
            k: v
            for k, v in raw_body.items()
            if k not in {"config", "provider", "stream"}
        }

    config = _build_config(
        envelope.config.model_dump(by_alias=True) if envelope.config else None,
        url,
        apikey,
        source,
    )

    client_provider = _normalize_provider(provider) or _normalize_provider(
        envelope.provider
    ) or _detect_provider(payload)
    if client_provider is None:
        raise HTTPException(
            status_code=400,
            detail="Unable to detect provider; pass provider explicitly or use OpenAI/Anthropic/Gemini shapes.",
        )

    target_provider = config.source
    if not ProviderRegistry.is_supported(client_provider):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported client provider: {client_provider.value}",
        )
    if not ProviderRegistry.is_supported(target_provider):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported upstream provider: {target_provider.value}",
        )

    client_adapter = ProviderRegistry.get_adapter(client_provider)
    upstream_adapter = ProviderRegistry.get_adapter(target_provider)

    try:
        logger.info("Incoming payload (%s): %s", client_provider.value, payload)
        unified_request = client_adapter.to_unified_request(payload)
        logger.info("Unified request: %s", unified_request.model_dump())
        upstream_body = upstream_adapter.from_unified_request(unified_request)

        if envelope.stream is not False:
            upstream_body.setdefault("stream", True)
        # OpenAI newer models expect max_completion_tokens instead of max_tokens.
        if (
            target_provider == Provider.openai
            and "max_tokens" in upstream_body
            and "max_completion_tokens" not in upstream_body
        ):
            upstream_body["max_completion_tokens"] = upstream_body.pop("max_tokens")
        logger.info(
            "Upstream request body (%s -> %s): %s",
            client_provider.value,
            target_provider.value,
            upstream_body,
        )
    except (ConversionError, UnsupportedProviderError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net
        raise HTTPException(
            status_code=400,
            detail=f"Failed to normalize request: {exc}",
        ) from exc

    async def event_publisher() -> AsyncGenerator[str, None]:
        stream_converter = StreamConverter()
        headers = {
            "Authorization": f"Bearer {config.apikey}",
            "x-api-key": config.apikey,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        timeout = config.timeout_seconds if config.timeout_seconds else None

        seen_message_start = False
        seen_content_start = False
        seen_content_stop = False
        seen_message_delta = False
        seen_message_stop = False
        output_text_parts: list[str] = []

        def _build_message_delta() -> str | None:
            nonlocal seen_message_delta
            if seen_message_delta:
                return None
            output_text = "".join(output_text_parts)
            output_tokens = _estimate_tokens(output_text, model=unified_request.model)
            seen_message_delta = True
            return _sse_frame(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": output_tokens},
                },
                event="message_delta",
            )

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", str(config.url), headers=headers, json=upstream_body
                ) as response:
                    if response.status_code >= 400:
                        error_body = await response.aread()
                        logger.error(
                            "Upstream stream error %s: %s",
                            response.status_code,
                            _truncate_text(error_body.decode() or response.reason_phrase or ""),
                        )
                        yield _sse_frame(
                            {
                                "status": response.status_code,
                                "detail": error_body.decode() or response.reason_phrase,
                            },
                            event="error",
                        )
                        return

                    content_type = response.headers.get("content-type", "")
                    if "text/event-stream" not in content_type:
                        raw_bytes = await response.aread()
                        try:
                            upstream_payload = json.loads(raw_bytes)
                        except json.JSONDecodeError:
                            upstream_payload = {"raw": raw_bytes.decode()}
                        logger.info(
                            "Upstream response payload (%s): %s",
                            target_provider.value,
                            upstream_payload,
                        )
                        converted = _convert_response_payload(
                            upstream_payload, target_provider, client_provider
                        )
                        logger.info(
                            "Converted response payload (%s): %s",
                            client_provider.value,
                            converted,
                        )
                        yield _sse_frame(converted)
                        if client_provider != Provider.anthropic:
                            yield _sse_frame("[DONE]", event="end")
                        return

                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        event_payload = line.strip()
                        if event_payload.startswith("data:"):
                            event_payload = event_payload.split("data:", 1)[1].strip()

                        if event_payload in {"", "[DONE]", "[done]"}:
                            if client_provider == Provider.anthropic:
                                # Ensure stop events are emitted if we synthesized starts
                                if seen_content_start and not seen_content_stop:
                                    yield _sse_frame(
                                        {"type": "content_block_stop", "index": 0},
                                        event="content_block_stop",
                                    )
                                    seen_content_stop = True
                                if not seen_message_delta:
                                    delta_frame = _build_message_delta()
                                    if delta_frame:
                                        yield delta_frame
                                if not seen_message_stop:
                                    yield _sse_frame(
                                        {"type": "message_stop"}, event="message_stop"
                                    )
                                    seen_message_stop = True
                            else:
                                yield _sse_frame("[DONE]", event="end")
                            break

                        try:
                            upstream_event = json.loads(event_payload)
                        except json.JSONDecodeError:
                            # Skip malformed lines to keep the stream alive
                            continue
                        logger.info(
                            "Upstream stream event (%s): %s",
                            target_provider.value,
                            upstream_event,
                        )

                        try:
                            unified_event = stream_converter.to_unified_event(
                                upstream_event, target_provider
                            )
                            client_event = stream_converter.from_unified_event(
                                unified_event, client_provider
                            )
                            logger.info("Unified stream event: %s", unified_event)
                            logger.info(
                                "Converted stream event (%s): %s",
                                client_provider.value,
                                client_event,
                            )
                            event_name = None
                            event_type = None
                            if client_provider == Provider.anthropic and isinstance(
                                client_event, dict
                            ):
                                event_type = client_event.get("type")
                                event_name = event_type

                                # Track seen starts/stops
                                if event_type == "message_start":
                                    # Ensure usage/model fields are present
                                    msg_obj = client_event.get("message", {})
                                    client_event["message"] = _augment_message_start(
                                        msg_obj, unified_request.model
                                    )
                                    seen_message_start = True
                                if event_type == "content_block_start":
                                    seen_content_start = True
                                if event_type == "content_block_stop":
                                    seen_content_stop = True
                                if event_type == "message_delta":
                                    seen_message_delta = True
                                if event_type == "message_stop":
                                    seen_message_stop = True

                                # Synthesize starts if missing
                                if not seen_message_start and event_type not in {
                                    "message_start",
                                    "stream_end",
                                }:
                                    synthetic_msg = _augment_message_start(
                                        {
                                            "id": "msg_local",
                                            "type": "message",
                                            "role": "assistant",
                                            "content": [],
                                        },
                                        unified_request.model,
                                    )
                                    yield _sse_frame(
                                        {
                                            "type": "message_start",
                                            "message": synthetic_msg,
                                        },
                                        event="message_start",
                                    )
                                    seen_message_start = True

                                if not seen_content_start and event_type in {
                                    "content_block_delta",
                                    "content_delta",
                                    "content_finish",
                                    "tool_call_delta",
                                }:
                                    yield _sse_frame(
                                        {
                                            "type": "content_block_start",
                                            "index": 0,
                                            "content_block": {"type": "text", "text": ""},
                                        },
                                        event="content_block_start",
                                    )
                                    seen_content_start = True

                                # Handle stream_end by emitting stops instead
                                if event_type == "stream_end":
                                    if seen_content_start and not seen_content_stop:
                                        yield _sse_frame(
                                            {"type": "content_block_stop", "index": 0},
                                            event="content_block_stop",
                                        )
                                        seen_content_stop = True
                                    if not seen_message_delta:
                                        delta_frame = _build_message_delta()
                                        if delta_frame:
                                            yield delta_frame
                                    if not seen_message_stop:
                                        yield _sse_frame(
                                            {"type": "message_stop"},
                                            event="message_stop",
                                        )
                                        seen_message_stop = True
                                    continue

                                if event_type == "content_block_delta":
                                    delta = client_event.get("delta", {})
                                    text_val = ""
                                    has_text = False
                                    has_partial_json = False
                                    if isinstance(delta, dict):
                                        if "text" in delta:
                                            has_text = True
                                            text_val = delta.get("text") or ""
                                        has_partial_json = "partial_json" in delta
                                    if text_val and not seen_content_stop:
                                        output_text_parts.append(text_val)
                                    if seen_content_stop or (
                                        has_text and text_val == "" and not has_partial_json
                                    ):
                                        continue
                                if event_type == "message_stop" and not seen_message_delta:
                                    delta_frame = _build_message_delta()
                                    if delta_frame:
                                        yield delta_frame

                            yield _sse_frame(client_event, event=event_name)
                        except Exception as exc:  # pragma: no cover - runtime protection
                            yield _sse_frame({"detail": str(exc)}, event="error")
                            break
        except httpx.RequestError as exc:
            logger.error("Upstream stream request error: %s", exc)
            yield _sse_frame({"detail": f"Upstream request failed: {exc}"}, event="error")

    sse_headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    #
    # return StreamingResponse(
    #     event_publisher(),
    #     media_type="text/event-stream",
    #     headers=sse_headers,
    # )

    if client_provider is Provider.anthropic:
        return StreamingResponse(
            event_publisher(),
            media_type="text/event-stream",
            headers=sse_headers,
        )
    else:
        pass