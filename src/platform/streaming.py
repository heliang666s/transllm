"""SSE streaming event publisher that converts provider events to client format."""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator

import httpx

from src.platform.token_utils import _estimate_tokens, _safe_json
from src.transllm.converters.stream_converter import StreamConverter
from src.transllm.core.exceptions import ConversionError
from src.transllm.core.schema import Provider

logger = logging.getLogger(__name__)


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


def _convert_response_payload(
    data: dict[str, Any], from_provider: Provider, to_provider: Provider
) -> dict[str, Any]:
    """Convert response payload between providers."""
    from src.transllm.converters.response_converter import ResponseConverter
    from src.transllm.core.exceptions import ConversionError, UnsupportedProviderError

    try:
        return ResponseConverter.convert(data, from_provider, to_provider)
    except (UnsupportedProviderError, ConversionError) as exc:
        raise
    except Exception as exc:
        raise ConversionError(f"Failed to convert response: {exc}") from exc


async def event_publisher(
    client_provider: Provider,
    target_provider: Provider,
    config: Any,
    upstream_body: dict[str, Any],
    unified_request: Any,
    stream_converter: StreamConverter,
) -> AsyncGenerator[str, None]:
    """SSE event publisher that proxies and converts streaming events.

    Args:
        client_provider: The client/provider format to convert to
        target_provider: The upstream provider format
        config: Upstream configuration
        upstream_body: The request body for upstream
        unified_request: The unified request object
        stream_converter: The stream converter instance

    Yields:
        SSE formatted event strings
    """
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

    def _truncate_text(text: str, limit: int = 2000) -> str:
        """Guardrail to avoid logging extremely large bodies."""
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

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
                        import json

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
                        import json

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
                    except Exception as exc:
                        yield _sse_frame({"detail": str(exc)}, event="error")
                        break
    except httpx.RequestError as exc:
        logger.error("Upstream stream request error: %s", exc)
        yield _sse_frame({"detail": f"Upstream request failed: {exc}"}, event="error")