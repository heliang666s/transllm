from __future__ import annotations

import json
import math
from typing import Any

from src.transllm.core.schema import ContentBlock

__all__ = [
    "_safe_json",
    "_extract_content_text",
    "_estimate_tokens",
    "_estimate_request_tokens",
]


def _safe_json(obj: Any) -> str:
    """Best-effort JSON serialization for logging/token accounting."""
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


def _extract_content_text(content: Any) -> str:
    """Flatten text-like pieces from a unified content payload for token estimation."""
    parts: list[str] = []

    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, ContentBlock):
                if block.text:
                    parts.append(block.text)
                if block.reasoning and getattr(block.reasoning, "content", None):
                    parts.append(str(block.reasoning.content))
                if block.thinking and getattr(block.thinking, "content", None):
                    parts.append(str(block.thinking.content))
                if block.redacted_thinking and getattr(block.redacted_thinking, "content", None):
                    parts.append(str(block.redacted_thinking.content))
                if block.tool_result and getattr(block.tool_result, "result", None):
                    parts.append(_safe_json(block.tool_result.result))
                if block.image_url and getattr(block.image_url, "url", None):
                    parts.append(block.image_url.url)
            else:
                text_val = ""
                if isinstance(block, dict):
                    text_val = block.get("text") or block.get("reasoning") or ""
                if text_val:
                    parts.append(str(text_val))

    return " ".join(parts)


def _estimate_tokens(text: str, model: str | None = None) -> int:
    """Estimate token count via tiktoken when available, otherwise a simple heuristic."""
    try:
        import tiktoken  # type: ignore
    except ImportError:
        tiktoken = None  # type: ignore

    if tiktoken:
        try:
            encoding = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            pass

    return max(1, math.ceil(len(text) / 4))


def _estimate_request_tokens(unified_request: Any) -> int:
    """Estimate input tokens for a unified request."""
    model_name = getattr(unified_request, "model", None)
    total_text: list[str] = []

    sys_instr = getattr(unified_request, "system_instruction", None)
    if sys_instr:
        total_text.append(str(sys_instr))

    for msg in getattr(unified_request, "messages", []) or []:
        role_val = msg.role.value if hasattr(msg.role, "value") else getattr(msg, "role", "")
        if role_val:
            total_text.append(str(role_val))

        total_text.append(_extract_content_text(getattr(msg, "content", "")))

        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls or []:
                name = getattr(tc, "name", "")
                args = getattr(tc, "arguments", {})
                total_text.append(str(name))
                total_text.append(_safe_json(args))

    combined = " ".join(part for part in total_text if part)
    return _estimate_tokens(combined, model=model_name)
