"""Utility functions for Gemini adapter"""

from __future__ import annotations

import base64
import mimetypes
import re
import uuid
from typing import Any, Dict, List, Optional, Union, Set
from urllib.parse import urlparse

def is_http_url(url: str) -> bool:
    """Check if URL is HTTP/HTTPS

    Args:
        url: URL to check

    Returns:
        True if URL is HTTP/HTTPS
    """
    parsed = urlparse(url)
    return parsed.scheme in ["http", "https"]


def is_base64_data(url: str) -> bool:
    """Check if URL is base64 data URI

    Args:
        url: URL to check

    Returns:
        True if URL is base64 data URI
    """
    return url.startswith("data:")


def detect_media_type(url: str) -> Optional[str]:
    """Detect media type from URL

    Args:
        url: URL or data URI

    Returns:
        Media type string or None
    """
    if is_base64_data(url):
        # Extract from data URI
        match = re.match(r'data:([^;]+)', url)
        if match:
            return match.group(1)

    # Try to detect from URL extension
    parsed = urlparse(url)
    path = parsed.path
    media_type, _ = mimetypes.guess_type(path)
    return media_type


def convert_image_url_to_gemini(
    image_url: str,
    detail: Optional[str] = None
) -> Dict[str, Any]:
    """Convert OpenAI image_url to Gemini format

    Google AI Studio supports:
    - Base64 data URIs
    - HTTP/HTTPS URLs (automatically converted to base64)

    Args:
        image_url: OpenAI image URL or data URI
        detail: Detail level (low, high, auto)

    Returns:
        Gemini image part dictionary
    """
    if is_base64_data(image_url):
        # Base64 data URI → inline_data
        match = re.match(r'data:([^;]+);base64,(.+)', image_url)
        if match:
            media_type = match.group(1)
            data = match.group(2)
            return {
                "inline_data": {
                    "mime_type": media_type,
                    "data": data
                }
            }
    elif is_http_url(image_url):
        # Google AI Studio doesn't support HTTP URLs directly
        # Automatically download and convert to base64 (like litellm does)
        try:
            import httpx
            import base64

            # Download image with 10 second timeout
            response = httpx.get(image_url, timeout=10.0)
            response.raise_for_status()

            # Detect MIME type (prefer Content-Type header)
            content_type = response.headers.get("content-type", "image/jpeg")
            if not content_type.startswith("image/"):
                # Fallback: guess from URL
                content_type = detect_media_type(image_url) or "image/jpeg"

            # Convert to base64
            base64_data = base64.b64encode(response.content).decode("utf-8")

            # Return inline_data format (supported by Google AI Studio)
            return {
                "inline_data": {
                    "mime_type": content_type,
                    "data": base64_data
                }
            }
        except Exception as e:
            raise ValueError(
                f"Failed to fetch and convert HTTP URL to base64: {e}. "
                "Consider using base64 data URI instead."
            )
    else:
        # Invalid format
        raise ValueError(
            "Invalid image URL format. Use base64 data URI or HTTP/HTTPS URL."
        )

    # Default fallback
    raise ValueError(f"Unable to process image URL: {image_url}")


def generate_tool_call_id() -> str:
    """Generate unique tool call ID for Gemini

    Gemini doesn't return tool call IDs in responses, so we need to
    generate UUID-based IDs. For thinking blocks, we encode with
    thoughtSignature.

    Returns:
        UUID string
    """
    return str(uuid.uuid4())


def decode_thought_signature(signature: str) -> Optional[str]:
    """Decode thoughtSignature to extract tool call ID

    Args:
        signature: Base64 encoded signature

    Returns:
        Original tool call ID or None if invalid
    """
    try:
        decoded = base64.b64decode(signature.encode()).decode()
        if decoded.startswith("toolcall:"):
            return decoded.split(":", 1)[1]
    except Exception:
        pass
    return None


def is_candidate_token_count_inclusive(
    prompt_tokens: int,
    candidates_tokens: int,
    total_tokens: int
) -> bool:
    """Check if candidate tokens are included in total token count

    Gemini 3.x models include thinking tokens in the candidate count,
    while Gemini 2.x models may not. This function detects which
    behavior applies to avoid double-counting.

    Args:
        prompt_tokens: Token count for prompt
        candidates_tokens: Token count for candidates
        total_tokens: Total token count

    Returns:
        True if candidate tokens are already included in total
    """
    # If prompt + candidates equals total, they're already counted separately
    if prompt_tokens + candidates_tokens == total_tokens:
        return False
    elif candidates_tokens == total_tokens:
        # If candidates equal total, they include everything
        return True
    else:
        # Default to assuming they're separate (safe approach)
        return False


def merge_duplicate_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge consecutive messages with the same role

    Gemini prefers fewer, more complete messages rather than
    many small messages.

    Args:
        messages: List of message dictionaries

    Returns:
        Merged messages list
    """
    if not messages:
        return []

    merged = []
    current = messages[0].copy()
    current_parts = []

    # Extract parts if present
    if isinstance(current.get("parts"), list):
        current_parts = current["parts"]
        # Remove parts from current to process separately
        current = {k: v for k, v in current.items() if k != "parts"}

    for msg in messages[1:]:
        if msg.get("role") == current.get("role"):
            # Same role - merge parts
            if isinstance(msg.get("parts"), list):
                current_parts.extend(msg["parts"])
        else:
            # Different role - save current and start new
            if current_parts and "parts" not in current:
                current["parts"] = current_parts
            elif current_parts:
                current["parts"].extend(current_parts)
            merged.append(current)

            current = msg.copy()
            current_parts = []
            if isinstance(current.get("parts"), list):
                current_parts = current["parts"]
                current = {k: v for k, v in current.items() if k != "parts"}

    # Don't forget the last message
    if current_parts and "parts" not in current:
        current["parts"] = current_parts
    elif current_parts:
        current["parts"].extend(current_parts)
    merged.append(current)

    return merged


def validate_gemini_request(request: Dict[str, Any]) -> None:
    """Validate Gemini request for common issues

    Args:
        request: Gemini request dictionary

    Raises:
        ValueError: If request is invalid
    """
    # Check for required fields
    if "contents" not in request:
        raise ValueError("Gemini request must contain 'contents' field")

    contents = request["contents"]
    if not isinstance(contents, list) or not contents:
        raise ValueError("Gemini 'contents' must be a non-empty list")

    # Check for empty parts
    for content in contents:
        if "parts" not in content:
            raise ValueError("Each content must have 'parts' field")
        parts = content["parts"]
        if not isinstance(parts, list) or not parts:
            raise ValueError("Parts must be a non-empty list")

        # Check each part has required fields
        for part in parts:
            if not any(key in part for key in ["text", "inline_data", "file_data", "function_call"]):
                raise ValueError("Each part must have at least one of: text, inline_data, file_data, function_call")


def detect_circular_reference(
    obj: Any,
    visited: Optional[Set[int]] = None,
    depth: int = 0,
    max_depth: int = 50
) -> bool:
    """Detect circular references in nested objects

    Args:
        obj: Object to check
        visited: Set of visited object ids
        depth: Current recursion depth
        max_depth: Maximum recursion depth

    Returns:
        True if circular reference detected

    Raises:
        ValueError: If circular reference or max depth exceeded
    """
    if visited is None:
        visited = set()

    if depth > max_depth:
        raise ValueError(f"Max recursion depth ({max_depth}) exceeded")

    obj_id = id(obj)
    if obj_id in visited:
        return True

    if isinstance(obj, (dict, list)):
        visited.add(obj_id)

        try:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if detect_circular_reference(value, visited.copy(), depth + 1, max_depth):
                        return True
            elif isinstance(obj, list):
                for item in obj:
                    if detect_circular_reference(item, visited.copy(), depth + 1, max_depth):
                        return True
        finally:
            visited.discard(obj_id)

    return False


def validate_empty_properties(schema: Dict[str, Any]) -> None:
    """Validate schema for empty properties

    Args:
        schema: Schema to validate

    Raises:
        ValueError: If schema has empty properties
    """
    if not isinstance(schema, dict):
        return

    schema_type = schema.get("type")

    if schema_type == "object":
        properties = schema.get("properties", {})
        if not properties:
            raise ValueError("Object schema cannot have empty properties")

        # Recursively validate properties
        for prop_name, prop_schema in properties.items():
            if not prop_schema:
                raise ValueError(f"Property '{prop_name}' has empty schema")

            # Check for empty required list
            if "required" in prop_schema:
                required = prop_schema["required"]
                if isinstance(required, list) and not required:
                    raise ValueError(f"Property '{prop_name}' has empty required list")

            validate_empty_properties(prop_schema)

    elif schema_type == "array":
        items = schema.get("items")
        if not items:
            # Set default empty object schema
            schema["items"] = {"type": "object"}

        # Recursively validate items
        if items:
            validate_empty_properties(items)

    elif "anyOf" in schema:
        anyof = schema["anyOf"]
        if len(anyof) == 0:
            raise ValueError("anyOf cannot be empty")

        if len(anyof) == 1:
            only_item = anyof[0]
            if isinstance(only_item, dict) and only_item.get("type") == "null":
                raise ValueError("anyOf cannot contain only null type")

        # Recursively validate anyOf items
        for item in anyof:
            validate_empty_properties(item)

    elif "allOf" in schema:
        allof = schema["allOf"]
        if not allof:
            raise ValueError("allOf cannot be empty")

        # Recursively validate allOf items
        for item in allof:
            validate_empty_properties(item)
