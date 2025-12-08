"""Response handling for Gemini adapter"""

from __future__ import annotations

import json
from typing import Any, Dict, Generator, List, Optional

from .utils import (
    decode_thought_signature,
    generate_tool_call_id,
    is_candidate_token_count_inclusive,
)


class GeminiResponseHandler:
    """Handle Gemini response format conversion to OpenAI format"""

    def transform_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Gemini response to OpenAI format

        Args:
            response: Gemini response dictionary

        Returns:
            OpenAI-style response dictionary
        """
        # Extract candidates
        candidates = response.get("candidates", [])

        if not candidates:
            # Handle empty response
            return self._create_empty_response(response)

        # Take first candidate
        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        # Build message
        message = self._extract_message_from_parts(parts)

        # Extract usage statistics
        usage = self._extract_usage(response, candidate)

        # Build response
        result = {
            "id": response.get("id", f"gemini-{hash(str(response))}"),
            "object": "chat.completion",
            "created": response.get("createTime", 0),
            "model": response.get("model", "unknown"),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": candidate.get("finishReason", "stop"),
                }
            ],
            "usage": usage,
        }

        # Add system fingerprint if present
        if "systemInstruction" in response:
            result["system_fingerprint"] = "gemini"

        return result

    def _create_empty_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Create empty response structure

        Args:
            response: Original Gemini response

        Returns:
            Empty OpenAI-style response
        """
        return {
            "id": response.get("id", f"gemini-{hash(str(response))}"),
            "object": "chat.completion",
            "created": response.get("createTime", 0),
            "model": response.get("model", "unknown"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    def _extract_message_from_parts(
        self, parts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract OpenAI message from Gemini parts

        Args:
            parts: Gemini content parts

        Returns:
            OpenAI message dictionary
        """
        content_parts = []
        tool_calls = []

        for part in parts:
            if "text" in part:
                content_parts.append({"type": "text", "text": part["text"]})

            elif "function_call" in part:
                func_call = part["function_call"]
                tool_calls.append(
                    {
                        "id": generate_tool_call_id(),
                        "type": "function",
                        "function": {
                            "name": func_call["name"],
                            "arguments": func_call.get("args", {}),
                        },
                    }
                )

            elif "thinking" in part:
                # Handle thinking blocks
                thinking_content = part["thinking"]

                # Try to decode thoughtSignature
                thought_signature = part.get("thoughtSignature")
                if thought_signature:
                    tool_call_id = decode_thought_signature(thought_signature)
                    if tool_call_id:
                        # This thinking block belongs to a tool call
                        # We'll add it as reasoning content
                        pass

                # Add thinking as part of content (for transparency)
                content_parts.append(
                    {
                        "type": "text",
                        "text": f"<thinking>{thinking_content}</thinking>",
                        "thinking": True,
                    }
                )

        # Build message
        message: Dict[str, Any] = {"role": "assistant"}

        if (
            content_parts
            and len(content_parts) == 1
            and content_parts[0]["type"] == "text"
        ):
            # Simple text-only message
            message["content"] = content_parts[0]["text"]
        elif content_parts:
            # Multimodal or multi-part message
            message["content"] = content_parts

        if tool_calls:
            message["tool_calls"] = tool_calls

        return message

    def _extract_tool_calls(self, parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract tool calls from Gemini parts

        Args:
            parts: Gemini content parts

        Returns:
            List of tool calls
        """
        tool_calls = []

        for part in parts:
            if "function_call" in part:
                func_call = part["function_call"]
                tool_calls.append(
                    {
                        "id": generate_tool_call_id(),
                        "type": "function",
                        "function": {
                            "name": func_call["name"],
                            "arguments": json.dumps(func_call.get("args", {})),
                        },
                    }
                )

        return tool_calls

    def _extract_usage(
        self, response: Dict[str, Any], candidate: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract usage statistics from Gemini response

        Args:
            response: Full Gemini response
            candidate: Selected candidate (optional, for backward compatibility)

        Returns:
            OpenAI usage statistics
        """
        # Extract from usageMetadata if available
        usage_metadata = response.get("usageMetadata", {})

        prompt_tokens = usage_metadata.get("promptTokenCount", 0)
        candidates_tokens = usage_metadata.get("candidatesTokenCount", 0)
        total_tokens = usage_metadata.get("totalTokenCount", 0)

        # Check for reasoning tokens
        reasoning_tokens = usage_metadata.get("thoughtsTokenCount", 0)

        # Detect if candidate tokens are inclusive
        inclusive = is_candidate_token_count_inclusive(
            prompt_tokens, candidates_tokens, total_tokens
        )

        if inclusive:
            # Candidate tokens include everything (Gemini 3.x behavior)
            completion_tokens = total_tokens
            if reasoning_tokens > 0:
                # Separate reasoning from completion
                completion_tokens = candidates_tokens - reasoning_tokens
        else:
            # Tokens counted separately (Gemini 2.x behavior)
            completion_tokens = candidates_tokens

        # Add reasoning tokens if not included
        if not inclusive and reasoning_tokens > 0:
            completion_tokens += reasoning_tokens

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "reasoning_tokens": reasoning_tokens if reasoning_tokens > 0 else None,
        }

    def transform_streaming_response(
        self, stream: Generator[Dict[str, Any], None, None]
    ) -> Generator[Dict[str, Any], None, None]:
        """Transform Gemini streaming response to OpenAI format

        Args:
            stream: Generator of Gemini streaming chunks

        Yields:
            OpenAI-style streaming chunks
        """
        for chunk in stream:
            chunk_type = (
                chunk.get("chunk", {})
                .get("content", {})
                .get("parts", [{}])[0]
                .get("type", "text")
            )

            if chunk_type == "thinking":
                # Handle thinking chunks
                yield self._transform_thinking_chunk(chunk)
            else:
                # Handle regular text chunks
                yield self._transform_text_chunk(chunk)

    def _transform_text_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Gemini text chunk to OpenAI format

        Args:
            chunk: Gemini chunk

        Returns:
            OpenAI chunk
        """
        # Extract delta content
        content = chunk.get("chunk", {}).get("content", {})
        parts = content.get("parts", [])

        delta_text = ""
        for part in parts:
            if "text" in part:
                delta_text += part["text"]

        # Build chunk
        return {
            "id": chunk.get("id", ""),
            "object": "chat.completion.chunk",
            "created": chunk.get("createTime", 0),
            "model": chunk.get("model", "unknown"),
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": delta_text if delta_text else None,
                    },
                    "finish_reason": None,
                }
            ],
        }

    def _transform_thinking_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Gemini thinking chunk to OpenAI format

        Args:
            chunk: Gemini chunk with thinking

        Returns:
            OpenAI chunk with reasoning content
        """
        # Extract thinking content
        content = chunk.get("chunk", {}).get("content", {})
        parts = content.get("parts", [])

        thinking_content = ""
        for part in parts:
            if "thinking" in part:
                thinking_content += part["thinking"]

        # Build chunk with reasoning content
        chunk_data = {
            "id": chunk.get("id", ""),
            "object": "chat.completion.chunk",
            "created": chunk.get("createTime", 0),
            "model": chunk.get("model", "unknown"),
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": None,
                }
            ],
        }

        # Add reasoning content if available
        if thinking_content:
            chunk_data["choices"][0]["delta"]["content"] = (
                f"<thinking>{thinking_content}</thinking>"
            )
            chunk_data["choices"][0]["delta"]["reasoning_content"] = thinking_content

        return chunk_data

    def transform_function_response(
        self, tool_call_id: str, function_name: str, result: Any, is_error: bool = False
    ) -> Dict[str, Any]:
        """Transform function/tool response to Gemini format

        Args:
            tool_call_id: Tool call ID
            function_name: Function name
            result: Function result
            is_error: Whether result is an error

        Returns:
            Gemini function response dictionary
        """
        return {
            "role": "tool",
            "parts": [
                {
                    "function_response": {
                        "name": function_name,
                        "response": str(result),
                        "id": tool_call_id if is_error else None,
                    }
                }
            ],
        }

    def extract_grounding_metadata(
        self, response: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract grounding/web search metadata from Gemini response

        Args:
            response: Gemini response

        Returns:
            OpenAI annotations dictionary or None
        """
        grounding_metadata = response.get("groundingMetadata", {})

        if not grounding_metadata:
            return None

        # Convert to OpenAI annotations format
        annotations = []

        # Process grounding chunks
        for chunk in grounding_metadata.get("groundingChunks", []):
            if "webSearchEntity" in chunk:
                annotations.append(
                    {
                        "type": "grounding",
                        "text": chunk["webSearchEntity"].get("name", ""),
                        "start_index": 0,
                        "end_index": 0,
                    }
                )

        if annotations:
            return {"type": "annotations", "annotations": annotations}

        return None


class ModelResponseIterator:
    """Track state during streaming response processing

    Based on litellm's ModelResponseIterator, this class tracks:
    - content_blocks: All content blocks seen so far
    - tool_index: Count of tool calls
    - is_response_format_tool: Whether JSON mode is active
    """

    def __init__(self) -> None:
        """Initialize iterator state"""
        self.content_blocks: List[Dict[str, Any]] = []
        self.tool_index: int = 0
        self.is_response_format_tool: bool = False
        self.current_tool_call_id: Optional[str] = None
        self.delta_text_accumulator: str = ""

    def handle_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a Gemini streaming chunk

        Args:
            chunk: Gemini chunk data

        Returns:
            Processed chunk for OpenAI format
        """
        # Extract content block from chunk
        content_block = chunk.get("chunk", {}).get("content", {})
        content_parts = content_block.get("parts", [])

        # Track content blocks
        if content_parts:
            self.content_blocks.append(content_parts[0] if content_parts else {})

        # Handle tool calls
        for part in content_parts:
            if "function_call" in part:
                self.tool_index += 1
                # Track current tool call
                self.current_tool_call_id = part["function_call"].get("id")

        # Detect JSON mode (response_format_tool)
        generation_config = chunk.get("chunk", {}).get("generationConfig", {})
        if generation_config.get("responseMimeType") == "application/json":
            self.is_response_format_tool = True

        # Build delta
        delta = {"content": ""}

        # Accumulate text
        for part in content_parts:
            if "text" in part:
                delta["content"] += part["text"]
                self.delta_text_accumulator += part["text"]

            elif "thinking" in part:
                # Handle thinking blocks
                thinking = part["thinking"]
                delta["reasoning_content"] = thinking

        # Check for finish
        finish_reason = chunk.get("chunk", {}).get("finishReason")
        if finish_reason:
            delta["finish_reason"] = finish_reason

        # Build result chunk
        result = {
            "id": chunk.get("id", ""),
            "object": "chat.completion.chunk",
            "created": chunk.get("createTime", 0),
            "model": chunk.get("model", "unknown"),
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": None if not finish_reason else finish_reason,
                }
            ],
        }

        # Add usage if available
        if "usageMetadata" in chunk:
            result["usage"] = {
                "prompt_tokens": chunk["usageMetadata"].get("promptTokenCount", 0),
                "completion_tokens": chunk["usageMetadata"].get(
                    "candidatesTokenCount", 0
                ),
                "total_tokens": chunk["usageMetadata"].get("totalTokenCount", 0),
            }

        return result

    def reset(self) -> None:
        """Reset iterator state"""
        self.content_blocks.clear()
        self.tool_index = 0
        self.is_response_format_tool = False
        self.current_tool_call_id = None
        self.delta_text_accumulator = ""

    def get_state(self) -> Dict[str, Any]:
        """Get current iterator state

        Returns:
            Dictionary of current state
        """
        return {
            "content_blocks_count": len(self.content_blocks),
            "tool_index": self.tool_index,
            "is_response_format_tool": self.is_response_format_tool,
            "current_tool_call_id": self.current_tool_call_id,
            "accumulated_text_length": len(self.delta_text_accumulator),
        }
