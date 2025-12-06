"""Main Gemini adapter implementation"""

from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional

from ...core.base_adapter import BaseAdapter
from ...core.exceptions import ValidationError
from src.transllm.core.schema import (
    CoreRequest,
    CoreResponse,
    StreamEvent,
    Message,
    UsageStatistics,
    ToolDefinition,
    GenerationParameters,
    Provider,
)
from .transformation import GeminiRequestTransformer
from .response_handler import GeminiResponseHandler


class GeminiAdapter(BaseAdapter):
    """Adapter for Google Gemini API format

    Supports Gemini 1.5 and 2.x models with:
    - Text and multimodal content (images, audio, video)
    - Function calling (tools)
    - Thinking blocks (Gemini 2.x/3.x)
    - Streaming responses
    - Structured output
    - Grounding and web search
    """

    def __init__(
        self,
        provider_name: Provider = Provider.gemini,
        model: Optional[str] = None
    ) -> None:
        """Initialize Gemini adapter

        Args:
            provider_name: Provider identifier (Provider enum)
            model: Model name for version-specific behavior
        """
        super().__init__(provider_name)
        self.model = model
        self.request_transformer = GeminiRequestTransformer()
        self.response_handler = GeminiResponseHandler()

    def to_unified_request(self, data: Dict[str, Any]) -> CoreRequest:
        """Convert Gemini request to unified IR format

        Args:
            data: Gemini-style request dictionary

        Returns:
            Unified IR CoreRequest

        Raises:
            ConversionError: If conversion fails
            ValidationError: If data is invalid
        """
        try:
            # Extract system instruction
            system_instruction = None
            if "system_instruction" in data:
                system_parts = data["system_instruction"].get("parts", [])
                if system_parts:
                    system_instruction = system_parts[0].get("text")

            # Extract contents (messages)
            contents = data.get("contents", [])
            messages = []

            # Add system message if present
            if system_instruction:
                messages.append(
                    self.to_unified_message({
                        "role": "system",
                        "content": system_instruction
                    })
                )

            # Convert contents to messages
            for content in contents:
                role = content.get("role")
                parts = content.get("parts", [])

                # Merge parts into content
                content_parts = []
                tool_calls = []

                for part in parts:
                    if "text" in part:
                        content_parts.append({
                            "type": "text",
                            "text": part["text"]
                        })
                    elif "inline_data" in part:
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{part['inline_data']['mime_type']};base64,{part['inline_data']['data']}"
                            }
                        })
                    elif "file_data" in part:
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": part["file_data"]["file_uri"]
                            }
                        })
                    elif "function_call" in part:
                        tool_calls.append({
                            "id": part["function_call"].get("id", ""),
                            "type": "function",
                            "function": {
                                "name": part["function_call"]["name"],
                                "arguments": part["function_call"].get("args", {})
                            }
                        })

                # Build content
                if content_parts and len(content_parts) == 1:
                    content_str = content_parts[0]["text"]
                else:
                    content_str = content_parts

                message = self.to_unified_message({
                    "role": role,
                    "content": content_str,
                    "tool_calls": tool_calls if tool_calls else None
                })

                messages.append(message)

            # Extract tools
            tools = []
            for tool in data.get("tools", []):
                if "function_declarations" in tool:
                    for func_decl in tool["function_declarations"]:
                        tools.append(
                            ToolDefinition(
                                name=func_decl["name"],
                                description=func_decl.get("description", ""),
                                parameters=func_decl.get("parameters", {})
                            )
                        )

            # Extract generation parameters
            generation_config = data.get("generationConfig", {})

            # Extract response format (JSON mode)
            response_format = None
            if "responseMimeType" in generation_config:
                response_format = {"type": "json_object"}

            # Extract thinking config (Gemini 2.x/3.x)
            thinking_config = data.get("thinkingConfig")
            thinking_blocks = None
            if thinking_config:
                thinking_blocks = [thinking_config]

            parameters = GenerationParameters(
                temperature=generation_config.get("temperature"),
                max_tokens=generation_config.get("maxOutputTokens"),
                top_p=generation_config.get("topP"),
                top_k=generation_config.get("topK"),
                stop_sequences=generation_config.get("stopSequences"),
                response_format=response_format,
                thinking_blocks=thinking_blocks,
            )

            return CoreRequest(
                messages=messages,
                tools=tools if tools else None,
                generation_params=parameters,
                metadata=data.get("metadata"),
                model=data.get("model", self.model),
                system_instruction=system_instruction,
            )

        except Exception as e:
            raise ValidationError(
                f"Failed to convert Gemini request to unified format: {e}"
            ) from e

    def from_unified_request(self, unified_request: CoreRequest) -> Dict[str, Any]:
        """Convert unified IR request to Gemini format

        Args:
            unified_request: Unified IR CoreRequest

        Returns:
            Gemini-style request dictionary

        Raises:
            ConversionError: If conversion fails
            ValidationError: If request is invalid
        """
        try:
            # Validate request
            self._validate_unified_request(unified_request)

            # Build Gemini request
            gemini_request: Dict[str, Any] = {
                "model": unified_request.model or self.model or "gemini-1.5-pro"
            }

            # Add messages as contents
            if unified_request.messages:
                gemini_request["contents"] = [
                    self._message_to_dict(msg) for msg in unified_request.messages
                ]

            # Add generation config if present
            if unified_request.generation_params:
                gen_config = {}
                gp = unified_request.generation_params
                if gp.temperature is not None:
                    gen_config["temperature"] = gp.temperature
                if gp.max_tokens is not None:
                    gen_config["maxOutputTokens"] = gp.max_tokens
                if gp.top_p is not None:
                    gen_config["topP"] = gp.top_p
                if gp.top_k is not None:
                    gen_config["topK"] = gp.top_k
                if gp.stop_sequences is not None:
                    gen_config["stopSequences"] = gp.stop_sequences

                # Check for response format (JSON mode)
                if gp.response_format:
                    if isinstance(gp.response_format, dict):
                        if gp.response_format.get("type") == "json_object":
                            gen_config["responseMimeType"] = "application/json"

                if gen_config:
                    gemini_request["generationConfig"] = gen_config

            # Add tools if present
            if unified_request.tools:
                tools = []
                for tool in unified_request.tools:
                    # Convert ToolDefinition to Gemini function declarations
                    func_decl = {
                        "name": tool.name,
                        "description": tool.description or ""
                    }
                    if tool.parameters:
                        func_decl["parameters"] = tool.parameters
                    tools.append({
                        "function_declarations": [func_decl]
                    })
                if tools:
                    gemini_request["tools"] = tools
                    # Add toolConfig for function calling
                    gemini_request["toolConfig"] = {
                        "function_calling_config": {
                            "mode": "ANY"
                        }
                    }

            # Add system instruction if present
            if unified_request.system_instruction:
                gemini_request["system_instruction"] = {
                    "parts": [{"text": unified_request.system_instruction}]
                }

            # Add metadata if present
            if unified_request.metadata:
                gemini_request["metadata"] = unified_request.metadata

            # Add thinking config if present (Gemini 2.x/3.x)
            if unified_request.generation_params:
                thinking_blocks = getattr(unified_request.generation_params, 'thinking_blocks', None)
                if thinking_blocks:
                    gemini_request["thinkingConfig"] = thinking_blocks[0]

            # Validate the final request
            from .utils import validate_gemini_request
            validate_gemini_request(gemini_request)

            return gemini_request

        except Exception as e:
            raise ValidationError(
                f"Failed to convert unified request to Gemini format: {e}"
            ) from e

    def to_unified_response(self, data: Dict[str, Any]) -> CoreResponse:
        """Convert Gemini response to unified IR format

        Args:
            data: Gemini-style response dictionary

        Returns:
            Unified IR CoreResponse

        Raises:
            ConversionError: If conversion fails
            ValidationError: If response is invalid
        """
        try:
            # Extract candidates
            candidates = data.get("candidates", [])

            if not candidates:
                # Handle empty response
                return CoreResponse(
                    id=data.get("id"),
                    object="chat.completion",
                    created=data.get("createTime"),
                    model=data.get("model"),
                    choices=[],
                    usage=UsageStatistics(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                )

            # Take first candidate
            candidate = candidates[0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            # Build message
            message = self._extract_message_from_parts(parts)

            # Build choice
            from src.transllm.core.schema import Choice, ResponseMessage as IRResponseMessage, FinishReason
            response_message = IRResponseMessage(
                role=message["role"],
                content=message["content"],
                tool_calls=message.get("tool_calls"),
            )

            # Convert finish reason string to enum
            finish_reason_str = candidate.get("finishReason")
            finish_reason_enum = None
            if finish_reason_str:
                # Map Gemini finish reasons to IR
                # Normalize to lowercase to handle Gemini's uppercase values (STOP, MAX_TOKENS, SAFETY, etc.)
                reason_mapping = {
                    "stop": FinishReason.stop,
                    "length": FinishReason.length,
                    "content_filter": FinishReason.content_filter,
                    "tool_calls": FinishReason.tool_calls,
                    "safety": FinishReason.safety,
                    "recitation": FinishReason.recitation,
                    # Handle additional Gemini-specific finish reasons
                    "max_tokens": FinishReason.length,  # MAX_TOKENS
                    "language": FinishReason.content_filter,  # LANGUAGE
                    "other": FinishReason.other,  # OTHER
                    "blocklist": FinishReason.content_filter,  # BLOCKLIST
                    "prohibited_content": FinishReason.content_filter,  # PROHIBITED_CONTENT
                    "spii": FinishReason.content_filter,  # SPII
                    "malformed_function_call": FinishReason.stop,  # MALFORMED_FUNCTION_CALL
                    "image_safety": FinishReason.content_filter,  # IMAGE_SAFETY
                    "finish_reason_unspecified": FinishReason.stop,  # FINISH_REASON_UNSPECIFIED
                }
                # Convert Gemini's uppercase values to lowercase before lookup
                normalized_reason = finish_reason_str.lower()
                finish_reason_enum = reason_mapping.get(normalized_reason)

            choice = Choice(
                index=0,
                message=response_message,
                finish_reason=finish_reason_enum,
            )

            # Extract usage
            usage_data = self.response_handler._extract_usage(data)
            usage = UsageStatistics(
                prompt_tokens=usage_data["prompt_tokens"],
                completion_tokens=usage_data["completion_tokens"],
                total_tokens=usage_data["total_tokens"],
                reasoning_tokens=usage_data.get("reasoning_tokens"),
            )

            return CoreResponse(
                id=data.get("id"),
                object="chat.completion",
                created=data.get("createTime"),
                model=data.get("model"),
                choices=[choice],
                usage=usage,
            )

        except Exception as e:
            raise ValidationError(
                f"Failed to convert Gemini response to unified format: {e}"
            ) from e

    def _extract_message_from_parts(self, parts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract message from Gemini parts

        Args:
            parts: Gemini content parts

        Returns:
            Message dictionary
        """
        content_parts = []
        tool_calls = []

        for part in parts:
            if "text" in part:
                content_parts.append(part["text"])

            elif "function_call" in part:
                func_call = part["function_call"]
                # Convert to ToolCall object
                from src.transllm.core.schema import ToolCall
                tool_call = ToolCall(
                    identifier=None,  # Gemini doesn't provide IDs
                    name=func_call["name"],
                    arguments=func_call.get("args", {})
                )
                tool_calls.append(tool_call)

        # Build content
        if len(content_parts) == 1:
            content = content_parts[0]
        elif content_parts:
            content = "\n".join(content_parts)
        else:
            content = ""

        # Build message
        message = {
            "role": "assistant",
            "content": content
        }

        if tool_calls:
            message["tool_calls"] = tool_calls

        return message

    def from_unified_response(self, unified_response: CoreResponse) -> Dict[str, Any]:
        """Convert unified IR response to Gemini format

        Args:
            unified_response: Unified IR CoreResponse

        Returns:
            Gemini-style response dictionary

        Raises:
            ConversionError: If conversion fails
            ValidationError: If response is invalid
        """
        try:
            # Build Gemini response from unified response
            gemini_response: Dict[str, Any] = {
                "id": unified_response.id or f"gemini-{hash(str(unified_response))}",
                "model": unified_response.model,
                "createTime": unified_response.created or 0,
            }

            # Add candidates from choices
            candidates = []
            for choice in unified_response.choices:
                message = choice.message

                # Build content parts
                parts = []
                if isinstance(message.content, str):
                    if message.content:  # Only add non-empty text
                        parts.append({"text": message.content})
                elif isinstance(message.content, list):
                    for content_part in message.content:
                        part_type = content_part.get("type")
                        if part_type == "text":
                            parts.append({"text": content_part.get("text", "")})
                        elif part_type == "image_url":
                            # For image parts, we'd need to know if it's inline_data or file_data
                            # This is a simplified version
                            url = content_part.get("image_url", {}).get("url", "")
                            if url.startswith("data:"):
                                # Extract mime type and data
                                match = url.match(r'data:([^;]+);base64,(.+)') if hasattr(url, 'match') else None
                                if match:
                                    parts.append({
                                        "inline_data": {
                                            "mime_type": match.group(1),
                                            "data": match.group(2)
                                        }
                                    })
                        elif part_type == "tool_result":
                            # Handle tool results
                            parts.append({
                                "function_response": {
                                    "name": content_part.get("tool_name", "unknown"),
                                    "response": content_part.get("content", "")
                                }
                            })

                # Handle tool calls
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        parts.append({
                            "function_call": {
                                "name": tool_call.name,
                                "args": tool_call.arguments
                            }
                        })

                candidate = {
                    "content": {
                        "parts": parts,
                        "role": message.role
                    },
                    "finishReason": choice.finish_reason.value if choice.finish_reason else "stop",
                }

                candidates.append(candidate)

            gemini_response["candidates"] = candidates

            # Add usage metadata
            usage = unified_response.usage
            if usage:
                gemini_response["usageMetadata"] = {
                    "promptTokenCount": usage.prompt_tokens,
                    "candidatesTokenCount": usage.completion_tokens,
                    "totalTokenCount": usage.total_tokens,
                }
                if usage.reasoning_tokens:
                    gemini_response["usageMetadata"]["thoughtsTokenCount"] = usage.reasoning_tokens

            return gemini_response

        except Exception as e:
            raise ValidationError(
                f"Failed to convert unified response to Gemini format: {e}"
            ) from e

    def transform_stream(
        self,
        stream: Generator[Dict[str, Any], None, None]
    ) -> Generator[StreamEvent, None, None]:
        """Transform Gemini streaming response to unified IR

        Args:
            stream: Generator of Gemini streaming chunks

        Yields:
            Unified IR StreamEvent
        """
        for chunk in self.response_handler.transform_streaming_response(stream):
            # Convert to StreamEvent
            choices_data = chunk.get("choices", [])
            if choices_data:
                delta = choices_data[0].get("delta", {})
                content = delta.get("content")
                finish_reason = choices_data[0].get("finish_reason")

                yield StreamEvent(
                    type="message_delta" if content else "message_complete",
                    index=choices_data[0].get("index", 0),
                    content=content,
                    finish_reason=finish_reason,
                    model=chunk.get("model"),
                )

    def _validate_unified_request(self, unified_request: CoreRequest) -> None:
        """Validate unified request before conversion

        Args:
            unified_request: Unified IR request

        Raises:
            ValidationError: If request is invalid
        """
        if not unified_request.messages:
            raise ValidationError("Request must contain at least one message")

        # Validate each message
        for msg in unified_request.messages:
            if not msg.role:
                raise ValidationError("Message must have a role")
            if not msg.content and not msg.tool_calls:
                raise ValidationError("Message must have content or tool calls")

    def _message_to_dict(self, message: Message) -> Dict[str, Any]:
        """Convert unified Message to Gemini dict format

        Args:
            message: Unified Message

        Returns:
            Gemini message dictionary with 'role' and 'parts'
        """
        # Convert role enum to string
        role_str = message.role.value if hasattr(message.role, 'value') else str(message.role)

        result = {
            "role": role_str,
        }

        # Build parts from content
        parts = []

        # Handle content
        if message.content:
            if isinstance(message.content, str):
                # Simple text content
                parts.append({
                    "text": message.content
                })
            elif isinstance(message.content, list):
                # Multimodal content
                for content_part in message.content:
                    if hasattr(content_part, 'type'):
                        # ContentBlock object
                        part_type = content_part.type
                        if part_type.value == "text":
                            parts.append({
                                "text": content_part.text
                            })
                        elif part_type.value == "image_url":
                            # Convert image_url to Gemini format
                            image_url = content_part.image_url.url
                            if image_url.startswith("data:"):
                                # Base64 data URI
                                import re
                                match = re.match(r'data:([^;]+);base64,(.+)', image_url)
                                if match:
                                    parts.append({
                                        "inline_data": {
                                            "mime_type": match.group(1),
                                            "data": match.group(2)
                                        }
                                    })
                            else:
                                # Assume it's a file URI
                                parts.append({
                                    "file_data": {
                                        "file_uri": image_url
                                    }
                                })
                    else:
                        # Dict-like content part
                        part_type = content_part.get("type")
                        if part_type == "text":
                            parts.append({
                                "text": content_part.get("text", "")
                            })
                        elif part_type == "image_url":
                            image_url = content_part.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:"):
                                import re
                                match = re.match(r'data:([^;]+);base64,(.+)', image_url)
                                if match:
                                    parts.append({
                                        "inline_data": {
                                            "mime_type": match.group(1),
                                            "data": match.group(2)
                                        }
                                    })
                            else:
                                parts.append({
                                    "file_data": {
                                        "file_uri": image_url
                                    }
                                })

        # Handle tool calls
        if message.tool_calls:
            for call in message.tool_calls:
                parts.append({
                    "function_call": {
                        "id": call.identifier or "",
                        "name": call.name,
                        "arguments": call.arguments
                    }
                })

        # Add parts to result
        if parts:
            result["parts"] = parts

        return result

    def get_supported_features(self) -> Dict[str, bool]:
        """Get list of features supported by this adapter

        Returns:
            Dictionary of feature names to support status
        """
        return {
            "streaming": True,
            "multimodal": True,
            "function_calling": True,
            "json_mode": True,
            "reasoning": True,
            "grounding": True,
            "structured_output": True,
        }

    def get_model_capabilities(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Get model-specific capabilities

        Args:
            model: Model name

        Returns:
            Capabilities dictionary
        """
        model = model or self.model or "gemini-1.5-pro"

        # Base capabilities for all Gemini models
        capabilities = {
            "max_input_tokens": 1048576,  # 1M tokens for 1.5 Pro
            "max_output_tokens": 8192,
            "multimodal": True,
            "streaming": True,
        }

        # Model-specific adjustments
        if "flash" in model.lower():
            capabilities["max_input_tokens"] = 1048576
            capabilities["max_output_tokens"] = 8192

        if "pro" in model.lower():
            capabilities["max_input_tokens"] = 2097152  # 2M tokens
            capabilities["max_output_tokens"] = 8192

        # Thinking blocks (Gemini 2.x/3.x)
        if "2.0" in model or "3.0" in model:
            capabilities["reasoning"] = True
            capabilities["thinking_blocks"] = True

        return capabilities
