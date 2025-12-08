"""OpenAI adapter for format conversion"""

from __future__ import annotations

import json
from typing import Any

from ...core.base_adapter import BaseAdapter
from ...core.schema import (
    CoreRequest,
    CoreResponse,
    StreamEvent,
    Message,
    Choice,
    ResponseMessage,
    UsageStatistics,
    ContentBlock,
    ToolCall,
    ToolDefinition,
    GenerationParameters,
    Provider,
)


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI API format

    OpenAI uses a straightforward chat completion format that's widely adopted
    as a de facto standard. This adapter converts between OpenAI format and
    the brand-neutral IR.
    """

    def __init__(self, provider_name: Provider = Provider.openai) -> None:
        super().__init__(provider_name)

    def to_unified_request(self, data: dict[str, Any]) -> CoreRequest:
        """Convert OpenAI chat completion request to unified IR"""
        # Extract system instruction - either from dedicated field or from messages array
        system_instruction = data.get("system_instruction")
        messages = []

        # Convert all messages from messages array
        for msg_data in data.get("messages", []):
            message = self.to_unified_message(msg_data)
            # Extract system message content for cross-provider compatibility
            if message.role.value == "system" and system_instruction is None:
                extracted_instruction = self._extract_system_instruction_from_message(
                    message
                )
                if extracted_instruction:
                    system_instruction = extracted_instruction
            # IMPORTANT: Keep system message in messages array to preserve metadata and structure
            # This ensures idempotency for OpenAI format
            messages.append(message)

        tools = None
        if "tools" in data:
            tools = []
            for tool in data["tools"]:
                # OpenAI tools are wrapped in a "function" object
                if "function" in tool:
                    tools.append(
                        ToolDefinition(
                            name=tool["function"]["name"],
                            description=tool["function"].get("description", ""),
                            parameters=tool["function"].get("parameters", {}),
                            metadata=tool.get("metadata"),
                        )
                    )
                # Or direct tool format
                elif "name" in tool:
                    tools.append(
                        ToolDefinition(
                            name=tool["name"],
                            description=tool.get("description", ""),
                            parameters=tool.get("parameters", {}),
                            metadata=tool.get("metadata"),
                        )
                    )

        # Extract generation parameters from top level or nested
        gen_params = data.get("generation_params", {})
        if not gen_params:
            # Try to extract from top-level fields, but only if they exist in data
            # Map OpenAI field names to IR field names
            field_mapping = {
                "temperature": "temperature",
                "max_tokens": "max_tokens",
                "max_completion_tokens": "max_completion_tokens",
                "top_p": "top_p",
                "top_k": "top_k",
                "stop": "stop_sequences",
                "stream": "stream",
                "seed": "seed",
                "presence_penalty": "presence_penalty",
                "frequency_penalty": "frequency_penalty",
                "logit_bias": "logit_bias",
                "response_format": "response_format",
                "logprobs": "logprobs",
                "top_logprobs": "top_logprobs",
                "n": "n",
                "stream_options": "stream_options",
            }
            gen_params = {}
            for openai_field, ir_field in field_mapping.items():
                if openai_field in data:
                    gen_params[ir_field] = data[openai_field]

        generation_params = None
        # Only create GenerationParameters if there are non-None values
        filtered_params = {k: v for k, v in gen_params.items() if v is not None}
        if filtered_params:
            generation_params = GenerationParameters(**filtered_params)

        return CoreRequest(
            model=data.get("model", ""),
            messages=messages,
            tools=tools,
            tool_choice=data.get("tool_choice"),
            parallel_tool_calls=data.get("parallel_tool_calls"),
            system_instruction=system_instruction or data.get("system_instruction"),
            generation_params=generation_params,
            metadata=data.get("metadata"),
        )

    def from_unified_request(self, unified_request: CoreRequest) -> dict[str, Any]:
        """Convert unified IR request to OpenAI format"""
        messages = []

        # Check if there's already a system message in the messages array
        has_system_message = any(
            msg.role.value == "system"
            if hasattr(msg.role, "value")
            else msg.role == "system"
            for msg in unified_request.messages
        )

        # Add system instruction as a system message only if not already present
        if unified_request.system_instruction and not has_system_message:
            messages.append(
                {
                    "role": "system",
                    "content": unified_request.system_instruction,
                }
            )

        # Convert all messages (including system messages if present)
        messages.extend(
            [self.from_unified_message(msg) for msg in unified_request.messages]
        )

        tools = None
        if unified_request.tools:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in unified_request.tools
            ]

        generation_params = {}
        if unified_request.generation_params:
            gp = unified_request.generation_params
            if gp.temperature is not None:
                generation_params["temperature"] = gp.temperature
            if gp.max_tokens is not None:
                generation_params["max_tokens"] = gp.max_tokens
            if gp.max_completion_tokens is not None:
                generation_params["max_completion_tokens"] = gp.max_completion_tokens
            if gp.top_p is not None:
                generation_params["top_p"] = gp.top_p
            if gp.top_k is not None:
                generation_params["top_k"] = gp.top_k
            if gp.stop_sequences is not None:
                generation_params["stop"] = gp.stop_sequences
            if gp.stream is not None:
                generation_params["stream"] = gp.stream
            if gp.seed is not None:
                generation_params["seed"] = gp.seed
            if gp.presence_penalty is not None:
                generation_params["presence_penalty"] = gp.presence_penalty
            if gp.frequency_penalty is not None:
                generation_params["frequency_penalty"] = gp.frequency_penalty
            if gp.logit_bias is not None:
                generation_params["logit_bias"] = gp.logit_bias
            if gp.response_format is not None:
                generation_params["response_format"] = gp.response_format
            if gp.logprobs is not None:
                generation_params["logprobs"] = gp.logprobs
            if gp.top_logprobs is not None:
                generation_params["top_logprobs"] = gp.top_logprobs
            if gp.n is not None:
                generation_params["n"] = gp.n
            if gp.stream_options is not None:
                generation_params["stream_options"] = gp.stream_options

        result = {
            "model": unified_request.model,
            "messages": messages,
        }

        if unified_request.system_instruction and not has_system_message:
            result["system_instruction"] = unified_request.system_instruction

        if tools:
            result["tools"] = tools

        if unified_request.tool_choice:
            result["tool_choice"] = unified_request.tool_choice

        if unified_request.parallel_tool_calls is not None:
            result["parallel_tool_calls"] = unified_request.parallel_tool_calls

        if generation_params:
            result.update(generation_params)

        if unified_request.metadata:
            result["metadata"] = unified_request.metadata

        return result

    def to_unified_response(self, data: dict[str, Any]) -> CoreResponse:
        """Convert OpenAI chat completion response to unified IR"""
        choices = []
        for choice_data in data.get("choices", []):
            choice = self._convert_choice(choice_data)
            choices.append(choice)

        usage = None
        if "usage" in data:
            usage_data = data["usage"]
            usage = UsageStatistics(
                prompt_tokens=usage_data.get("prompt_tokens"),
                completion_tokens=usage_data.get("completion_tokens"),
                total_tokens=usage_data.get("total_tokens"),
                input_tokens=usage_data.get("input_tokens"),
                output_tokens=usage_data.get("output_tokens"),
                cached_tokens=usage_data.get("cached_tokens"),
            )

        # Extract grounding attributions if present
        grounding_attributions = None
        if "grounding_attributions" in data and data["grounding_attributions"]:
            grounding_attributions = []
            for attr in data["grounding_attributions"]:
                from ...core.schema import GroundingAttribution

                grounding_attributions.append(
                    GroundingAttribution(
                        content_index=attr.get("content_index"),
                        model_decision=attr.get("model_decision"),
                        grounding_chunk_indices=attr.get("grounding_chunk_indices"),
                        source_chunk_indices=attr.get("source_chunk_indices"),
                    )
                )

        return CoreResponse(
            id=data.get("id"),
            object=data.get("object"),
            created=data.get("created"),
            model=data.get("model", ""),
            choices=choices,
            usage=usage,
            grounding_attributions=grounding_attributions,
            metadata=data.get("metadata"),
        )

    def from_unified_response(self, unified_response: CoreResponse) -> dict[str, Any]:
        """Convert unified IR response to OpenAI format"""
        choices = []
        for choice in unified_response.choices:
            # Convert ResponseMessage to dict
            message_dict = self._response_message_to_dict(choice.message)
            choice_data = {
                "index": choice.index,
                "message": message_dict,
            }
            if choice.finish_reason:
                # Convert enum to string
                finish_reason_str = (
                    choice.finish_reason.value
                    if hasattr(choice.finish_reason, "value")
                    else str(choice.finish_reason)
                )
                choice_data["finish_reason"] = finish_reason_str
            if choice.logprobs:
                choice_data["logprobs"] = choice.logprobs
            choices.append(choice_data)

        result = {
            "id": unified_response.id,
            "object": unified_response.object,
            "created": unified_response.created,
            "model": unified_response.model,
            "choices": choices,
        }

        if unified_response.usage:
            usage_dict = {}
            if unified_response.usage.prompt_tokens is not None:
                usage_dict["prompt_tokens"] = unified_response.usage.prompt_tokens
            if unified_response.usage.completion_tokens is not None:
                usage_dict["completion_tokens"] = (
                    unified_response.usage.completion_tokens
                )
            usage_dict["total_tokens"] = unified_response.usage.total_tokens
            if unified_response.usage.input_tokens is not None:
                usage_dict["input_tokens"] = unified_response.usage.input_tokens
            if unified_response.usage.output_tokens is not None:
                usage_dict["output_tokens"] = unified_response.usage.output_tokens
            if unified_response.usage.cached_tokens is not None:
                usage_dict["cached_tokens"] = unified_response.usage.cached_tokens
            result["usage"] = usage_dict

        # Add grounding attributions if present
        if unified_response.grounding_attributions:
            result["grounding_attributions"] = [
                {
                    "content_index": attr.content_index,
                    "model_decision": attr.model_decision,
                    "grounding_chunk_indices": attr.grounding_chunk_indices,
                    "source_chunk_indices": attr.source_chunk_indices,
                }
                for attr in unified_response.grounding_attributions
            ]

        if unified_response.metadata:
            result["metadata"] = unified_response.metadata

        return result

    def _convert_choice(self, choice_data: dict[str, Any]) -> Choice:
        """Convert OpenAI choice to unified IR"""
        message_data = choice_data.get("message", {})
        message = self._convert_response_message(message_data)

        return Choice(
            message=message,
            index=choice_data.get("index", 0),
            finish_reason=choice_data.get("finish_reason"),
            logprobs=choice_data.get("logprobs"),
        )

    def _convert_response_message(
        self, message_data: dict[str, Any]
    ) -> ResponseMessage:
        """Convert OpenAI message to unified IR"""
        content = message_data.get("content", "")

        # Convert tool_calls
        tool_calls = None
        if "tool_calls" in message_data and message_data["tool_calls"]:
            tool_calls = []
            for tc in message_data["tool_calls"]:
                # OpenAI tool_calls have arguments as a JSON string that needs to be parsed
                arguments = tc.get("function", {}).get("arguments", "")
                if isinstance(arguments, str):
                    # Try to parse JSON string to dict
                    try:
                        import json

                        arguments = json.loads(arguments)
                    except (json.JSONDecodeError, TypeError):
                        # Keep as string if parsing fails
                        pass

                tool_calls.append(
                    ToolCall(
                        identifier=tc.get("id", ""),
                        name=tc.get("function", {}).get("name", ""),
                        arguments=arguments,
                    )
                )

        return ResponseMessage(
            role=message_data.get("role", "assistant"),
            content=content,
            tool_calls=tool_calls,
            reasoning_content=message_data.get("reasoning_content"),
            thinking_blocks=message_data.get("thinking_blocks"),
            identifier=message_data.get("id"),
        )

    def _convert_message(self, message_data: dict[str, Any]) -> Message:
        """Convert OpenAI message to unified IR"""
        content = message_data.get("content", "")

        # Convert content list of dicts to list of ContentBlock objects
        if isinstance(content, list):
            content_blocks = []
            for content_item in content:
                if isinstance(content_item, dict):
                    content_type = content_item.get("type")
                    if content_type == "text":
                        content_blocks.append(
                            ContentBlock(type="text", text=content_item.get("text", ""))
                        )
                    elif content_type == "image_url":
                        from ...core.schema import ImageUrl

                        content_blocks.append(
                            ContentBlock(
                                type="image_url",
                                image_url=ImageUrl(
                                    url=content_item.get("image_url", {}).get(
                                        "url", ""
                                    ),
                                    detail=content_item.get("image_url", {}).get(
                                        "detail"
                                    ),
                                ),
                            )
                        )
                    elif content_type == "tool_result":
                        from ...core.schema import ToolResult

                        content_blocks.append(
                            ContentBlock(
                                type="tool_result",
                                tool_result=ToolResult(
                                    tool_name=content_item.get("tool_result", {}).get(
                                        "tool_name", ""
                                    ),
                                    result=content_item.get("tool_result", {}).get(
                                        "result", ""
                                    ),
                                ),
                            )
                        )
                    elif content_type == "reasoning":
                        from ...core.schema import Reasoning

                        content_blocks.append(
                            ContentBlock(
                                type="reasoning",
                                reasoning=Reasoning(
                                    content=content_item.get("reasoning", {}).get(
                                        "content", ""
                                    )
                                ),
                            )
                        )
            content = content_blocks

        tool_calls = None
        if "tool_calls" in message_data and message_data["tool_calls"]:
            import json

            tool_calls = []
            for tc in message_data["tool_calls"]:
                # OpenAI tool_calls have arguments as a JSON string that needs to be parsed
                arguments_raw = tc.get("function", {}).get("arguments", {})
                arguments = arguments_raw
                if isinstance(arguments_raw, str):
                    try:
                        arguments = json.loads(arguments_raw)
                    except (json.JSONDecodeError, TypeError):
                        # Keep as empty dict if parsing fails
                        arguments = {}

                tool_calls.append(
                    ToolCall(
                        identifier=tc.get("id", ""),
                        name=tc.get("function", {}).get("name", ""),
                        arguments=arguments,
                    )
                )

        return Message(
            role=message_data.get("role", "user"),
            content=content,
            metadata=message_data.get("metadata"),
            tool_calls=tool_calls,
            identifier=message_data.get("id"),
        )

    def _extract_system_instruction_from_message(self, message: Message) -> str | None:
        """Helper to pull the human-readable text out of system messages"""
        content = message.content
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list):
            text_parts = []
            for block in content:
                text_value = None
                if isinstance(block, ContentBlock):
                    text_value = block.text
                elif isinstance(block, dict):
                    text_value = block.get("text")
                if text_value:
                    text_parts.append(text_value)
            if text_parts:
                return "".join(text_parts)
        return None

    def _response_message_to_dict(
        self, response_message: ResponseMessage
    ) -> dict[str, Any]:
        """Convert ResponseMessage Pydantic model to dict"""
        # Convert enum to string
        role_str = (
            response_message.role.value
            if hasattr(response_message.role, "value")
            else str(response_message.role)
        )

        # Handle content (could be string or list of ContentBlock)
        content = response_message.content
        if isinstance(content, list):
            # Convert ContentBlock objects to dicts (only include non-None fields)
            new_content = []
            for cb in content:
                cb_dict = {}
                if cb.text is not None:
                    cb_dict["type"] = "text"
                    cb_dict["text"] = cb.text
                elif cb.image_url is not None:
                    cb_dict["type"] = "image_url"
                    cb_dict["image_url"] = {
                        "url": cb.image_url.url,
                        "detail": cb.image_url.detail.value
                        if hasattr(cb.image_url.detail, "value")
                        else str(cb.image_url.detail)
                        if cb.image_url.detail
                        else None,
                    }
                elif cb.tool_result is not None:
                    cb_dict["type"] = "tool_result"
                    cb_dict["tool_result"] = {
                        "tool_name": cb.tool_result.tool_name,
                        "result": cb.tool_result.result,
                    }
                elif cb.reasoning is not None:
                    cb_dict["type"] = "reasoning"
                    cb_dict["reasoning"] = {
                        "content": cb.reasoning.content,
                    }
                if cb_dict:
                    new_content.append(cb_dict)
            content = new_content

        result = {
            "role": role_str,
        }

        # Only add content field if it's not empty
        if content or (isinstance(content, list) and content):
            result["content"] = content

        if response_message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.identifier,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments)
                        if isinstance(tc.arguments, dict)
                        else tc.arguments,
                    },
                }
                for tc in response_message.tool_calls
            ]

        if response_message.reasoning_content is not None:
            result["reasoning_content"] = response_message.reasoning_content

        if response_message.thinking_blocks is not None:
            result["thinking_blocks"] = response_message.thinking_blocks

        if response_message.identifier:
            result["id"] = response_message.identifier

        return result

    def to_unified_message(self, data: dict[str, Any]) -> Message:
        """Convert OpenAI message to unified IR"""
        return self._convert_message(data)

    def from_unified_message(self, unified_message: Message) -> dict[str, Any]:
        """Convert unified IR message to OpenAI format"""
        # Convert enum to string
        role_str = (
            unified_message.role.value
            if hasattr(unified_message.role, "value")
            else str(unified_message.role)
        )

        # Handle content (could be string or list of ContentBlock)
        content = unified_message.content
        if isinstance(content, list):
            # Convert ContentBlock objects to dicts (only include non-None fields)
            new_content = []
            for cb in content:
                cb_dict = {}
                if cb.text is not None:
                    cb_dict["type"] = "text"
                    cb_dict["text"] = cb.text
                elif cb.image_url is not None:
                    cb_dict["type"] = "image_url"
                    cb_dict["image_url"] = {
                        "url": cb.image_url.url,
                        "detail": cb.image_url.detail.value
                        if hasattr(cb.image_url.detail, "value")
                        else str(cb.image_url.detail)
                        if cb.image_url.detail
                        else None,
                    }
                elif cb.tool_result is not None:
                    cb_dict["type"] = "tool_result"
                    cb_dict["tool_result"] = {
                        "tool_name": cb.tool_result.tool_name,
                        "result": cb.tool_result.result,
                    }
                elif cb.reasoning is not None:
                    cb_dict["type"] = "reasoning"
                    cb_dict["reasoning"] = {
                        "content": cb.reasoning.content,
                    }
                if cb_dict:
                    new_content.append(cb_dict)
            content = new_content

        result = {
            "role": role_str,
            "content": content,
        }

        if unified_message.metadata:
            result["metadata"] = unified_message.metadata

        if unified_message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.identifier,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments)
                        if isinstance(tc.arguments, dict)
                        else tc.arguments,
                    },
                }
                for tc in unified_message.tool_calls
            ]

        if unified_message.identifier:
            result["id"] = unified_message.identifier

        return result

    def _to_unified_stream_event_impl(
        self, data: dict[str, Any], sequence_id: int, timestamp: float
    ) -> StreamEvent:
        """Convert OpenAI stream event to unified IR"""
        # OpenAI uses delta chunks in choices
        choice = data.get("choices", [{}])[0]
        delta = choice.get("delta", {})

        content_delta = delta.get("content", "")
        finish_reason = choice.get("finish_reason")

        # Dynamically determine event type based on content and finish reason
        event_type = "content_delta"  # Default type

        if finish_reason:
            # If there's a finish reason, this is a completion event
            event_type = "content_finish"
        elif "tool_calls" in delta and delta["tool_calls"]:
            # Tool call event
            tc = delta["tool_calls"][0]
            if tc.get("function", {}).get("name") or tc.get("function", {}).get(
                "arguments"
            ):
                event_type = "tool_call_delta"

        # Handle tool calls in streaming
        tool_call_delta = None
        if "tool_calls" in delta and delta["tool_calls"]:
            tc = delta["tool_calls"][0]  # First tool call
            tool_call_delta = {
                "name": tc.get("function", {}).get("name"),
                "arguments_delta": tc.get("function", {}).get("arguments"),
                "identifier": tc.get("id"),
            }

        return StreamEvent(
            type=event_type,
            sequence_id=sequence_id,
            timestamp=timestamp,
            content_delta=content_delta,
            tool_call_delta=tool_call_delta,
            finish_reason=finish_reason,
            content_index=choice.get("index"),
            metadata=data.get("metadata"),
        )

    def from_unified_stream_event(self, unified_event: StreamEvent) -> dict[str, Any]:
        """Convert unified IR stream event to OpenAI format"""
        # OpenAI uses delta chunks
        delta = {}

        # Only add content_delta if it's not None and not empty string
        if (
            unified_event.content_delta is not None
            and unified_event.content_delta != ""
        ):
            delta["content"] = unified_event.content_delta

        if unified_event.tool_call_delta:
            tc_delta = unified_event.tool_call_delta
            tool_call = {
                "index": 0,
                "id": tc_delta.identifier or "",
                "type": "function",
                "function": {
                    "name": tc_delta.name or "",
                    "arguments": tc_delta.arguments_delta or "",
                },
            }
            delta["tool_calls"] = [tool_call]

        choice = {
            "index": unified_event.content_index or 0,
            "delta": delta,
        }

        if unified_event.finish_reason:
            choice["finish_reason"] = unified_event.finish_reason

        result = {
            "choices": [choice],
            "sequence_id": unified_event.sequence_id,
            "timestamp": unified_event.timestamp,
        }

        if unified_event.metadata is not None:
            result["metadata"] = unified_event.metadata

        return result
