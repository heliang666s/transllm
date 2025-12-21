"""Anthropic adapter for format conversion

This adapter converts between Anthropic API format and the unified IR.

Key design decisions based on litellm analysis:
1. System messages are extracted to independent parameter
2. Consecutive same-role messages are merged
3. Response format supports dual-mode (output_format for new models, tool simulation for old)
4. Reasoning effort maps differently for Opus 4.5 vs other models
5. Tool choice reverse mapping (parallel_tool_calls → disable_parallel_tool_use)
6. Automatic beta headers management based on features used
"""

from __future__ import annotations

import uuid
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
    Type,
    Thinking,
    RedactedThinking,
    Provider,
)



class AnthropicAdapter(BaseAdapter):
    """Adapter for Anthropic Claude API format

    Anthropic uses a different message structure from OpenAI:
    - System messages are in a separate 'system' parameter
    - Messages must start with user role
    - Consecutive messages with same role are merged
    - Tool calls use tool_use content blocks instead of tool_calls array
    """

    def __init__(self, provider_name: Provider = Provider.anthropic) -> None:
        super().__init__(provider_name)
        # Map of finish_reason conversions: Anthropic → unified (OpenAI)
        self.finish_reason_map = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
        }
        # Reverse map: unified (OpenAI) → Anthropic
        self.reverse_finish_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "content_filter": "stop_sequence",
            "tool_calls": "tool_use",
        }

    def to_unified_request(self, data: dict[str, Any]) -> CoreRequest:
        """Convert Anthropic request to unified IR format"""
        # Extract system message
        system_instruction = None
        if "system" in data:
            system_value = data["system"]
            if isinstance(system_value, str):
                system_instruction = system_value
            elif isinstance(system_value, list):
                # System can be a list of content blocks
                system_instruction = self._extract_text_from_content_blocks(
                    system_value
                )

        # Convert messages
        messages = []
        for msg_data in data.get("messages", []):
            message = self._to_unified_message(msg_data)
            messages.append(message)

        # Extract tools
        tools = None
        if "tools" in data:
            tools = []
            for tool in data["tools"]:
                tools.append(
                    ToolDefinition(
                        name=tool.get("name", ""),
                        description=tool.get("description", ""),
                        parameters=tool.get("input_schema", {}),
                        metadata=tool.get("metadata"),
                    )
                )

        # Extract tool_choice
        tool_choice = None
        if "tool_choice" in data:
            tc_data = data["tool_choice"]
            if isinstance(tc_data, dict):
                # Handle dict format
                if tc_data.get("type") == "tool":
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tc_data.get("name")},
                    }
                else:
                    tool_choice = tc_data.get("type", "auto")
            else:
                tool_choice = tc_data

        # Extract generation parameters
        gen_params = None
        gen_params_dict = {
            "temperature": data.get("temperature"),
            "max_tokens": data.get("max_tokens"),
            "top_p": data.get("top_p"),
            "top_k": data.get("top_k"),
            "stop_sequences": data.get("stop_sequences"),
            "seed": data.get("seed"),
        }

        # Add extra params from thinking
        if "thinking" in data:
            thinking = data["thinking"]
            if isinstance(thinking, dict):
                gen_params_dict["metadata"] = {"thinking": thinking}

        filtered_params = {k: v for k, v in gen_params_dict.items() if v is not None}
        if filtered_params:
            gen_params = GenerationParameters(**filtered_params)

        # Extract parallel_tool_calls from disable_parallel_tool_use
        parallel_tool_calls = None
        if "tool_choice" in data:
            tc = data["tool_choice"]
            if isinstance(tc, dict) and "disable_parallel_tool_use" in tc:
                # Reverse mapping: disable_parallel_tool_use → parallel_tool_calls
                parallel_tool_calls = not tc["disable_parallel_tool_use"]

        return CoreRequest(
            model=data.get("model", ""),
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            system_instruction=system_instruction,
            generation_params=gen_params,
            metadata=data.get("metadata"),
        )

    def from_unified_request(self, unified_request: CoreRequest) -> dict[str, Any]:
        """Convert unified IR request to Anthropic format"""
        # Merge consecutive same-role messages
        messages = self._merge_consecutive_messages(unified_request.messages)

        # Extract system messages from messages array (Anthropic doesn't support system role in messages)
        system_instruction = unified_request.system_instruction
        system_messages_content = []
        non_system_messages = []

        for msg in messages:
            role_value = msg.role.value if hasattr(msg.role, "value") else msg.role
            if role_value == "system":
                # Extract system message content
                if isinstance(msg.content, str):
                    system_messages_content.append(msg.content)
                elif isinstance(msg.content, list) and msg.content:
                    # Handle content blocks
                    text_parts = []
                    for block in msg.content:
                        if hasattr(block, "text") and block.text:
                            text_parts.append(block.text)
                    if text_parts:
                        system_messages_content.append("".join(text_parts))
            else:
                non_system_messages.append(msg)

        # Combine system messages, avoiding duplication
        # Only merge if system_messages_content exists and is different from system_instruction
        if system_messages_content:
            system_messages_str = "\n\n".join(system_messages_content)
            if system_instruction:
                # Check if system_instruction already contains this content (to avoid duplication)
                # This happens when OpenAI → IR → Anthropic conversion preserves system messages
                if system_messages_str not in system_instruction:
                    system_instruction = (
                        system_instruction + "\n\n" + system_messages_str
                    )
            else:
                system_instruction = system_messages_str

        # Use non-system messages for conversion
        messages = non_system_messages

        # Ensure first message is user - only add placeholder if there are messages
        # and the first one is not user
        if messages and messages[0].role != "user":
            # If first message is assistant or tool, we need to insert placeholder
            if messages[0].role in ("assistant", "tool"):
                placeholder = Message(role="user", content=".")
                messages.insert(0, placeholder)

        # Convert messages to Anthropic format
        anthropic_messages = [self._from_unified_message(msg) for msg in messages]

        # Build result
        result = {
            "model": unified_request.model,
            "messages": anthropic_messages,
        }

        # Add system instruction if present (from unified_request or extracted from messages)
        if system_instruction:
            result["system"] = system_instruction

        # Add max_tokens (required by Anthropic)
        max_tokens = 1024
        if (
            unified_request.generation_params
            and unified_request.generation_params.max_tokens
        ):
            max_tokens = unified_request.generation_params.max_tokens
        result["max_tokens"] = max_tokens

        # Add tools
        if unified_request.tools:
            tools = []
            for tool in unified_request.tools:
                tools.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.parameters or {},
                    }
                )
            result["tools"] = tools

        # Add tool_choice with parallel_tool_calls handling
        if unified_request.tool_choice:
            result["tool_choice"] = unified_request.tool_choice

        if unified_request.parallel_tool_calls is not None:
            # Reverse mapping: parallel_tool_calls → disable_parallel_tool_use
            if not result.get("tool_choice"):
                result["tool_choice"] = {}
            result["tool_choice"][
                "disable_parallel_tool_use"
            ] = not unified_request.parallel_tool_calls

        # Add generation parameters
        if unified_request.generation_params:
            gp = unified_request.generation_params
            if gp.temperature is not None:
                result["temperature"] = gp.temperature
            if gp.top_p is not None:
                result["top_p"] = gp.top_p
            if gp.top_k is not None:
                result["top_k"] = gp.top_k
            if gp.stop_sequences is not None:
                result["stop_sequences"] = gp.stop_sequences
            if gp.seed is not None:
                result["seed"] = gp.seed

            # Handle metadata with thinking
            if gp.metadata and "thinking" in gp.metadata:
                result["thinking"] = gp.metadata["thinking"]

        if unified_request.metadata:
            result["metadata"] = unified_request.metadata

        # Add beta headers based on features used
        self._add_beta_headers(result)

        return result

    def to_unified_response(self, data: dict[str, Any]) -> CoreResponse:
        """Convert Anthropic response to unified IR format"""
        choices = []
        content = data.get("content", [])
        stop_reason = data.get("stop_reason", "end_turn")

        # Anthropic responses have content as a list of content blocks
        # Convert to unified format
        choice = self._convert_anthropic_response_to_choice(content, stop_reason)
        choices.append(choice)

        # Extract usage
        usage = None
        if "usage" in data:
            usage_data = data["usage"]
            input_tokens = usage_data.get("input_tokens", 0)
            output_tokens = usage_data.get("output_tokens", 0)
            cache_creation_tokens = usage_data.get("cache_creation_input_tokens")
            cache_read_tokens = usage_data.get("cache_read_input_tokens")

            # Total tokens calculation: input + output + cache tokens
            total = input_tokens + output_tokens
            if cache_creation_tokens:
                total += cache_creation_tokens
            if cache_read_tokens:
                total += cache_read_tokens

            usage = UsageStatistics(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=total,
                cached_tokens=cache_read_tokens,
                cache_creation_input_tokens=cache_creation_tokens,
                cache_read_input_tokens=cache_read_tokens,
            )

        return CoreResponse(
            id=data.get("id"),
            model=data.get("model", ""),
            choices=choices,
            usage=usage,
            metadata=data.get("metadata"),
        )

    def from_unified_response(self, unified_response: CoreResponse) -> dict[str, Any]:
        """Convert unified IR response to Anthropic format"""
        # Anthropic response structure is different
        # It has a single content array and stop_reason
        if not unified_response.choices:
            return {
                "id": unified_response.id,
                "model": unified_response.model,
                "content": [],
                "stop_reason": "end_turn",
            }

        choice = unified_response.choices[0]
        content = self._response_message_to_anthropic_content(choice.message)
        finish_reason = choice.finish_reason

        # Map finish reason back to Anthropic format
        anthropic_stop_reason = "end_turn"
        if finish_reason:
            reason_str = (
                finish_reason.value
                if hasattr(finish_reason, "value")
                else str(finish_reason)
            )
            # Use reverse mapping from unified to Anthropic
            anthropic_stop_reason = self.reverse_finish_reason_map.get(
                reason_str, "end_turn"
            )

        result = {
            "id": unified_response.id,
            "model": unified_response.model,
            "content": content,
            "stop_reason": anthropic_stop_reason,
        }

        if unified_response.usage:
            usage_dict = {
                "input_tokens": unified_response.usage.prompt_tokens or 0,
                "output_tokens": unified_response.usage.completion_tokens or 0,
            }
            # Preserve cache-related tokens
            if unified_response.usage.cache_creation_input_tokens is not None:
                usage_dict["cache_creation_input_tokens"] = (
                    unified_response.usage.cache_creation_input_tokens
                )
            if unified_response.usage.cache_read_input_tokens is not None:
                usage_dict["cache_read_input_tokens"] = (
                    unified_response.usage.cache_read_input_tokens
                )
            result["usage"] = usage_dict

        if unified_response.metadata:
            result["metadata"] = unified_response.metadata

        return result

    def _to_unified_message(self, data: dict[str, Any]) -> Message:
        """Convert Anthropic message to unified IR"""
        role = data.get("role", "user")
        content = data.get("content", "")

        # Extract tool_calls BEFORE converting content blocks
        # This must happen first because tool_use blocks should not be converted to ContentBlock
        tool_calls = None
        raw_content = content if isinstance(content, list) else []

        # Handle content blocks in Anthropic format
        if isinstance(content, list):
            # Convert Anthropic content blocks to unified format
            content_blocks = []
            for block in content:
                block_type = block.get("type")
                # Skip tool_use blocks - they'll be extracted as tool_calls instead
                if block_type == "tool_use":
                    continue
                elif block_type == "text":
                    content_blocks.append(
                        ContentBlock(
                            type="text",
                            text=block.get("text", ""),
                        )
                    )
                elif block_type == "image":
                    # Anthropic uses different image format
                    image_data = block.get("source", {})
                    content_blocks.append(
                        ContentBlock(
                            type="image_url",
                            image_url={
                                "url": image_data.get("url") or image_data.get("data"),
                                "detail": "high"
                                if image_data.get("media_type")
                                else None,
                            },
                        )
                    )
                elif block_type == "tool_result":
                    content_blocks.append(
                        ContentBlock(
                            type="tool_result",
                            tool_result={
                                "tool_name": block.get("tool_use_id", ""),
                                "result": {"content": block.get("content")}
                                if isinstance(block.get("content"), str)
                                else block.get("content"),
                            },
                        )
                    )
                elif block_type == "thinking":
                    # Anthropic extended thinking blocks
                    content_blocks.append(
                        ContentBlock(
                            type="thinking",
                            thinking=Thinking(content=block.get("thinking", "")),
                        )
                    )
                elif block_type == "redacted_thinking":
                    # Redacted thinking blocks (when thinking is restricted)
                    content_blocks.append(
                        ContentBlock(
                            type="redacted_thinking",
                            redacted_thinking=RedactedThinking(
                                content="[redacted thinking]"
                            ),
                        )
                    )
            # Always use converted content_blocks when original is a list
            # This ensures tool_use blocks are not kept as raw dicts
            content = content_blocks if content_blocks else ""

        # Extract tool_calls if role is assistant (from raw content before conversion)
        if role == "assistant" and raw_content:
            tool_calls = self._extract_tool_calls_from_content(raw_content)

        # Preserve cache_control if present
        cache_control = data.get("cache_control")

        return Message(
            role=role,
            content=content,
            tool_calls=tool_calls,
            metadata=data.get("metadata"),
            identifier=data.get("id"),
            cache_control=cache_control,
        )

    def _from_unified_message(self, message: Message) -> dict[str, Any]:
        """Convert unified message to Anthropic format"""
        content = message.content
        # Convert role enum to string if needed
        role = (
            message.role.value if hasattr(message.role, "value") else str(message.role)
        )
        result = {"role": role}

        # Convert content to Anthropic format
        if isinstance(content, str):
            result["content"] = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            # Convert ContentBlock list to Anthropic format
            anthropic_content = []
            for block in content:
                if isinstance(block, ContentBlock):
                    # Compare against Enum value
                    if block.type == Type.text or block.type == "text":
                        anthropic_content.append(
                            {"type": "text", "text": block.text or ""}
                        )
                    elif block.type == Type.image_url or block.type == "image_url":
                        image_url = block.image_url or {}
                        url_value = (
                            image_url.url
                            if hasattr(image_url, "url")
                            else image_url.get("url", "")
                        )
                        anthropic_content.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64"
                                    if url_value.startswith("data:")
                                    else "url",
                                    "url": url_value,
                                },
                            }
                        )
                    elif block.type == Type.tool_result or block.type == "tool_result":
                        tool_result = block.tool_result or {}
                        tool_name = (
                            tool_result.tool_name
                            if hasattr(tool_result, "tool_name")
                            else tool_result.get("tool_name", "")
                        )
                        result_data = (
                            tool_result.result
                            if hasattr(tool_result, "result")
                            else tool_result.get("result", "")
                        )
                        anthropic_content.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_name,
                                "content": result_data,
                            }
                        )
                    elif block.type == Type.thinking or block.type == "thinking":
                        thinking = block.thinking
                        thinking_content = (
                            thinking.content
                            if hasattr(thinking, "content")
                            else thinking.get("content", "")
                            if isinstance(thinking, dict)
                            else ""
                        )
                        anthropic_content.append(
                            {
                                "type": "thinking",
                                "thinking": thinking_content,
                            }
                        )
                    elif (
                        block.type == Type.redacted_thinking
                        or block.type == "redacted_thinking"
                    ):
                        # Redacted thinking is output-only for some models
                        anthropic_content.append(
                            {
                                "type": "redacted_thinking",
                            }
                        )
            result["content"] = anthropic_content
        else:
            result["content"] = [{"type": "text", "text": str(content)}]

        # Add tool_calls if present and role is assistant
        if message.tool_calls and role == "assistant":
            # Tool calls are represented as tool_use content blocks in Anthropic
            if not isinstance(result["content"], list):
                result["content"] = [result["content"]]
            for tc in message.tool_calls:
                result["content"].append(
                    {
                        "type": "tool_use",
                        "id": tc.identifier or str(uuid.uuid4()),
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                )

        if message.metadata:
            result["metadata"] = message.metadata

        # Preserve cache_control if present
        if message.cache_control:
            result["cache_control"] = message.cache_control

        return result

    def _merge_consecutive_messages(self, messages: list[Message]) -> list[Message]:
        """Merge consecutive messages with same role"""
        if not messages:
            return []

        merged = []
        current = messages[0]

        for msg in messages[1:]:
            if msg.role == current.role and msg.role in ("user", "assistant"):
                # Merge by combining content
                if isinstance(current.content, str) and isinstance(msg.content, str):
                    current = Message(
                        role=current.role,
                        content=current.content + "\n" + msg.content,
                        tool_calls=msg.tool_calls or current.tool_calls,
                        metadata=msg.metadata or current.metadata,
                        identifier=msg.identifier or current.identifier,
                    )
                else:
                    # Complex merge for content blocks - don't merge, just return as is
                    merged.append(current)
                    current = msg
            else:
                merged.append(current)
                current = msg

        merged.append(current)
        return merged

    def _extract_text_from_content_blocks(self, blocks: list[dict[str, Any]]) -> str:
        """Extract text from content blocks"""
        texts = []
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts) if texts else ""

    def _extract_tool_calls_from_content(
        self, content: list[ContentBlock]
    ) -> list[ToolCall] | None:
        """Extract tool_calls from content blocks"""
        tool_calls = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        identifier=block.get("id", ""),
                        name=block.get("name", ""),
                        arguments=block.get("input", {}),
                    )
                )
        return tool_calls if tool_calls else None

    def _convert_anthropic_response_to_choice(
        self, content: list[dict[str, Any]], stop_reason: str
    ) -> Choice:
        """Convert Anthropic response content to unified Choice"""
        # Extract text and tool_calls from content blocks
        content_blocks = []
        tool_calls = []

        for block in content:
            block_type = block.get("type")
            if block_type == "text":
                content_blocks.append(
                    ContentBlock(
                        type="text",
                        text=block.get("text", ""),
                    )
                )
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        identifier=block.get("id", ""),
                        name=block.get("name", ""),
                        arguments=block.get("input", {}),
                    )
                )
            elif block_type == "thinking":
                # Anthropic extended thinking blocks
                content_blocks.append(
                    ContentBlock(
                        type="thinking",
                        thinking=Thinking(content=block.get("thinking", "")),
                    )
                )
            elif block_type == "redacted_thinking":
                # Redacted thinking blocks
                content_blocks.append(
                    ContentBlock(
                        type="redacted_thinking",
                        redacted_thinking=RedactedThinking(content="[redacted]"),
                    )
                )

        # Map finish reason
        unified_finish_reason = self.finish_reason_map.get(stop_reason, "stop")

        message = ResponseMessage(
            role="assistant",
            content=content_blocks if content_blocks else "",
            tool_calls=tool_calls if tool_calls else None,
        )

        return Choice(
            message=message,
            index=0,
            finish_reason=unified_finish_reason,
        )

    def _response_message_to_anthropic_content(
        self, message: ResponseMessage
    ) -> list[dict[str, Any]]:
        """Convert response message to Anthropic content blocks"""
        content = []

        # Add text content
        if isinstance(message.content, str):
            if message.content:
                content.append({"type": "text", "text": message.content})
        elif isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, ContentBlock):
                    # Compare against Enum value
                    if block.type == Type.text or block.type == "text":
                        if block.text:
                            content.append({"type": "text", "text": block.text})
                    elif block.type == Type.tool_result or block.type == "tool_result":
                        tool_result = block.tool_result or {}
                        tool_name = (
                            tool_result.tool_name
                            if hasattr(tool_result, "tool_name")
                            else tool_result.get("tool_name", "")
                        )
                        result_data = (
                            tool_result.result
                            if hasattr(tool_result, "result")
                            else tool_result.get("result", "")
                        )
                        content.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_name,
                                "content": result_data,
                            }
                        )
                    elif block.type == Type.thinking or block.type == "thinking":
                        thinking = block.thinking
                        thinking_content = (
                            thinking.content
                            if hasattr(thinking, "content")
                            else thinking.get("content", "")
                            if isinstance(thinking, dict)
                            else ""
                        )
                        content.append(
                            {
                                "type": "thinking",
                                "thinking": thinking_content,
                            }
                        )
                    elif (
                        block.type == Type.redacted_thinking
                        or block.type == "redacted_thinking"
                    ):
                        # Redacted thinking is output-only
                        content.append(
                            {
                                "type": "redacted_thinking",
                            }
                        )

        # Add tool_use blocks
        if message.tool_calls:
            for tc in message.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc.identifier or str(uuid.uuid4()),
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                )

        return content

    def _to_unified_stream_event_impl(
        self, data: dict[str, Any], sequence_id: int, timestamp: float
    ) -> StreamEvent:
        """Convert Anthropic stream event to unified IR format

        Anthropic uses different event types:
        - content_block_start: start of content block
        - content_block_delta: incremental text or tool input
        - content_block_stop: end of content block
        - message_start: start of message (contains model info)
        - message_stop: end of message
        """
        event_type = data.get("type", "")
        content_index = data.get("index", 0)

        # Handle content block events
        if event_type == "content_block_delta":
            delta = data.get("delta", {})
            delta_type = delta.get("type", "")

            if delta_type == "text_delta":
                # Convert to unified content_delta event
                return StreamEvent(
                    type="content_delta",
                    sequence_id=sequence_id,
                    timestamp=timestamp,
                    content_delta=delta.get("text", ""),
                    content_index=content_index,
                    metadata=data.get("metadata"),
                )
            elif delta_type == "input_json_delta":
                # Tool call input being streamed
                return StreamEvent(
                    type="tool_call_delta",
                    sequence_id=sequence_id,
                    timestamp=timestamp,
                    tool_call_delta={
                        "arguments_delta": delta.get("partial_json", ""),
                    },
                    content_index=content_index,
                    metadata=data.get("metadata"),
                )

        elif event_type == "content_block_start":
            # Just track the start - no content to emit yet
            content_block = data.get("content_block", {})
            cb_type = content_block.get("type", "text")

            return StreamEvent(
                type="metadata_update",
                sequence_id=sequence_id,
                timestamp=timestamp,
                content_index=content_index,
                metadata={
                    "event": "content_block_start",
                    "content_block_type": cb_type,
                },
            )

        elif event_type == "content_block_stop":
            # Mark end of content block
            return StreamEvent(
                type="content_finish",
                sequence_id=sequence_id,
                timestamp=timestamp,
                content_index=content_index,
                metadata={"event": "content_block_stop"},
            )

        elif event_type == "message_start":
            # Message start with model info
            message = data.get("message", {})
            return StreamEvent(
                type="metadata_update",
                sequence_id=sequence_id,
                timestamp=timestamp,
                metadata={
                    "event": "message_start",
                    "message_id": message.get("id"),
                    "model": message.get("model"),
                },
            )

        elif event_type == "message_stop":
            # Message complete
            return StreamEvent(
                type="stream_end",
                sequence_id=sequence_id,
                timestamp=timestamp,
                metadata={"event": "message_stop"},
            )

        # Default fallback
        return StreamEvent(
            type="metadata_update",
            sequence_id=sequence_id,
            timestamp=timestamp,
            metadata={"event": event_type, "raw": data},
        )

    def from_unified_stream_event(self, unified_event: StreamEvent) -> dict[str, Any]:
        """Convert unified IR stream event to Anthropic format

        Maps unified event types back to Anthropic event format
        """
        # Extract metadata if present
        metadata = unified_event.metadata or {}

        # Get event type as string for comparison (handle both enum and string)
        event_type = (
            unified_event.type.value
            if hasattr(unified_event.type, "value")
            else str(unified_event.type)
        )

        # Handle content_delta -> content_block_delta
        if event_type == "content_delta":
            return {
                "type": "content_block_delta",
                "index": unified_event.content_index or 0,
                "delta": {
                    "type": "text_delta",
                    "text": unified_event.content_delta or "",
                },
            }

        # Handle tool_call_delta
        elif event_type == "tool_call_delta":
            tc_delta = unified_event.tool_call_delta or {}
            # Handle both dict and Pydantic model for tool_call_delta
            arguments_delta = (
                tc_delta.arguments_delta
                if hasattr(tc_delta, "arguments_delta")
                else tc_delta.get("arguments_delta", "")
            )
            return {
                "type": "content_block_delta",
                "index": unified_event.content_index or 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": arguments_delta,
                },
            }

        # Handle content_finish -> content_block_stop
        elif event_type == "content_finish":
            return {
                "type": "content_block_stop",
                "index": unified_event.content_index or 0,
            }

        # Handle stream_end -> message_stop
        elif event_type == "stream_end":
            return {
                "type": "message_stop",
            }

        # Handle metadata_update based on embedded event type
        elif event_type == "metadata_update":
            event_name = metadata.get("event", "")

            if event_name == "content_block_start":
                return {
                    "type": "content_block_start",
                    "index": unified_event.content_index or 0,
                    "content_block": {
                        "type": metadata.get("content_block_type", "text"),
                    },
                }
            elif event_name == "message_start":
                return {
                    "type": "message_start",
                    "message": {
                        "id": metadata.get("message_id"),
                        "model": metadata.get("model"),
                    },
                }
            else:
                # Generic metadata event
                return {
                    "type": event_name or "metadata_update",
                    "metadata": metadata,
                }

        # Error event
        elif event_type == "error":
            error = unified_event.error or {}
            return {
                "type": "error",
                "error": {
                    "type": "error",
                    "message": error.get("message", "Unknown error"),
                },
            }

        # Default fallback
        return {
            "type": "metadata_update",
            "metadata": metadata,
        }

    def _add_beta_headers(self, request: dict[str, Any]) -> None:
        """Add beta headers based on features used in request

        Intelligently detects Anthropic beta features and adds corresponding
        headers. Headers track which beta APIs are required for the request.
        """
        headers = set()  # Use set to avoid duplicates

        # 1. Check for prompt caching (cache_control can appear in multiple places)
        if self._has_cache_control(request):
            headers.add("prompt-caching-2024-07-31")

        # 2. Check for thinking/extended thinking
        if "thinking" in request:
            headers.add("interleaved-thinking-2025-05-14")

        # 3. Check for advanced tool use
        if "tools" in request and request["tools"]:
            headers.add("advanced-tool-use-2025-11-20")

        # 4. Check for vision/multimodal content
        if self._has_vision_content(request):
            # Vision is not a beta feature anymore, but we can track it
            pass

        # 5. Check for specific tool features (computer, hosted, MCP, etc.)
        if self._has_advanced_tool_types(request):
            # Computer and other advanced tool use
            headers.add("advanced-tool-use-2025-11-20")

        # Convert set to sorted list for consistent output
        if headers:
            request["betas"] = sorted(list(headers))

    def _has_cache_control(self, request: dict[str, Any]) -> bool:
        """Check if request uses cache_control in any location"""
        # Check system-level cache control
        if "system" in request:
            system = request["system"]
            if isinstance(system, dict) and "cache_control" in system:
                return True
            # System could also be a list of content blocks
            elif isinstance(system, list):
                for block in system:
                    if isinstance(block, dict) and block.get("type") == "text":
                        if "cache_control" in block:
                            return True

        # Check message-level cache control
        if "messages" in request:
            for msg in request["messages"]:
                if isinstance(msg, dict) and "cache_control" in msg:
                    return True
                # Check content blocks for cache control
                if "content" in msg and isinstance(msg["content"], list):
                    for block in msg["content"]:
                        if isinstance(block, dict) and "cache_control" in block:
                            return True

        # Check tool definition cache control
        if "tools" in request:
            for tool in request["tools"]:
                if isinstance(tool, dict) and "cache_control" in tool:
                    return True

        return False

    def _has_vision_content(self, request: dict[str, Any]) -> bool:
        """Check if request contains vision/image content"""
        if "messages" in request:
            for msg in request["messages"]:
                if "content" in msg:
                    content = msg["content"]
                    # Check for image blocks in list format
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "image":
                                return True
                    # Check for image key in dict content (shouldn't happen but be safe)
                    elif isinstance(content, dict) and "image" in content:
                        return True

        return False

    def _has_advanced_tool_types(self, request: dict[str, Any]) -> bool:
        """Check if request uses advanced tool types (computer, hosted, MCP, etc.)

        Advanced tool types require the advanced-tool-use beta.
        """
        if "tools" in request:
            for tool in request["tools"]:
                if isinstance(tool, dict):
                    # Check for tool type field (future feature)
                    tool_type = tool.get("type")
                    if tool_type in ["computer", "hosted", "mcp", "tool_search"]:
                        return True

        return False

def _augment_message_start(message: dict[str, Any], model_name: str | None) -> dict[str, Any]:
    """Fill Anthropic message_start payload with common defaults."""
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

