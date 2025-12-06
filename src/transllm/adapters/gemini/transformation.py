"""Request/response transformation logic for Gemini adapter"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from .schema_converter import convert_json_schema_to_gemini
from .utils import (
    convert_image_url_to_gemini,
    merge_duplicate_messages,
    validate_gemini_request,
)


class GeminiRequestTransformer:
    """Transform requests between OpenAI and Gemini formats"""

    def __init__(self) -> None:
        pass

    def transform_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Transform OpenAI request to Gemini format

        Args:
            request: OpenAI-style request dictionary

        Returns:
            Gemini request dictionary
        """
        # Extract system instruction
        system_instruction = self._extract_system_instruction(request)

        # Transform contents (messages)
        contents = self._transform_messages(request.get("messages", []))

        # Build base request
        gemini_request: Dict[str, Any] = {
            "contents": contents
        }

        # Add system instruction if present
        if system_instruction:
            gemini_request["system_instruction"] = {
                "parts": [{"text": system_instruction}]
            }

        # Transform generation config
        generation_config = self._transform_generation_config(request)
        if generation_config:
            gemini_request["generationConfig"] = generation_config

        # Transform tools
        tools = self._transform_tools(request.get("tools", []))
        if tools:
            gemini_request["tools"] = tools

        # Transform tool config
        tool_config = self._transform_tool_config(request)
        if tool_config:
            gemini_request["toolConfig"] = tool_config

        # Transform safety settings
        safety_settings = self._transform_safety_settings(request)
        if safety_settings:
            gemini_request["safetySettings"] = safety_settings

        # Transform thinking config
        thinking_config = self._transform_thinking_config(request)
        if thinking_config:
            gemini_request["thinkingConfig"] = thinking_config

        # Validate final request
        validate_gemini_request(gemini_request)

        return gemini_request

    def _extract_system_instruction(self, request: Dict[str, Any]) -> Optional[str]:
        """Extract system instruction from messages or separate field"""
        # Check for explicit system_instruction field
        if "system_instruction" in request:
            return request["system_instruction"]

        # Extract from messages
        system_messages = [
            msg.get("content", "")
            for msg in request.get("messages", [])
            if msg.get("role") == "system"
        ]

        if system_messages:
            return "\n\n".join(system_messages)

        return None

    def _transform_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transform OpenAI messages to Gemini contents

        Args:
            messages: OpenAI messages list

        Returns:
            Gemini contents list
        """
        contents: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                # System messages handled separately
                continue

            # Build parts for this message
            parts = self._transform_content_to_parts(content, msg)

            # Map role
            gemini_role = self._map_role(role)
            if gemini_role:
                contents.append({
                    "role": gemini_role,
                    "parts": parts
                })

        # Merge duplicate messages with same role
        contents = merge_duplicate_messages(contents)

        return contents

    def _transform_content_to_parts(
        self,
        content: Union[str, List[Dict[str, Any]]],
        msg: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Transform message content to Gemini parts format

        Args:
            content: Message content (string or array)
            msg: Full message dictionary

        Returns:
            List of Gemini parts
        """
        parts: List[Dict[str, Any]] = []

        # Handle string content
        if isinstance(content, str):
            if content.strip():
                parts.append({"text": content})

        # Handle array content (multimodal)
        elif isinstance(content, list):
            for item in content:
                part_type = item.get("type")

                if part_type == "text":
                    text = item.get("text", "")
                    if text:
                        parts.append({"text": text})

                elif part_type == "image_url":
                    image_url = item.get("image_url", {})
                    url = image_url.get("url", "") if isinstance(image_url, dict) else image_url

                    if url:
                        detail = image_url.get("detail") if isinstance(image_url, dict) else None
                        gemini_part = convert_image_url_to_gemini(
                            url,
                            detail=detail
                        )
                        parts.append(gemini_part)

                elif part_type == "tool_result":
                    # Tool results are handled via function_response parts
                    tool_result = self._transform_tool_result(item)
                    if tool_result:
                        parts.append(tool_result)

        # Handle tool_calls
        tool_calls = msg.get("tool_calls", [])
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                function = tool_call.get("function", {})
                function_name = function.get("name")
                function_args = function.get("arguments")

                if function_name and function_args:
                    # Parse arguments
                    try:
                        args_dict = (
                            json.loads(function_args)
                            if isinstance(function_args, str)
                            else function_args
                        )
                    except (json.JSONDecodeError, TypeError):
                        args_dict = {}

                    parts.append({
                        "function_call": {
                            "name": function_name,
                            "args": args_dict
                        }
                    })

        return parts

    def _transform_tool_result(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transform tool result to Gemini function_response

        Args:
            item: Tool result item

        Returns:
            Gemini function_response part or None
        """
        content = item.get("content", "")

        # Determine if this is a tool result that needs ID
        tool_call_id = item.get("tool_call_id")

        # Extract function call info if available
        function_name = item.get("function_name") or item.get("tool_name")
        function_response = {
            "name": function_name or "unknown",
            "response": content
        }

        if tool_call_id:
            function_response["id"] = tool_call_id

        return {"function_response": function_response}

    def _map_role(self, role: Optional[str]) -> Optional[str]:
        """Map OpenAI role to Gemini role

        Args:
            role: OpenAI role

        Returns:
            Gemini role or None
        """
        role_mapping = {
            "user": "user",
            "assistant": "model",
            "system": None,  # Handled separately
            "tool": None,  # Tool results become function_response
        }
        return role_mapping.get(role)

    def _transform_generation_config(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Transform generation parameters

        Args:
            request: OpenAI request

        Returns:
            Gemini generation config
        """
        config: Dict[str, Any] = {}

        # Temperature
        if "temperature" in request:
            config["temperature"] = request["temperature"]

        # Top P
        if "top_p" in request:
            config["topP"] = request["top_p"]

        # Top K
        if "top_k" in request:
            config["topK"] = request["top_k"]

        # Max Output Tokens
        if "max_tokens" in request:
            config["maxOutputTokens"] = request["max_tokens"]

        # Stop Sequences
        if "stop" in request:
            stop = request["stop"]
            if isinstance(stop, str):
                config["stopSequences"] = [stop]
            elif isinstance(stop, list):
                config["stopSequences"] = stop

        # Response MIME Type (for JSON mode)
        if "response_format" in request:
            response_format = request["response_format"]
            if isinstance(response_format, dict) and "type" in response_format:
                if response_format["type"] == "json_object":
                    config["responseMimeType"] = "application/json"

        return config

    def _transform_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transform OpenAI tools to Gemini tools

        Args:
            tools: OpenAI tools list

        Returns:
            Gemini tools list
        """
        if not tools:
            return []

        gemini_tools: List[Dict[str, Any]] = []

        for tool in tools:
            if tool.get("type") == "function":
                function = tool["function"]

                # Convert parameters to Gemini schema
                parameters = function.get("parameters", {})
                if parameters:
                    # Convert JSON Schema to Gemini Schema
                    gemini_parameters = convert_json_schema_to_gemini(parameters)
                else:
                    gemini_parameters = {}

                gemini_tools.append({
                    "function_declarations": [{
                        "name": function["name"],
                        "description": function.get("description", ""),
                        "parameters": gemini_parameters
                    }]
                })

        return gemini_tools

    def _transform_tool_config(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transform tool configuration

        Args:
            request: OpenAI request

        Returns:
            Gemini tool config or None
        """
        # Tool choice configuration
        tool_choice = request.get("tool_choice")

        if not tool_choice:
            return None

        if tool_choice == "auto":
            return {"function_calling_config": {"mode": "ANY"}}
        elif tool_choice == "none":
            return {"function_calling_config": {"mode": "NONE"}}
        elif isinstance(tool_choice, dict) and "function" in tool_choice:
            function_name = tool_choice["function"].get("name")
            if function_name:
                return {
                    "function_calling_config": {
                        "mode": "SPECIFIC",
                        "allowed_function_names": [function_name]
                    }
                }

        return None

    def _transform_safety_settings(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Transform safety settings

        Args:
            request: OpenAI request

        Returns:
            Gemini safety settings list
        """
        # Simplified implementation
        # In practice, this would map OpenAI's safety settings to Gemini's
        return []

    def _transform_thinking_config(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transform thinking configuration for Gemini 2.x/3.x

        Args:
            request: OpenAI request

        Returns:
            Gemini thinking config or None
        """
        # Check for reasoning_effort (OpenAI)
        reasoning_effort = request.get("reasoning_effort")

        if not reasoning_effort:
            return None

        # Map to thinking config based on effort level
        effort_lower = reasoning_effort.lower()

        if effort_lower == "low":
            # Google AI Studio uses thinkingBudget + includeThoughts
            return {
                "thinkingBudget": 8000,
                "includeThoughts": True
            }
        elif effort_lower == "medium":
            return {
                "thinkingBudget": 16000,
                "includeThoughts": True
            }
        elif effort_lower == "high":
            return {
                "thinkingBudget": 32000,
                "includeThoughts": True
            }

        return None
