"""Gemini idempotency tests

Tests verify that Gemini → IR → Gemini roundtrip conversions
preserve basic features including messages, tools, and multimodal content.
"""

from src.transllm.adapters.gemini import GeminiAdapter
from src.transllm.core.schema import Role, FinishReason, Type


class TestGeminiBasicIdempotency:
    """Test 1: Basic chat request/response roundtrip"""

    def setup_method(self):
        self.adapter = GeminiAdapter()

    def test_chat_request_roundtrip(self):
        """Gemini chat request should survive roundtrip conversion"""
        gemini_req = {
            "model": "gemini-1.5-pro",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": "Hello, how are you?"
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 100,
            }
        }

        # Gemini → IR
        unified = self.adapter.to_unified_request(gemini_req)

        # Verify system instruction extracted
        assert unified.messages[0].role == Role.user
        assert "Hello" in unified.messages[0].content

        # Verify generation parameters
        assert unified.generation_params.temperature == 0.7
        assert unified.generation_params.max_tokens == 100

        # IR → Gemini
        gemini_req2 = self.adapter.from_unified_request(unified)

        # Verify request structure preserved
        assert "contents" in gemini_req2
        assert "generationConfig" in gemini_req2
        assert gemini_req2["generationConfig"]["temperature"] == 0.7

    def test_chat_response_roundtrip(self):
        """Gemini chat response should survive roundtrip conversion"""
        gemini_resp = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "Hello! I'm doing well, thank you. How can I help you today?"
                            }
                        ],
                        "role": "model"
                    },
                    "finishReason": "stop",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 20,
                "candidatesTokenCount": 12,
                "totalTokenCount": 32,
            }
        }

        # Gemini → IR
        unified = self.adapter.to_unified_response(gemini_resp)

        # Verify message content
        assert unified.choices[0].message.content == "Hello! I'm doing well, thank you. How can I help you today?"
        assert unified.choices[0].finish_reason == FinishReason.stop

        # Verify usage
        assert unified.usage.prompt_tokens == 20
        assert unified.usage.completion_tokens == 12
        assert unified.usage.total_tokens == 32

        # IR → Gemini
        gemini_resp2 = self.adapter.from_unified_response(unified)

        # Verify response structure preserved
        assert "candidates" in gemini_resp2
        assert len(gemini_resp2["candidates"]) == 1
        assert gemini_resp2["candidates"][0]["finishReason"] == "stop"


class TestGeminiToolsIdempotency:
    """Test 2: Tool calling request/response preservation"""

    def setup_method(self):
        self.adapter = GeminiAdapter()

    def test_tool_request_roundtrip(self):
        """Gemini tool request should survive roundtrip conversion"""
        gemini_req = {
            "model": "gemini-1.5-pro",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": "What's the weather like in Beijing?"
                        }
                    ]
                }
            ],
            "tools": [
                {
                    "function_declarations": [
                        {
                            "name": "get_weather",
                            "description": "Get weather information",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "City name"
                                    }
                                },
                                "required": ["location"]
                            }
                        }
                    ]
                }
            ],
            "toolConfig": {
                "function_calling_config": {
                    "mode": "ANY"
                }
            }
        }

        # Gemini → IR
        unified = self.adapter.to_unified_request(gemini_req)

        # Verify tools
        assert unified.tools is not None
        assert len(unified.tools) == 1
        assert unified.tools[0].name == "get_weather"
        assert "location" in unified.tools[0].parameters["properties"]

        # IR → Gemini
        gemini_req2 = self.adapter.from_unified_request(unified)

        # Verify tool structure preserved
        assert "tools" in gemini_req2
        assert "toolConfig" in gemini_req2

    def test_tool_call_response_roundtrip(self):
        """Gemini tool call response should survive roundtrip conversion"""
        gemini_resp = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "function_call": {
                                    "name": "get_weather",
                                    "args": {
                                        "location": "Beijing"
                                    }
                                }
                            }
                        ],
                        "role": "model"
                    },
                    "finishReason": "tool_calls",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 25,
                "candidatesTokenCount": 10,
                "totalTokenCount": 35,
            }
        }

        # Gemini → IR
        unified = self.adapter.to_unified_response(gemini_resp)

        # Verify tool calls
        assert unified.choices[0].message.tool_calls is not None
        assert len(unified.choices[0].message.tool_calls) == 1
        assert unified.choices[0].message.tool_calls[0].name == "get_weather"
        assert unified.choices[0].finish_reason == FinishReason.tool_calls

        # IR → Gemini
        gemini_resp2 = self.adapter.from_unified_response(unified)

        # Verify tool call preserved
        assert "candidates" in gemini_resp2
        assert "function_call" in gemini_resp2["candidates"][0]["content"]["parts"][0]
        assert gemini_resp2["candidates"][0]["finishReason"] == "tool_calls"


class TestGeminiMultimodalIdempotency:
    """Test 3: Multimodal content (images) preservation"""

    def setup_method(self):
        self.adapter = GeminiAdapter()

    def test_image_content_roundtrip(self):
        """Gemini image content should survive roundtrip conversion"""
        gemini_req = {
            "model": "gemini-1.5-pro-vision",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": "What do you see in this image?"
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.5,
            }
        }

        # Gemini → IR
        unified = self.adapter.to_unified_request(gemini_req)

        # Verify multimodal content
        assert isinstance(unified.messages[0].content, list)
        assert len(unified.messages[0].content) == 2
        assert unified.messages[0].content[0].type == Type.text
        assert unified.messages[0].content[1].type == Type.image_url
        assert unified.messages[0].content[1].image_url.url.startswith("data:")

        # IR → Gemini
        gemini_req2 = self.adapter.from_unified_request(unified)

        # Verify image preserved
        assert "contents" in gemini_req2
        assert len(gemini_req2["contents"][0]["parts"]) == 2
        assert "inline_data" in gemini_req2["contents"][0]["parts"][1]
        assert gemini_req2["contents"][0]["parts"][1]["inline_data"]["mime_type"] == "image/jpeg"


class TestGeminiSystemInstructionIdempotency:
    """Test 4: System instruction preservation"""

    def setup_method(self):
        self.adapter = GeminiAdapter()

    def test_system_instruction_roundtrip(self):
        """Gemini system instruction should survive roundtrip conversion"""
        gemini_req = {
            "model": "gemini-1.5-pro",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": "Hello"
                        }
                    ]
                }
            ],
            "system_instruction": {
                "parts": [
                    {
                        "text": "You are a helpful assistant."
                    }
                ]
            }
        }

        # Gemini → IR
        unified = self.adapter.to_unified_request(gemini_req)

        # Verify system instruction
        # Note: System instruction is handled separately in the transformation
        # IR → Gemini
        gemini_req2 = self.adapter.from_unified_request(unified)

        # Verify system instruction preserved
        assert "system_instruction" in gemini_req2
        assert "parts" in gemini_req2["system_instruction"]


class TestGeminiJSONModeIdempotency:
    """Test 5: JSON mode (structured output) preservation"""

    def setup_method(self):
        self.adapter = GeminiAdapter()

    def test_json_mode_roundtrip(self):
        """Gemini JSON mode should survive roundtrip conversion"""
        gemini_req = {
            "model": "gemini-1.5-pro",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": "Extract the name and age"
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "maxOutputTokens": 100,
            }
        }

        # Gemini → IR
        unified = self.adapter.to_unified_request(gemini_req)

        # Verify response format info
        # Note: IR doesn't directly store responseMimeType, but it can be inferred
        # from generation parameters

        # IR → Gemini
        gemini_req2 = self.adapter.from_unified_request(unified)

        # Verify JSON mode preserved
        assert "generationConfig" in gemini_req2
        assert "responseMimeType" in gemini_req2["generationConfig"]
        assert gemini_req2["generationConfig"]["responseMimeType"] == "application/json"


class TestGeminiThinkingBlocksIdempotency:
    """Test 6: Thinking blocks (Gemini 2.x/3.x) preservation"""

    def setup_method(self):
        self.adapter = GeminiAdapter(model="gemini-2.0-flash-exp")

    def test_thinking_config_roundtrip(self):
        """Gemini thinking config should survive roundtrip conversion"""
        gemini_req = {
            "model": "gemini-2.0-flash-exp",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": "What is 2+2?"
                        }
                    ]
                }
            ],
            "thinkingConfig": {
                "thinkingBudget": 8000,
                "includeThoughts": True
            }
        }

        # Gemini → IR
        unified = self.adapter.to_unified_request(gemini_req)

        # Note: Thinking config is stored in generation parameters metadata
        # IR → Gemini
        gemini_req2 = self.adapter.from_unified_request(unified)

        # Verify thinking config preserved
        assert "thinkingConfig" in gemini_req2
        assert gemini_req2["thinkingConfig"]["includeThoughts"] is True

    def test_thinking_response_roundtrip(self):
        """Gemini response with thinking tokens should survive roundtrip"""
        gemini_resp = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "4"
                            }
                        ],
                        "role": "model"
                    },
                    "finishReason": "stop",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
                "thoughtsTokenCount": 20
            }
        }

        # Gemini → IR
        unified = self.adapter.to_unified_response(gemini_resp)

        # Verify thinking tokens
        assert unified.usage.reasoning_tokens == 20

        # IR → Gemini
        gemini_resp2 = self.adapter.from_unified_response(unified)

        # Verify thinking tokens preserved
        assert "usageMetadata" in gemini_resp2
        assert gemini_resp2["usageMetadata"]["thoughtsTokenCount"] == 20


class TestGeminiStopSequencesIdempotency:
    """Test 7: Stop sequences preservation"""

    def setup_method(self):
        self.adapter = GeminiAdapter()

    def test_stop_sequences_roundtrip(self):
        """Gemini stop sequences should survive roundtrip conversion"""
        gemini_req = {
            "model": "gemini-1.5-pro",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": "Count to 5"
                        }
                    ]
                }
            ],
            "generationConfig": {
                "stopSequences": ["3"]
            }
        }

        # Gemini → IR
        unified = self.adapter.to_unified_request(gemini_req)

        # Verify stop sequences
        assert unified.generation_params.stop_sequences == ["3"]

        # IR → Gemini
        gemini_req2 = self.adapter.from_unified_request(unified)

        # Verify stop sequences preserved
        assert "generationConfig" in gemini_req2
        assert "stopSequences" in gemini_req2["generationConfig"]
        assert gemini_req2["generationConfig"]["stopSequences"] == ["3"]


class TestGeminiEmptyResponseIdempotency:
    """Test 8: Empty response handling"""

    def setup_method(self):
        self.adapter = GeminiAdapter()

    def test_empty_response_roundtrip(self):
        """Gemini empty response should survive roundtrip conversion"""
        gemini_resp = {
            "candidates": [
                {
                    "content": {
                        "parts": [],
                        "role": "model"
                    },
                    "finishReason": "stop",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 0,
                "totalTokenCount": 5,
            }
        }

        # Gemini → IR
        unified = self.adapter.to_unified_response(gemini_resp)

        # Verify empty content handled gracefully
        assert unified.choices[0].message.content == ""

        # IR → Gemini
        gemini_resp2 = self.adapter.from_unified_response(unified)

        # Verify empty response preserved
        assert "candidates" in gemini_resp2
        assert len(gemini_resp2["candidates"][0]["content"]["parts"]) == 0
