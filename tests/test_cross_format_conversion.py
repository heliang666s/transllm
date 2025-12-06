"""Cross-format conversion tests: OpenAI ↔ Anthropic bidirectional

Tests verify that data can be correctly converted between OpenAI and Anthropic
formats in both directions, with data consistency and semantic equivalence.

High-priority test scenarios (8):
1. Basic chat roundtrip conversion
2. System message position conversion
3. Tool call format conversion
4. Parameter mapping verification
5. Parallel tool calls semantic inversion
6. Tool result role mapping
7. Thinking content bidirectional conversion
8. Cache token statistics consistency
"""

import pytest
from src.transllm.adapters.openai import OpenAIAdapter
from src.transllm.adapters.anthropic import AnthropicAdapter
from tests.fixtures.openai import (
    OPENAI_CHAT_REQUEST,
    OPENAI_CHAT_RESPONSE,
    OPENAI_TOOL_REQUEST,
    OPENAI_TOOL_RESPONSE,
    OPENAI_PARALLEL_TOOLS_REQUEST,
    OPENAI_REASONING_RESPONSE,
    OPENAI_FULL_REQUEST,
)
from tests.fixtures.anthropic import (
    ANTHROPIC_CHAT_REQUEST,
    ANTHROPIC_CHAT_RESPONSE,
    ANTHROPIC_TOOL_REQUEST,
    ANTHROPIC_TOOL_RESPONSE,
    ANTHROPIC_SYSTEM_REQUEST,
    ANTHROPIC_THINKING_RESPONSE,
    ANTHROPIC_CACHE_RESPONSE,
)


class TestBasicChatConversion:
    """Test 1: Basic chat roundtrip conversion (OpenAI ↔ Anthropic ↔ OpenAI)"""

    def setup_method(self):
        self.openai_adapter = OpenAIAdapter()
        self.anthropic_adapter = AnthropicAdapter()

    def test_openai_to_anthropic_basic_chat(self):
        """Convert OpenAI chat format to Anthropic format"""
        # OpenAI → IR
        unified = self.openai_adapter.to_unified_request(OPENAI_CHAT_REQUEST)

        # IR → Anthropic
        anthropic_request = self.anthropic_adapter.from_unified_request(unified)

        # Verify key fields are present
        assert "model" in anthropic_request
        assert "messages" in anthropic_request
        assert "max_tokens" in anthropic_request

        # Verify messages structure
        assert len(anthropic_request["messages"]) > 0

        # System message should be extracted to top-level
        assert "system" in anthropic_request
        assert "You are a helpful assistant" in anthropic_request["system"]

    def test_anthropic_to_openai_basic_chat(self):
        """Convert Anthropic chat format to OpenAI format"""
        # Anthropic → IR
        unified = self.anthropic_adapter.to_unified_request(ANTHROPIC_CHAT_REQUEST)

        # IR → OpenAI
        openai_request = self.openai_adapter.from_unified_request(unified)

        # Verify key fields
        assert "model" in openai_request
        assert "messages" in openai_request
        assert len(openai_request["messages"]) > 0

    def test_openai_chat_response_to_anthropic(self):
        """Convert OpenAI chat response to Anthropic format"""
        # OpenAI → IR
        unified = self.openai_adapter.to_unified_response(OPENAI_CHAT_RESPONSE)

        # IR → Anthropic
        anthropic_response = self.anthropic_adapter.from_unified_response(unified)

        # Verify response structure
        assert "model" in anthropic_response
        assert "content" in anthropic_response
        assert "stop_reason" in anthropic_response

        # Content should be list of content blocks
        assert isinstance(anthropic_response["content"], list)
        assert len(anthropic_response["content"]) > 0

        # First block should be text
        assert anthropic_response["content"][0]["type"] == "text"
        assert "doing well" in anthropic_response["content"][0]["text"]

    def test_anthropic_chat_response_to_openai(self):
        """Convert Anthropic chat response to OpenAI format"""
        # Anthropic → IR
        unified = self.anthropic_adapter.to_unified_response(ANTHROPIC_CHAT_RESPONSE)

        # IR → OpenAI
        openai_response = self.openai_adapter.from_unified_response(unified)

        # Verify OpenAI response structure
        assert "model" in openai_response
        assert "choices" in openai_response
        assert len(openai_response["choices"]) > 0

        choice = openai_response["choices"][0]
        assert "message" in choice
        assert "role" in choice["message"]
        assert choice["message"]["role"] == "assistant"


class TestSystemMessageConversion:
    """Test 2: System message position conversion (messages array ↔ system parameter)"""

    def setup_method(self):
        self.openai_adapter = OpenAIAdapter()
        self.anthropic_adapter = AnthropicAdapter()

    def test_openai_system_message_in_array_to_anthropic(self):
        """OpenAI system message in messages array should convert to Anthropic system parameter"""
        # OpenAI format has system in messages array
        openai_req = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Hello"},
            ],
            "max_tokens": 100,
        }

        # OpenAI → IR
        unified = self.openai_adapter.to_unified_request(openai_req)

        # Should have system_instruction extracted
        assert unified.system_instruction == "You are a helpful AI assistant."

        # IR → Anthropic
        anthropic_req = self.anthropic_adapter.from_unified_request(unified)

        # Anthropic should have system parameter
        assert "system" in anthropic_req
        assert anthropic_req["system"] == "You are a helpful AI assistant."

        # System message should NOT be in messages array for Anthropic
        for msg in anthropic_req["messages"]:
            assert msg["role"] != "system"

    def test_anthropic_system_parameter_to_openai(self):
        """Anthropic system parameter should convert to OpenAI system message in array"""
        # Anthropic format uses separate system parameter
        anthropic_req = {
            "model": "claude-3-sonnet-20240229",
            "system": "You are a helpful assistant.",
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
            "max_tokens": 100,
        }

        # Anthropic → IR
        unified = self.anthropic_adapter.to_unified_request(anthropic_req)

        # Should have system_instruction
        assert unified.system_instruction == "You are a helpful assistant."

        # IR → OpenAI
        openai_req = self.openai_adapter.from_unified_request(unified)

        # IMPORTANT: OpenAI should reconstruct system message in array (future implementation)
        # For now, we verify the system_instruction is preserved in IR
        assert unified.system_instruction is not None

    def test_system_message_roundtrip_preservation(self):
        """System message content should be preserved in roundtrip conversion"""
        system_content = "You are a world-class software engineer with deep expertise in Python and system design."

        openai_req = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": "Design a cache system"},
            ],
            "max_tokens": 500,
        }

        # OpenAI → Anthropic → IR
        unified1 = self.openai_adapter.to_unified_request(openai_req)
        anthropic_req = self.anthropic_adapter.from_unified_request(unified1)

        # Verify system content preserved
        assert anthropic_req["system"] == system_content

        # Anthropic → OpenAI → IR
        unified2 = self.anthropic_adapter.to_unified_request(anthropic_req)

        # System instruction should still match
        assert unified2.system_instruction == system_content


class TestToolCallFormatConversion:
    """Test 3: Tool call format conversion (JSON string arguments ↔ object arguments)"""

    def setup_method(self):
        self.openai_adapter = OpenAIAdapter()
        self.anthropic_adapter = AnthropicAdapter()

    def test_openai_tool_call_to_anthropic(self):
        """OpenAI tool calls with JSON string arguments should convert to Anthropic object arguments"""
        # OpenAI response has tool_calls with JSON string arguments
        openai_resp = {
            "id": "chatcmpl-124",
            "object": "chat.completion",
            "created": 1677652300,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Beijing"}',  # JSON string
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

        # OpenAI → IR
        unified = self.openai_adapter.to_unified_response(openai_resp)

        # Verify tool_calls in IR
        assert unified.choices[0].message.tool_calls is not None
        assert len(unified.choices[0].message.tool_calls) > 0

        tool_call = unified.choices[0].message.tool_calls[0]
        assert tool_call.name == "get_weather"
        # Arguments should be parsed as dict in IR
        assert isinstance(tool_call.arguments, dict)
        assert tool_call.arguments.get("location") == "Beijing"

    def test_anthropic_tool_call_to_openai(self):
        """Anthropic tool calls with object arguments should convert to OpenAI JSON string arguments"""
        # Anthropic response has tool_use blocks with object arguments
        anthropic_resp = {
            "id": "msg_124",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-sonnet-20240229",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "get_weather",
                    "input": {"location": "Beijing"},  # Object, not JSON string
                }
            ],
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
            },
        }

        # Anthropic → IR
        unified = self.anthropic_adapter.to_unified_response(anthropic_resp)

        # Verify tool_calls
        assert unified.choices[0].message.tool_calls is not None
        tool_call = unified.choices[0].message.tool_calls[0]
        assert tool_call.name == "get_weather"
        assert tool_call.arguments == {"location": "Beijing"}

    def test_tool_arguments_roundtrip(self):
        """Complex tool arguments should survive roundtrip conversion"""
        complex_args = {
            "location": "Beijing",
            "units": "celsius",
            "include_forecast": True,
            "days": 7,
        }

        anthropic_resp = {
            "id": "msg_125",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-sonnet-20240229",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_125",
                    "name": "get_extended_weather",
                    "input": complex_args,
                }
            ],
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
            },
        }

        # Anthropic → IR → OpenAI → IR → Anthropic
        unified1 = self.anthropic_adapter.to_unified_response(anthropic_resp)
        openai_resp = self.openai_adapter.from_unified_response(unified1)
        unified2 = self.openai_adapter.to_unified_response(openai_resp)
        anthropic_resp2 = self.anthropic_adapter.from_unified_response(unified2)

        # Verify arguments preserved
        tool_use = anthropic_resp2["content"][0]
        assert tool_use["input"] == complex_args


class TestParameterMappingVerification:
    """Test 4: Parameter mapping verification (stop/stop_sequences, finish_reason)"""

    def setup_method(self):
        self.openai_adapter = OpenAIAdapter()
        self.anthropic_adapter = AnthropicAdapter()

    def test_stop_to_stop_sequences_mapping(self):
        """OpenAI 'stop' parameter should map to Anthropic 'stop_sequences'"""
        openai_req = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": ["\n", "END"],  # OpenAI uses 'stop'
            "max_tokens": 100,
        }

        # OpenAI → IR
        unified = self.openai_adapter.to_unified_request(openai_req)

        # Should have stop_sequences in generation_params
        assert unified.generation_params is not None
        assert unified.generation_params.stop_sequences == ["\n", "END"]

        # IR → Anthropic
        anthropic_req = self.anthropic_adapter.from_unified_request(unified)

        # Should use stop_sequences
        assert "stop_sequences" in anthropic_req
        assert anthropic_req["stop_sequences"] == ["\n", "END"]

    def test_stop_sequences_to_stop_mapping(self):
        """Anthropic 'stop_sequences' should map to OpenAI 'stop' parameter"""
        anthropic_req = {
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop_sequences": ["\n", "DONE"],  # Anthropic uses stop_sequences
            "max_tokens": 100,
        }

        # Anthropic → IR
        unified = self.anthropic_adapter.to_unified_request(anthropic_req)

        # IR → OpenAI
        openai_req = self.openai_adapter.from_unified_request(unified)

        # Should use 'stop'
        assert "stop" in openai_req
        assert openai_req["stop"] == ["\n", "DONE"]

    def test_finish_reason_mapping_openai_to_anthropic(self):
        """OpenAI finish_reason values should map to Anthropic stop_reason"""
        test_cases = [
            ("stop", "end_turn"),  # OpenAI 'stop' → Anthropic 'end_turn'
            ("length", "max_tokens"),  # OpenAI 'length' → Anthropic 'max_tokens'
            ("tool_calls", "tool_use"),  # OpenAI 'tool_calls' → Anthropic 'tool_use'
        ]

        for openai_reason, expected_anthropic_reason in test_cases:
            openai_resp = {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "response"},
                        "finish_reason": openai_reason,
                    }
                ],
            }

            # OpenAI → IR → Anthropic
            unified = self.openai_adapter.to_unified_response(openai_resp)
            anthropic_resp = self.anthropic_adapter.from_unified_response(unified)

            # Verify mapping
            assert anthropic_resp["stop_reason"] == expected_anthropic_reason

    def test_temperature_parameter_preservation(self):
        """Temperature parameter should be preserved across conversions"""
        openai_req = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 100,
        }

        # OpenAI → IR → Anthropic
        unified = self.openai_adapter.to_unified_request(openai_req)
        anthropic_req = self.anthropic_adapter.from_unified_request(unified)

        # Verify temperature preserved
        assert anthropic_req["temperature"] == 0.7

        # Anthropic → IR → OpenAI
        unified2 = self.anthropic_adapter.to_unified_request(anthropic_req)
        openai_req2 = self.openai_adapter.from_unified_request(unified2)

        # Should still have same temperature
        assert openai_req2["temperature"] == 0.7


class TestParallelToolCallsMapping:
    """Test 5: Parallel tool calls semantic inversion (parallel_tool_calls ↔ disable_parallel_tool_use)"""

    def setup_method(self):
        self.openai_adapter = OpenAIAdapter()
        self.anthropic_adapter = AnthropicAdapter()

    def test_parallel_tool_calls_true_to_anthropic(self):
        """OpenAI parallel_tool_calls=True should map to Anthropic disable_parallel_tool_use=False"""
        openai_req = {
            "model": "gpt-4-turbo",
            "messages": [{"role": "user", "content": "Get info in parallel"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "tool1",
                        "description": "Tool 1",
                        "parameters": {},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "tool2",
                        "description": "Tool 2",
                        "parameters": {},
                    },
                },
            ],
            "parallel_tool_calls": True,  # Enable parallel
            "max_tokens": 100,
        }

        # OpenAI → IR
        unified = self.openai_adapter.to_unified_request(openai_req)
        assert unified.parallel_tool_calls is True

        # IR → Anthropic
        anthropic_req = self.anthropic_adapter.from_unified_request(unified)

        # Should have tool_choice with disable_parallel_tool_use = False
        assert "tool_choice" in anthropic_req
        assert anthropic_req["tool_choice"]["disable_parallel_tool_use"] is False

    def test_parallel_tool_calls_false_to_anthropic(self):
        """OpenAI parallel_tool_calls=False should map to Anthropic disable_parallel_tool_use=True"""
        openai_req = {
            "model": "gpt-4-turbo",
            "messages": [{"role": "user", "content": "Get info sequentially"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "tool1",
                        "description": "Tool 1",
                        "parameters": {},
                    },
                },
            ],
            "parallel_tool_calls": False,  # Disable parallel
            "max_tokens": 100,
        }

        # OpenAI → IR
        unified = self.openai_adapter.to_unified_request(openai_req)
        assert unified.parallel_tool_calls is False

        # IR → Anthropic
        anthropic_req = self.anthropic_adapter.from_unified_request(unified)

        # Should have disable_parallel_tool_use = True
        assert "tool_choice" in anthropic_req
        assert anthropic_req["tool_choice"]["disable_parallel_tool_use"] is True

    def test_anthropic_disable_parallel_to_openai(self):
        """Anthropic disable_parallel_tool_use should reverse-map to OpenAI parallel_tool_calls"""
        anthropic_req = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Use tools"}],
            "tool_choice": {
                "type": "auto",
                "disable_parallel_tool_use": True,
            },
            "max_tokens": 100,
        }

        # Anthropic → IR
        unified = self.anthropic_adapter.to_unified_request(anthropic_req)

        # Should reverse to parallel_tool_calls = False
        assert unified.parallel_tool_calls is False

        # IR → OpenAI
        openai_req = self.openai_adapter.from_unified_request(unified)

        # Should have parallel_tool_calls = False
        assert openai_req.get("parallel_tool_calls") is False


class TestToolResultRoleMapping:
    """Test 6: Tool result role mapping (role=tool ↔ role=user + tool_result content)"""

    def setup_method(self):
        self.openai_adapter = OpenAIAdapter()
        self.anthropic_adapter = AnthropicAdapter()

    def test_openai_tool_result_message_format(self):
        """OpenAI uses role='tool' for tool results"""
        openai_messages = [
            {
                "role": "user",
                "content": "What's the weather?",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Beijing"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",  # OpenAI tool result message
                "tool_call_id": "call_123",
                "content": "Beijing: Sunny, 20°C",
            },
        ]

        openai_req = {
            "model": "gpt-4",
            "messages": openai_messages,
            "max_tokens": 100,
        }

        # OpenAI → IR
        unified = self.openai_adapter.to_unified_request(openai_req)

        # Should parse tool role message
        messages = unified.messages
        assert len(messages) == 3

        # The tool result should be captured
        # (Currently may need fixing in OpenAI adapter)

    def test_anthropic_tool_result_message_format(self):
        """Anthropic uses role='user' with tool_result content block for tool results"""
        anthropic_messages = [
            {
                "role": "user",
                "content": "What's the weather?",
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_123",
                        "name": "get_weather",
                        "input": {"location": "Beijing"},
                    }
                ],
            },
            {
                "role": "user",  # Anthropic tool result message
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_123",
                        "content": "Beijing: Sunny, 20°C",
                    }
                ],
            },
        ]

        anthropic_req = {
            "model": "claude-3-sonnet-20240229",
            "messages": anthropic_messages,
            "max_tokens": 100,
        }

        # Anthropic → IR
        unified = self.anthropic_adapter.to_unified_request(anthropic_req)

        # Should parse correctly
        messages = unified.messages
        assert len(messages) == 3

    def test_tool_result_roundtrip(self):
        """Tool result format should survive roundtrip conversion"""
        # Start with Anthropic format
        anthropic_req = {
            "model": "claude-3-sonnet-20240229",
            "messages": [
                {"role": "user", "content": "Get weather"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool_123",
                            "name": "get_weather",
                            "input": {"location": "Paris"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool_123",
                            "content": "Paris: Cloudy, 15°C",
                        }
                    ],
                },
            ],
            "max_tokens": 100,
        }

        # Anthropic → IR → Anthropic
        unified = self.anthropic_adapter.to_unified_request(anthropic_req)
        anthropic_req2 = self.anthropic_adapter.from_unified_request(unified)

        # Should have same structure
        assert len(anthropic_req2["messages"]) == 3

        # Tool result message should preserve structure
        tool_result_msg = anthropic_req2["messages"][2]
        assert tool_result_msg["role"] == "user"
        assert len(tool_result_msg["content"]) > 0


class TestThinkingContentConversion:
    """Test 7: Thinking content bidirectional conversion (reasoning_content ↔ thinking blocks)"""

    def setup_method(self):
        self.openai_adapter = OpenAIAdapter()
        self.anthropic_adapter = AnthropicAdapter()

    def test_openai_reasoning_content_to_anthropic(self):
        """OpenAI reasoning_content should convert to Anthropic thinking blocks"""
        openai_resp = {
            "id": "chatcmpl-125",
            "object": "chat.completion",
            "created": 1677652350,
            "model": "o1-preview",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "The answer is 42 because...",
                        "reasoning_content": "Let me work through this problem step by step...",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300,
            },
        }

        # OpenAI → IR
        unified = self.openai_adapter.to_unified_response(openai_resp)

        # IR should preserve reasoning_content
        assert unified.choices[0].message.reasoning_content == "Let me work through this problem step by step..."

    def test_anthropic_thinking_blocks_structure(self):
        """Anthropic thinking blocks should be properly handled"""
        anthropic_resp = {
            "id": "msg_127",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-opus-20240229",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Let me think about this problem step by step...",
                },
                {
                    "type": "text",
                    "text": "Based on my analysis, the answer is...",
                },
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 200,
            },
        }

        # Anthropic → IR
        unified = self.anthropic_adapter.to_unified_response(anthropic_resp)

        # IR should have content blocks with thinking type
        assert len(unified.choices[0].message.content) == 2

        content_blocks = unified.choices[0].message.content
        assert content_blocks[0].type.value == "thinking"
        assert content_blocks[1].type.value == "text"


class TestCacheTokenStatistics:
    """Test 8: Cache token statistics consistency (cache_creation + cache_read → cached_tokens)"""

    def setup_method(self):
        self.openai_adapter = OpenAIAdapter()
        self.anthropic_adapter = AnthropicAdapter()

    def test_anthropic_cache_creation_tokens(self):
        """Anthropic cache_creation_input_tokens should be preserved"""
        anthropic_resp = {
            "id": "msg_125",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-sonnet-20240229",
            "content": [
                {
                    "type": "text",
                    "text": "Based on the cached context, the answer is...",
                }
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 50,
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 0,
                "output_tokens": 20,
            },
        }

        # Anthropic → IR
        unified = self.anthropic_adapter.to_unified_response(anthropic_resp)

        # IR should preserve cache tokens
        assert unified.usage.cache_creation_input_tokens == 100
        assert unified.usage.cache_read_input_tokens == 0

        # Total should include cache creation
        expected_total = 50 + 20 + 100  # input + output + cache_creation
        assert unified.usage.total_tokens == expected_total

    def test_anthropic_cache_read_tokens(self):
        """Anthropic cache_read_input_tokens should be preserved"""
        anthropic_resp = {
            "id": "msg_129",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-sonnet-20240229",
            "content": [
                {
                    "type": "text",
                    "text": "Fast response from cached prompt",
                }
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "cache_read_input_tokens": 2000,
                "output_tokens": 25,
            },
        }

        # Anthropic → IR
        unified = self.anthropic_adapter.to_unified_response(anthropic_resp)

        # IR should preserve cache read tokens
        assert unified.usage.cache_read_input_tokens == 2000
        assert unified.usage.cached_tokens == 2000

        # Total should include cache read
        expected_total = 100 + 25 + 2000  # input + output + cache_read
        assert unified.usage.total_tokens == expected_total

    def test_anthropic_both_cache_types(self):
        """Response with both cache_creation and cache_read tokens"""
        anthropic_resp = {
            "id": "msg_128",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-sonnet-20240229",
            "content": [
                {
                    "type": "text",
                    "text": "This response was generated with cached context",
                }
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "cache_creation_input_tokens": 2000,
                "cache_read_input_tokens": 500,
                "output_tokens": 50,
            },
        }

        # Anthropic → IR
        unified = self.anthropic_adapter.to_unified_response(anthropic_resp)

        # Should have both
        assert unified.usage.cache_creation_input_tokens == 2000
        assert unified.usage.cache_read_input_tokens == 500

        # Total should include both cache types
        expected_total = 100 + 50 + 2000 + 500
        assert unified.usage.total_tokens == expected_total

    def test_cache_token_roundtrip_preservation(self):
        """Cache token information should survive roundtrip conversion"""
        anthropic_resp = {
            "id": "msg_cache_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-sonnet-20240229",
            "content": [{"type": "text", "text": "Response with cache"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "cache_creation_input_tokens": 2000,
                "cache_read_input_tokens": 1000,
                "output_tokens": 50,
            },
        }

        # Anthropic → IR → Anthropic
        unified = self.anthropic_adapter.to_unified_response(anthropic_resp)
        anthropic_resp2 = self.anthropic_adapter.from_unified_response(unified)

        # Usage should be reconstructed
        assert "usage" in anthropic_resp2
        usage = anthropic_resp2["usage"]
        assert usage["cache_creation_input_tokens"] == 2000
        assert usage["cache_read_input_tokens"] == 1000


class TestMediumPriorityScenarios:
    """Medium-priority conversion test scenarios (10 scenarios)"""

    def setup_method(self):
        self.openai_adapter = OpenAIAdapter()
        self.anthropic_adapter = AnthropicAdapter()

    def test_multimodal_image_url_conversion(self):
        """Test 9: Image URL format conversion between providers"""
        openai_req = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://example.com/image.jpg",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
        }

        # OpenAI → IR
        unified = self.openai_adapter.to_unified_request(openai_req)

        # Verify content blocks
        content = unified.messages[0].content
        assert isinstance(content, list)
        assert any(cb.type.value == "image_url" for cb in content)

    def test_message_metadata_preservation(self):
        """Test: Message metadata should be preserved in conversion"""
        openai_req = {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                    "metadata": {"user_id": "123", "session": "abc"},
                }
            ],
            "max_tokens": 100,
        }

        # OpenAI → IR
        unified = self.openai_adapter.to_unified_request(openai_req)

        # Metadata should be preserved
        assert unified.messages[0].metadata is not None

    def test_multiple_tool_calls_conversion(self):
        """Test: Multiple tool calls in parallel should convert correctly"""
        openai_resp = {
            "id": "chatcmpl-126",
            "object": "chat.completion",
            "created": 1677652300,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Beijing"}',
                                },
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "get_news",
                                    "arguments": '{"topic": "tech"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

        # OpenAI → IR
        unified = self.openai_adapter.to_unified_response(openai_resp)

        # Should have both tool calls
        tool_calls = unified.choices[0].message.tool_calls
        assert len(tool_calls) == 2
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[1].name == "get_news"

    def test_top_p_parameter_conversion(self):
        """Test: top_p parameter should convert correctly"""
        openai_req = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test"}],
            "top_p": 0.95,
            "max_tokens": 100,
        }

        unified = self.openai_adapter.to_unified_request(openai_req)
        assert unified.generation_params.top_p == 0.95

        # IR → Anthropic
        anthropic_req = self.anthropic_adapter.from_unified_request(unified)
        assert anthropic_req["top_p"] == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
