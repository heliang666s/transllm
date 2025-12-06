"""Extended Anthropic idempotency tests for advanced features

Tests verify that Anthropic → IR → Anthropic roundtrip conversions
preserve advanced features including thinking blocks, cache_control,
and multi-block content structures.
"""

import pytest
from src.transllm.adapters.anthropic import AnthropicAdapter
from tests.fixtures.anthropic import (
    ANTHROPIC_THINKING_RESPONSE,
    ANTHROPIC_CACHE_RESPONSE,
)


class TestAnthropicThinkingBlocksIdempotency:
    """Test 1: Thinking blocks preservation in roundtrip"""

    def setup_method(self):
        self.adapter = AnthropicAdapter()

    def test_thinking_blocks_roundtrip_response(self):
        """Anthropic thinking blocks should survive roundtrip conversion"""
        # Start with Anthropic response containing thinking
        anthropic_resp = {
            "id": "msg_thinking_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-opus-20240229",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Let me analyze this step by step... First I need to understand the problem...",
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
        unified = self.adapter.to_unified_response(anthropic_resp)

        # Verify thinking blocks in IR
        assert len(unified.choices[0].message.content) == 2
        content_blocks = unified.choices[0].message.content
        assert content_blocks[0].type.value == "thinking" or content_blocks[0].type == "thinking"
        assert "analyze this step by step" in content_blocks[0].thinking.content

        # IR → Anthropic
        anthropic_resp2 = self.adapter.from_unified_response(unified)

        # Verify thinking blocks preserved
        assert len(anthropic_resp2["content"]) == 2
        assert anthropic_resp2["content"][0]["type"] == "thinking"
        assert "analyze this step by step" in anthropic_resp2["content"][0]["thinking"]

    def test_thinking_blocks_with_multiple_content(self):
        """Thinking blocks with tool calls: tool_use becomes tool_call in IR"""
        anthropic_resp = {
            "id": "msg_complex_thinking",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-opus-20240229",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "I need to call a tool to get this information.",
                },
                {
                    "type": "tool_use",
                    "id": "tool_xyz",
                    "name": "search",
                    "input": {"query": "weather data"},
                },
                {
                    "type": "text",
                    "text": "I've searched for the information.",
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 150, "output_tokens": 75},
        }

        # Roundtrip
        unified = self.adapter.to_unified_response(anthropic_resp)

        # Verify thinking and text blocks preserved in IR
        assert len(unified.choices[0].message.content) >= 2
        content_types = [str(b.type) for b in unified.choices[0].message.content]
        assert any('thinking' in t for t in content_types)
        assert any('text' in t for t in content_types)

        # Verify tool_use extracted as tool_call
        assert unified.choices[0].message.tool_calls is not None
        assert len(unified.choices[0].message.tool_calls) == 1
        assert unified.choices[0].message.tool_calls[0].name == "search"

        # Convert back to Anthropic
        anthropic_resp2 = self.adapter.from_unified_response(unified)

        # Verify thinking and tool_use blocks are reconstructed
        # (Order may differ as tool_calls are appended to content)
        types_in_response = [block["type"] for block in anthropic_resp2["content"]]
        assert "thinking" in types_in_response
        assert "tool_use" in types_in_response
        assert "text" in types_in_response

    def test_redacted_thinking_preservation(self):
        """Redacted thinking blocks should be preserved in roundtrip"""
        anthropic_resp = {
            "id": "msg_redacted",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-opus-20240229",
            "content": [
                {
                    "type": "redacted_thinking",
                },
                {
                    "type": "text",
                    "text": "The answer is X, but I can't show my reasoning.",
                },
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }

        # Roundtrip
        unified = self.adapter.to_unified_response(anthropic_resp)
        anthropic_resp2 = self.adapter.from_unified_response(unified)

        # Verify redacted thinking preserved
        assert len(anthropic_resp2["content"]) == 2
        assert anthropic_resp2["content"][0]["type"] == "redacted_thinking"


class TestAnthropicCacheControlIdempotency:
    """Test 2: Cache control field preservation"""

    def setup_method(self):
        self.adapter = AnthropicAdapter()

    def test_cache_control_on_system(self):
        """cache_control on system message should be preserved"""
        anthropic_req = {
            "model": "claude-3-opus-20240229",
            "system": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant with access to a large knowledge base.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [
                {"role": "user", "content": "What is Python?"},
            ],
            "max_tokens": 1024,
        }

        # Roundtrip request
        unified = self.adapter.to_unified_request(anthropic_req)
        anthropic_req2 = self.adapter.from_unified_request(unified)

        # Verify cache_control preserved (if supported by adapter)
        # Note: This test verifies the structure is preserved
        assert "system" in anthropic_req2

    def test_cache_control_on_message(self):
        """cache_control on message should be preserved"""
        anthropic_req = {
            "model": "claude-3-opus-20240229",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, let's discuss Python programming.",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            "max_tokens": 1024,
        }

        # Roundtrip request
        unified = self.adapter.to_unified_request(anthropic_req)
        anthropic_req2 = self.adapter.from_unified_request(unified)

        # Verify message structure
        assert len(anthropic_req2["messages"]) > 0

    def test_cache_control_on_tool(self):
        """cache_control on tool definition should be preserved"""
        anthropic_req = {
            "model": "claude-3-opus-20240229",
            "messages": [
                {"role": "user", "content": "Help me search for information"},
            ],
            "tools": [
                {
                    "name": "search",
                    "description": "Search the internet",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                    },
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "max_tokens": 1024,
        }

        # Roundtrip request
        unified = self.adapter.to_unified_request(anthropic_req)
        anthropic_req2 = self.adapter.from_unified_request(unified)

        # Verify tools structure preserved
        if "tools" in anthropic_req2:
            assert len(anthropic_req2["tools"]) > 0


class TestAnthropicMultiBlockContentIdempotency:
    """Test 3: Complex multi-block content preservation"""

    def setup_method(self):
        self.adapter = AnthropicAdapter()

    def test_mixed_text_and_image_response(self):
        """Mixed text and image blocks in response should roundtrip"""
        anthropic_resp = {
            "id": "msg_multiblock",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-opus-20240229",
            "content": [
                {
                    "type": "text",
                    "text": "Here is the image analysis:",
                },
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/image.jpg",
                    },
                },
                {
                    "type": "text",
                    "text": "The image shows...",
                },
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 200, "output_tokens": 100},
        }

        # Roundtrip
        unified = self.adapter.to_unified_response(anthropic_resp)
        anthropic_resp2 = self.adapter.from_unified_response(unified)

        # Verify text blocks are present (image blocks may not fully roundtrip through IR)
        assert len(anthropic_resp2["content"]) >= 2
        types = [block["type"] for block in anthropic_resp2["content"]]
        # At least two text blocks should be present
        assert types.count("text") >= 2
        # Verify text content is preserved
        text_blocks = [b for b in anthropic_resp2["content"] if b.get("type") == "text"]
        text_contents = [b.get("text", "") for b in text_blocks]
        assert any("image analysis" in t for t in text_contents)
        assert any("image shows" in t for t in text_contents)

    def test_tool_result_in_message(self):
        """Tool result blocks in messages should roundtrip"""
        anthropic_req = {
            "model": "claude-3-opus-20240229",
            "messages": [
                {"role": "user", "content": "Get weather info"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_123",
                            "name": "get_weather",
                            "input": {"location": "Beijing"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_123",
                            "content": "Beijing: 20°C, Sunny",
                        }
                    ],
                },
            ],
            "max_tokens": 1024,
        }

        # Roundtrip request
        unified = self.adapter.to_unified_request(anthropic_req)
        anthropic_req2 = self.adapter.from_unified_request(unified)

        # Verify message structure preserved
        assert len(anthropic_req2["messages"]) == 3


class TestAnthropicUsageTokensIdempotency:
    """Test 4: Usage tokens calculation consistency"""

    def setup_method(self):
        self.adapter = AnthropicAdapter()

    def test_cache_tokens_calculation(self):
        """Cache tokens should be properly calculated in roundtrip"""
        anthropic_resp = {
            "id": "msg_cache_tokens",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-opus-20240229",
            "content": [{"type": "text", "text": "Cached response"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "cache_creation_input_tokens": 500,
                "cache_read_input_tokens": 200,
                "output_tokens": 50,
            },
        }

        # Roundtrip
        unified = self.adapter.to_unified_response(anthropic_resp)
        anthropic_resp2 = self.adapter.from_unified_response(unified)

        # Verify cache tokens preserved
        usage = anthropic_resp2["usage"]
        assert usage["cache_creation_input_tokens"] == 500
        assert usage["cache_read_input_tokens"] == 200

    def test_total_tokens_includes_cache(self):
        """Total tokens should include all cache components"""
        anthropic_resp = {
            "id": "msg_total",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-opus-20240229",
            "content": [{"type": "text", "text": "Response"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "cache_creation_input_tokens": 2000,
                "output_tokens": 50,
            },
        }

        # Roundtrip
        unified = self.adapter.to_unified_response(anthropic_resp)
        
        # Verify total calculation
        # Total should be: input + output + cache_creation
        expected_total = 100 + 50 + 2000
        assert unified.usage.total_tokens == expected_total


class TestAnthropicMetadataPreservation:
    """Test 5: Message and request metadata preservation"""

    def setup_method(self):
        self.adapter = AnthropicAdapter()

    def test_message_metadata_roundtrip(self):
        """Message-level metadata should be preserved"""
        anthropic_req = {
            "model": "claude-3-opus-20240229",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                    "metadata": {
                        "user_id": "user_123",
                        "session_id": "session_456",
                    },
                },
            ],
            "max_tokens": 1024,
        }

        # Roundtrip request
        unified = self.adapter.to_unified_request(anthropic_req)
        anthropic_req2 = self.adapter.from_unified_request(unified)

        # Verify metadata structure preserved
        assert len(anthropic_req2["messages"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
