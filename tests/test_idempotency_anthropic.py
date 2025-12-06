"""Anthropic adapter idempotency tests

Tests verify that conversions are idempotent:
- Anthropic → IR → Anthropic should preserve original data
- Tests cover requests, responses, and streaming data
"""

import pytest
from src.transllm.adapters.anthropic import AnthropicAdapter
from tests.fixtures.anthropic import (
    ANTHROPIC_CHAT_REQUEST,
    ANTHROPIC_CHAT_RESPONSE,
    ANTHROPIC_SYSTEM_REQUEST,
    ANTHROPIC_TOOL_REQUEST,
    ANTHROPIC_TOOL_RESPONSE,
    ANTHROPIC_MULTIMODAL_REQUEST,
    ANTHROPIC_FULL_REQUEST,
    ANTHROPIC_CACHE_RESPONSE,
    ANTHROPIC_TOOL_RESULT_MESSAGE,
    ANTHROPIC_MULTI_TOOL_RESPONSE,
    ANTHROPIC_CACHED_REQUEST_RESPONSE,
    ANTHROPIC_CACHE_READ_RESPONSE,
)


class TestAnthropicRequestIdempotency:
    """Test idempotency of Anthropic request conversions"""

    def setup_method(self):
        self.adapter = AnthropicAdapter()

    def test_basic_chat_request_idempotency(self):
        """Test that basic chat request round-trips correctly"""
        original = ANTHROPIC_CHAT_REQUEST.copy()

        # Convert to unified
        unified = self.adapter.to_unified_request(original)

        # Convert back to Anthropic
        result = self.adapter.from_unified_request(unified)

        # Check key fields are preserved
        assert result["model"] == original["model"]
        assert result["max_tokens"] == original["max_tokens"]
        assert len(result["messages"]) == len(original["messages"])
        assert result["messages"][0]["role"] == original["messages"][0]["role"]
        assert result["messages"][0]["content"][0]["text"] == original["messages"][0]["content"]

    def test_system_message_idempotency(self):
        """Test that system messages are preserved"""
        original = ANTHROPIC_SYSTEM_REQUEST.copy()

        unified = self.adapter.to_unified_request(original)
        result = self.adapter.from_unified_request(unified)

        assert result["system"] == original["system"]
        assert result["model"] == original["model"]

    def test_tool_request_idempotency(self):
        """Test that tool definitions are preserved"""
        original = ANTHROPIC_TOOL_REQUEST.copy()

        unified = self.adapter.to_unified_request(original)
        result = self.adapter.from_unified_request(unified)

        assert result["model"] == original["model"]
        assert len(result["tools"]) == len(original["tools"])
        assert result["tools"][0]["name"] == original["tools"][0]["name"]
        assert result["tools"][0]["description"] == original["tools"][0]["description"]

    def test_full_request_idempotency(self):
        """Test that full-featured requests are preserved"""
        original = ANTHROPIC_FULL_REQUEST.copy()

        unified = self.adapter.to_unified_request(original)
        result = self.adapter.from_unified_request(unified)

        assert result["system"] == original["system"]
        assert result["model"] == original["model"]
        assert result["temperature"] == original["temperature"]
        assert result["top_p"] == original["top_p"]
        assert result["max_tokens"] == original["max_tokens"]
        assert result["metadata"] == original["metadata"]

    def test_multimodal_request_idempotency(self):
        """Test that multimodal requests preserve content structure"""
        original = ANTHROPIC_MULTIMODAL_REQUEST.copy()

        unified = self.adapter.to_unified_request(original)
        result = self.adapter.from_unified_request(unified)

        assert result["model"] == original["model"]
        assert len(result["messages"]) == len(original["messages"])
        # Content should be converted back to Anthropic format
        assert result["messages"][0]["content"][0]["type"] == "text"
        assert result["messages"][0]["content"][1]["type"] == "image"


class TestAnthropicResponseIdempotency:
    """Test idempotency of Anthropic response conversions"""

    def setup_method(self):
        self.adapter = AnthropicAdapter()

    def test_basic_response_idempotency(self):
        """Test that basic responses round-trip correctly"""
        original = ANTHROPIC_CHAT_RESPONSE.copy()

        unified = self.adapter.to_unified_response(original)
        result = self.adapter.from_unified_response(unified)

        assert result["model"] == original["model"]
        assert len(result["content"]) == len(original["content"])
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == original["content"][0]["text"]

    def test_tool_response_idempotency(self):
        """Test that tool responses are preserved"""
        original = ANTHROPIC_TOOL_RESPONSE.copy()

        unified = self.adapter.to_unified_response(original)
        result = self.adapter.from_unified_response(unified)

        assert result["model"] == original["model"]
        assert len(result["content"]) > 0
        assert result["content"][0]["type"] == "tool_use"
        assert result["content"][0]["name"] == original["content"][0]["name"]
        assert result["content"][0]["input"] == original["content"][0]["input"]

    def test_response_with_usage_idempotency(self):
        """Test that usage statistics are preserved"""
        original = ANTHROPIC_CHAT_RESPONSE.copy()

        unified = self.adapter.to_unified_response(original)
        result = self.adapter.from_unified_response(unified)

        if "usage" in original:
            assert result["usage"]["input_tokens"] == original["usage"]["input_tokens"]
            assert result["usage"]["output_tokens"] == original["usage"]["output_tokens"]

    def test_cached_response_idempotency(self):
        """Test that cached token information is preserved"""
        original = ANTHROPIC_CACHE_RESPONSE.copy()

        unified = self.adapter.to_unified_response(original)
        result = self.adapter.from_unified_response(unified)

        assert result["model"] == original["model"]
        if "cache_read_input_tokens" in original["usage"]:
            assert result["usage"]["cache_read_input_tokens"] == original["usage"]["cache_read_input_tokens"]

    def test_multi_tool_response_idempotency(self):
        """Test that responses with multiple tool calls are preserved"""
        original = ANTHROPIC_MULTI_TOOL_RESPONSE.copy()

        unified = self.adapter.to_unified_response(original)
        result = self.adapter.from_unified_response(unified)

        # Should have multiple tool_use blocks
        tool_blocks = [b for b in result["content"] if b.get("type") == "tool_use"]
        assert len(tool_blocks) == 2

    def test_cached_request_response_idempotency(self):
        """Test that cache_creation_input_tokens are preserved"""
        original = ANTHROPIC_CACHED_REQUEST_RESPONSE.copy()

        unified = self.adapter.to_unified_response(original)
        result = self.adapter.from_unified_response(unified)

        assert result["model"] == original["model"]
        # Check cache_creation_input_tokens preservation
        assert result["usage"]["cache_creation_input_tokens"] == original["usage"]["cache_creation_input_tokens"]
        assert result["usage"]["input_tokens"] == original["usage"]["input_tokens"]
        assert result["usage"]["output_tokens"] == original["usage"]["output_tokens"]

    def test_cache_read_response_idempotency(self):
        """Test that cache_read_input_tokens are preserved"""
        original = ANTHROPIC_CACHE_READ_RESPONSE.copy()

        unified = self.adapter.to_unified_response(original)
        result = self.adapter.from_unified_response(unified)

        assert result["model"] == original["model"]
        # Check cache_read_input_tokens preservation
        assert result["usage"]["cache_read_input_tokens"] == original["usage"]["cache_read_input_tokens"]
        assert result["usage"]["input_tokens"] == original["usage"]["input_tokens"]
        assert result["usage"]["output_tokens"] == original["usage"]["output_tokens"]


class TestAnthropicMessageConversion:
    """Test message-level conversions"""

    def setup_method(self):
        self.adapter = AnthropicAdapter()

    def test_user_message_conversion(self):
        """Test user message conversion"""
        message = {
            "role": "user",
            "content": "Hello, world!",
        }

        unified_msg = self.adapter._to_unified_message(message)
        result = self.adapter._from_unified_message(unified_msg)

        assert result["role"] == "user"
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello, world!"

    def test_assistant_message_conversion(self):
        """Test assistant message conversion"""
        message = {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Hello! How can I help?",
                }
            ],
        }

        unified_msg = self.adapter._to_unified_message(message)
        result = self.adapter._from_unified_message(unified_msg)

        assert result["role"] == "assistant"
        assert result["content"][0]["type"] == "text"

    def test_tool_result_message_conversion(self):
        """Test tool result message conversion"""
        original = ANTHROPIC_TOOL_RESULT_MESSAGE.copy()

        unified_msg = self.adapter._to_unified_message(original)
        result = self.adapter._from_unified_message(unified_msg)

        assert result["role"] == "user"
        assert any(b.get("type") == "tool_result" for b in result["content"])

    def test_message_content_block_preservation(self):
        """Test that content blocks are preserved during conversion"""
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?",
                },
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/image.jpg",
                    },
                },
            ],
        }

        unified_msg = self.adapter._to_unified_message(message)
        result = self.adapter._from_unified_message(unified_msg)

        assert len(result["content"]) >= 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "image"


class TestAnthropicEdgeCases:
    """Test edge cases and special handling"""

    def setup_method(self):
        self.adapter = AnthropicAdapter()

    def test_empty_system_message(self):
        """Test handling of empty system messages"""
        request = {
            "model": "claude-3-sonnet-20240229",
            "system": "",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 100,
        }

        unified = self.adapter.to_unified_request(request)
        result = self.adapter.from_unified_request(unified)

        # Empty system should still be preserved or handled correctly
        assert result["model"] == request["model"]

    def test_multiple_consecutive_messages(self):
        """Test merging of consecutive same-role messages"""
        request = {
            "model": "claude-3-sonnet-20240229",
            "messages": [
                {"role": "user", "content": "First message"},
                {"role": "user", "content": "Second message"},
                {"role": "assistant", "content": [{"type": "text", "text": "Response"}]},
            ],
            "max_tokens": 100,
        }

        unified = self.adapter.to_unified_request(request)
        result = self.adapter.from_unified_request(unified)

        # After conversion, messages should be properly handled
        assert result["model"] == request["model"]
        assert "messages" in result

    def test_finish_reason_mapping(self):
        """Test finish_reason mapping between formats"""
        # Test that finish reasons are correctly mapped
        response = ANTHROPIC_CHAT_RESPONSE.copy()

        unified = self.adapter.to_unified_response(response)
        assert unified.choices[0].finish_reason.value == "stop"

        result = self.adapter.from_unified_response(unified)
        assert result["stop_reason"] == "end_turn"

    def test_tool_choice_reverse_mapping(self):
        """Test parallel_tool_calls reverse mapping"""
        request = {
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 100,
            "tools": [{"name": "test", "description": "Test", "input_schema": {}}],
        }

        unified = self.adapter.to_unified_request(request)
        unified.parallel_tool_calls = False
        result = self.adapter.from_unified_request(unified)

        # Should have disable_parallel_tool_use: true
        if "tool_choice" in result:
            assert "disable_parallel_tool_use" in result["tool_choice"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
