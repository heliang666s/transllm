"""Test OpenAI adapter with new features"""

import pytest
from src.transllm.adapters.openai import OpenAIAdapter
from tests.fixtures.openai import (
    OPENAI_JSON_MODE_REQUEST,
    OPENAI_LOGIT_BIAS_REQUEST,
    OPENAI_MULTI_COMPLETION_REQUEST,
    OPENAI_PARALLEL_TOOLS_REQUEST,
    OPENAI_REASONING_RESPONSE,
    OPENAI_LOGPROBS_RESPONSE,
    OPENAI_MULTI_COMPLETION_RESPONSE,
    OPENAI_MAX_COMPLETION_TOKENS_REQUEST,
)


def test_json_mode_idempotency():
    """Test JSON mode response_format parameter round-trip"""
    adapter = OpenAIAdapter()

    # Request: OpenAI -> IR -> OpenAI
    unified_request = adapter.to_unified_request(OPENAI_JSON_MODE_REQUEST)
    assert unified_request.generation_params is not None
    assert unified_request.generation_params.response_format == {"type": "json_object"}

    openai_request = adapter.from_unified_request(unified_request)
    assert openai_request["response_format"] == {"type": "json_object"}
    assert openai_request == OPENAI_JSON_MODE_REQUEST


def test_logit_bias_idempotency():
    """Test logit_bias parameter round-trip"""
    adapter = OpenAIAdapter()

    # Request: OpenAI -> IR -> OpenAI
    unified_request = adapter.to_unified_request(OPENAI_LOGIT_BIAS_REQUEST)
    assert unified_request.generation_params is not None
    assert unified_request.generation_params.logit_bias == {
        "1234": 50,
        "5678": -100,
    }

    openai_request = adapter.from_unified_request(unified_request)
    assert openai_request["logit_bias"] == {"1234": 50, "5678": -100}
    assert openai_request == OPENAI_LOGIT_BIAS_REQUEST


def test_multi_completion_idempotency():
    """Test n parameter (multiple completions) round-trip"""
    adapter = OpenAIAdapter()

    # Request: OpenAI -> IR -> OpenAI
    unified_request = adapter.to_unified_request(OPENAI_MULTI_COMPLETION_REQUEST)
    assert unified_request.generation_params is not None
    assert unified_request.generation_params.n == 3

    openai_request = adapter.from_unified_request(unified_request)
    assert openai_request["n"] == 3
    assert openai_request == OPENAI_MULTI_COMPLETION_REQUEST


def test_parallel_tool_calls_idempotency():
    """Test parallel_tool_calls parameter round-trip"""
    adapter = OpenAIAdapter()

    # Request: OpenAI -> IR -> OpenAI
    unified_request = adapter.to_unified_request(OPENAI_PARALLEL_TOOLS_REQUEST)
    assert unified_request.parallel_tool_calls is True
    assert len(unified_request.tools) == 2

    openai_request = adapter.from_unified_request(unified_request)
    assert openai_request["parallel_tool_calls"] is True
    assert openai_request == OPENAI_PARALLEL_TOOLS_REQUEST


def test_max_completion_tokens_idempotency():
    """Test max_completion_tokens parameter (alternative to max_tokens) round-trip"""
    adapter = OpenAIAdapter()

    # Request: OpenAI -> IR -> OpenAI
    unified_request = adapter.to_unified_request(OPENAI_MAX_COMPLETION_TOKENS_REQUEST)
    assert unified_request.generation_params is not None
    assert unified_request.generation_params.max_completion_tokens == 500

    openai_request = adapter.from_unified_request(unified_request)
    assert openai_request["max_completion_tokens"] == 500
    assert openai_request == OPENAI_MAX_COMPLETION_TOKENS_REQUEST


def test_reasoning_content_idempotency():
    """Test reasoning_content field in response round-trip"""
    adapter = OpenAIAdapter()

    # Response: OpenAI -> IR -> OpenAI
    unified_response = adapter.to_unified_response(OPENAI_REASONING_RESPONSE)
    assert len(unified_response.choices) == 1
    choice = unified_response.choices[0]
    assert choice.message.reasoning_content == "Let me work through this problem step by step..."

    openai_response = adapter.from_unified_response(unified_response)
    assert openai_response["choices"][0]["message"]["reasoning_content"] == \
           "Let me work through this problem step by step..."
    assert openai_response == OPENAI_REASONING_RESPONSE


def test_logprobs_idempotency():
    """Test logprobs field in response round-trip"""
    adapter = OpenAIAdapter()

    # Response: OpenAI -> IR -> OpenAI
    unified_response = adapter.to_unified_response(OPENAI_LOGPROBS_RESPONSE)
    assert len(unified_response.choices) == 1
    choice = unified_response.choices[0]
    assert choice.logprobs is not None
    assert "content" in choice.logprobs

    openai_response = adapter.from_unified_response(unified_response)
    assert openai_response == OPENAI_LOGPROBS_RESPONSE


def test_multi_completion_response_idempotency():
    """Test multiple completions in response (n > 1) round-trip"""
    adapter = OpenAIAdapter()

    # Response: OpenAI -> IR -> OpenAI
    unified_response = adapter.to_unified_response(OPENAI_MULTI_COMPLETION_RESPONSE)
    assert len(unified_response.choices) == 3
    assert unified_response.choices[0].message.content == "The weather is sunny!"
    assert unified_response.choices[1].message.content == "The weather is rainy."
    assert unified_response.choices[2].message.content == "The weather is cloudy."

    openai_response = adapter.from_unified_response(unified_response)
    assert len(openai_response["choices"]) == 3
    assert openai_response == OPENAI_MULTI_COMPLETION_RESPONSE


def test_combined_new_features():
    """Test multiple new features combined in a single request"""
    adapter = OpenAIAdapter()

    combined_request = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": "Generate JSON with multiple completions",
            }
        ],
        "response_format": {"type": "json_object"},
        "logit_bias": {"100": 10, "200": -50},
        "n": 2,
        "max_completion_tokens": 1000,
        "top_logprobs": 2,
    }

    # Request: OpenAI -> IR -> OpenAI
    unified_request = adapter.to_unified_request(combined_request)

    # Verify all features were captured
    assert unified_request.generation_params.response_format == {"type": "json_object"}
    assert unified_request.generation_params.logit_bias == {"100": 10, "200": -50}
    assert unified_request.generation_params.n == 2
    assert unified_request.generation_params.max_completion_tokens == 1000
    assert unified_request.generation_params.top_logprobs == 2

    openai_request = adapter.from_unified_request(unified_request)
    assert openai_request == combined_request


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
