"""Extended OpenAI idempotency tests for advanced features

Tests verify that OpenAI → IR → OpenAI roundtrip conversions
preserve advanced features including reasoning content, JSON mode,
logprobs, penalties, and multiple completions.
"""

import pytest
from src.transllm.adapters.openai import OpenAIAdapter


class TestOpenAIReasoningContentIdempotency:
    """Test 1: Reasoning content preservation in roundtrip"""

    def setup_method(self):
        self.adapter = OpenAIAdapter()

    def test_reasoning_content_roundtrip_response(self):
        """OpenAI reasoning_content should survive roundtrip conversion"""
        # Start with OpenAI response containing reasoning_content
        openai_resp = {
            "id": "chatcmpl-reasoning-test",
            "object": "chat.completion",
            "created": 1677652288,
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
        unified = self.adapter.to_unified_response(openai_resp)

        # Verify reasoning_content in IR
        assert (
            unified.choices[0].message.reasoning_content
            == "Let me work through this problem step by step..."
        )

        # IR → OpenAI
        openai_resp2 = self.adapter.from_unified_response(unified)

        # Verify reasoning_content preserved
        assert "reasoning_content" in openai_resp2["choices"][0]["message"]
        assert (
            openai_resp2["choices"][0]["message"]["reasoning_content"]
            == "Let me work through this problem step by step..."
        )


class TestOpenAIJSONModeIdempotency:
    """Test 2: JSON mode request/response preservation"""

    def setup_method(self):
        self.adapter = OpenAIAdapter()

    def test_json_mode_request_roundtrip(self):
        """OpenAI JSON mode request format should survive roundtrip"""
        openai_req = {
            "model": "gpt-4-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": "Generate a JSON object with user data",
                }
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 1024,
        }

        # OpenAI → IR
        unified = self.adapter.to_unified_request(openai_req)

        # Verify response_format preserved
        assert unified.generation_params is not None
        assert unified.generation_params.response_format is not None
        assert unified.generation_params.response_format["type"] == "json_object"

        # IR → OpenAI
        openai_req2 = self.adapter.from_unified_request(unified)

        # Verify response_format preserved
        assert "response_format" in openai_req2
        assert openai_req2["response_format"]["type"] == "json_object"

    def test_json_mode_response_roundtrip(self):
        """OpenAI JSON mode response should survive roundtrip"""
        openai_resp = {
            "id": "chatcmpl-json-test",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4-turbo",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": '{"name": "John", "age": 30}',
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }

        # OpenAI → IR
        unified = self.adapter.to_unified_response(openai_resp)

        # Verify content is preserved
        assert '{"name": "John", "age": 30}' in unified.choices[0].message.content

        # IR → OpenAI
        openai_resp2 = self.adapter.from_unified_response(unified)

        # Verify JSON content preserved
        assert (
            openai_resp2["choices"][0]["message"]["content"]
            == '{"name": "John", "age": 30}'
        )


class TestOpenAILogprobsIdempotency:
    """Test 3: Logprobs data preservation"""

    def setup_method(self):
        self.adapter = OpenAIAdapter()

    def test_logprobs_request_roundtrip(self):
        """OpenAI logprobs parameter should survive roundtrip"""
        openai_req = {
            "model": "gpt-4-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "logprobs": True,
            "top_logprobs": 2,
            "max_tokens": 100,
        }

        # OpenAI → IR
        unified = self.adapter.to_unified_request(openai_req)

        # Verify logprobs preserved
        assert unified.generation_params is not None
        assert unified.generation_params.logprobs is True
        assert unified.generation_params.top_logprobs == 2

        # IR → OpenAI
        openai_req2 = self.adapter.from_unified_request(unified)

        # Verify logprobs preserved
        assert openai_req2["logprobs"] is True
        assert openai_req2["top_logprobs"] == 2

    def test_logprobs_response_roundtrip(self):
        """OpenAI logprobs response data should survive roundtrip"""
        openai_resp = {
            "id": "chatcmpl-logprobs-test",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4-turbo",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello!",
                    },
                    "finish_reason": "stop",
                    "logprobs": {
                        "content": [
                            {
                                "token": "Hello",
                                "logprob": -0.5,
                                "top_logprobs": [
                                    {"token": "Hi", "logprob": -1.0},
                                    {"token": "Hey", "logprob": -1.5},
                                ],
                            }
                        ]
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        # OpenAI → IR
        unified = self.adapter.to_unified_response(openai_resp)

        # Verify logprobs preserved
        assert unified.choices[0].logprobs is not None
        assert "content" in unified.choices[0].logprobs
        assert len(unified.choices[0].logprobs["content"]) > 0

        # IR → OpenAI
        openai_resp2 = self.adapter.from_unified_response(unified)

        # Verify logprobs data preserved
        assert "logprobs" in openai_resp2["choices"][0]
        assert openai_resp2["choices"][0]["logprobs"] is not None


class TestOpenAIPenaltyParametersIdempotency:
    """Test 4-6: Penalty parameter preservation"""

    def setup_method(self):
        self.adapter = OpenAIAdapter()

    def test_presence_penalty_roundtrip(self):
        """OpenAI presence_penalty parameter should survive roundtrip"""
        openai_req = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "presence_penalty": 0.5,
            "max_tokens": 100,
        }

        # OpenAI → IR
        unified = self.adapter.to_unified_request(openai_req)

        # Verify presence_penalty preserved
        assert unified.generation_params is not None
        assert unified.generation_params.presence_penalty == 0.5

        # IR → OpenAI
        openai_req2 = self.adapter.from_unified_request(unified)

        # Verify presence_penalty preserved
        assert openai_req2["presence_penalty"] == 0.5

    def test_frequency_penalty_roundtrip(self):
        """OpenAI frequency_penalty parameter should survive roundtrip"""
        openai_req = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "frequency_penalty": 0.75,
            "max_tokens": 100,
        }

        # OpenAI → IR
        unified = self.adapter.to_unified_request(openai_req)

        # Verify frequency_penalty preserved
        assert unified.generation_params is not None
        assert unified.generation_params.frequency_penalty == 0.75

        # IR → OpenAI
        openai_req2 = self.adapter.from_unified_request(unified)

        # Verify frequency_penalty preserved
        assert openai_req2["frequency_penalty"] == 0.75

    def test_logit_bias_roundtrip(self):
        """OpenAI logit_bias parameter should survive roundtrip"""
        logit_bias = {"50256": -100, "50257": 50}

        openai_req = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "logit_bias": logit_bias,
            "max_tokens": 100,
        }

        # OpenAI → IR
        unified = self.adapter.to_unified_request(openai_req)

        # Verify logit_bias preserved
        assert unified.generation_params is not None
        assert unified.generation_params.logit_bias == logit_bias

        # IR → OpenAI
        openai_req2 = self.adapter.from_unified_request(unified)

        # Verify logit_bias preserved
        assert openai_req2["logit_bias"] == logit_bias


class TestOpenAIMultipleCompletionsIdempotency:
    """Test 7: Multiple completions (n parameter) preservation"""

    def setup_method(self):
        self.adapter = OpenAIAdapter()

    def test_multiple_completions_request_roundtrip(self):
        """OpenAI n parameter for multiple completions should survive roundtrip"""
        openai_req = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Generate ideas"}],
            "n": 3,
            "max_tokens": 100,
        }

        # OpenAI → IR
        unified = self.adapter.to_unified_request(openai_req)

        # Verify n parameter preserved
        assert unified.generation_params is not None
        assert unified.generation_params.n == 3

        # IR → OpenAI
        openai_req2 = self.adapter.from_unified_request(unified)

        # Verify n parameter preserved
        assert openai_req2["n"] == 3

    def test_multiple_completions_response_roundtrip(self):
        """OpenAI response with multiple completions should survive roundtrip"""
        openai_resp = {
            "id": "chatcmpl-multi-test",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Idea 1: Use machine learning...",
                    },
                    "finish_reason": "stop",
                },
                {
                    "index": 1,
                    "message": {
                        "role": "assistant",
                        "content": "Idea 2: Implement automation...",
                    },
                    "finish_reason": "stop",
                },
                {
                    "index": 2,
                    "message": {
                        "role": "assistant",
                        "content": "Idea 3: Create microservices...",
                    },
                    "finish_reason": "stop",
                },
            ],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 150,
                "total_tokens": 200,
            },
        }

        # OpenAI → IR
        unified = self.adapter.to_unified_response(openai_resp)

        # Verify all 3 choices preserved
        assert len(unified.choices) == 3
        contents = [choice.message.content for choice in unified.choices]
        assert "machine learning" in contents[0]
        assert "automation" in contents[1]
        assert "microservices" in contents[2]

        # IR → OpenAI
        openai_resp2 = self.adapter.from_unified_response(unified)

        # Verify all 3 choices preserved
        assert len(openai_resp2["choices"]) == 3
        assert (
            openai_resp2["choices"][0]["message"]["content"]
            == "Idea 1: Use machine learning..."
        )
        assert (
            openai_resp2["choices"][1]["message"]["content"]
            == "Idea 2: Implement automation..."
        )
        assert (
            openai_resp2["choices"][2]["message"]["content"]
            == "Idea 3: Create microservices..."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
