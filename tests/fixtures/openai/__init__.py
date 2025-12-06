"""OpenAI test fixtures"""

# Simple chat request
OPENAI_CHAT_REQUEST = {
    "model": "gpt-4",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "Hello, how are you?",
        },
    ],
    "temperature": 0.7,
    "max_tokens": 100,
}

# Simple chat response
OPENAI_CHAT_RESPONSE = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! I'm doing well, thank you. How can I help you today?",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 12,
        "total_tokens": 32,
    },
}

# Tool-enabled request
OPENAI_TOOL_REQUEST = {
    "model": "gpt-4",
    "messages": [
        {
            "role": "user",
            "content": "What's the weather like in Beijing?",
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ],
}

# Tool call response
OPENAI_TOOL_RESPONSE = {
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
                            "arguments": '{"location": "Beijing"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
}

# Streaming events
OPENAI_STREAM_EVENTS = [
    {
        "choices": [
            {
                "delta": {
                    "content": "Hello",
                },
                "index": 0,
            }
        ],
        "sequence_id": 0,
        "timestamp": 1677652288.1,
    },
    {
        "choices": [
            {
                "delta": {
                    "content": "!",
                },
                "index": 0,
            }
        ],
        "sequence_id": 1,
        "timestamp": 1677652288.2,
    },
    {
        "choices": [
            {
                "delta": {},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "sequence_id": 2,
        "timestamp": 1677652288.3,
    },
]

# Multimodal request
OPENAI_MULTIMODAL_REQUEST = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?",
                },
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

# Full-featured request
OPENAI_FULL_REQUEST = {
    "model": "gpt-4",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
            "metadata": {"id": "sys-1"},
        },
        {
            "role": "user",
            "content": "Tell me a joke",
            "metadata": {"id": "user-1"},
        },
    ],
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 50,
    "max_tokens": 150,
    "stop": ["\n", "END"],
    "stream": False,
    "presence_penalty": 0.5,
    "frequency_penalty": 0.3,
    "metadata": {
        "request_id": "req-123",
        "user_id": "user-456",
    },
}

# JSON mode request (new feature)
OPENAI_JSON_MODE_REQUEST = {
    "model": "gpt-4-turbo",
    "messages": [
        {
            "role": "user",
            "content": "Generate a JSON object with name and age",
        }
    ],
    "response_format": {
        "type": "json_object",
    },
}

# Logit bias request (new feature)
OPENAI_LOGIT_BIAS_REQUEST = {
    "model": "gpt-4",
    "messages": [
        {
            "role": "user",
            "content": "Say 'good' instead of 'bad'",
        }
    ],
    "logit_bias": {
        "1234": 50,  # Boost probability of token 1234
        "5678": -100,  # Suppress probability of token 5678
    },
}

# Multiple completions request (new feature)
OPENAI_MULTI_COMPLETION_REQUEST = {
    "model": "gpt-4",
    "messages": [
        {
            "role": "user",
            "content": "Complete this sentence: The weather is...",
        }
    ],
    "n": 3,  # Request 3 completions
}

# Parallel tool calls request (new feature)
OPENAI_PARALLEL_TOOLS_REQUEST = {
    "model": "gpt-4-turbo",
    "messages": [
        {
            "role": "user",
            "content": "Get weather and news simultaneously",
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_news",
                "description": "Get news headlines",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"}
                    },
                    "required": ["category"],
                },
            },
        }
    ],
    "parallel_tool_calls": True,
}

# Response with reasoning content (new feature - OpenAI o1 model)
OPENAI_REASONING_RESPONSE = {
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

# Response with logprobs (new feature)
OPENAI_LOGPROBS_RESPONSE = {
    "id": "chatcmpl-126",
    "object": "chat.completion",
    "created": 1677652360,
    "model": "gpt-4",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello there!",
            },
            "finish_reason": "stop",
            "logprobs": {
                "content": [
                    {
                        "token": "Hello",
                        "logprob": -0.123,
                        "top_logprobs": [
                            {"token": "Hello", "logprob": -0.123},
                            {"token": "Hi", "logprob": -0.456},
                            {"token": "Hey", "logprob": -0.789},
                        ],
                    },
                    {
                        "token": " there",
                        "logprob": -0.234,
                        "top_logprobs": [
                            {"token": " there", "logprob": -0.234},
                            {"token": " friend", "logprob": -0.567},
                            {"token": " buddy", "logprob": -0.890},
                        ],
                    },
                ]
            },
        }
    ],
    "usage": {
        "prompt_tokens": 15,
        "completion_tokens": 8,
        "total_tokens": 23,
    },
}

# Response with multiple completions (new feature)
OPENAI_MULTI_COMPLETION_RESPONSE = {
    "id": "chatcmpl-127",
    "object": "chat.completion",
    "created": 1677652370,
    "model": "gpt-4",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The weather is sunny!",
            },
            "finish_reason": "stop",
        },
        {
            "index": 1,
            "message": {
                "role": "assistant",
                "content": "The weather is rainy.",
            },
            "finish_reason": "stop",
        },
        {
            "index": 2,
            "message": {
                "role": "assistant",
                "content": "The weather is cloudy.",
            },
            "finish_reason": "stop",
        },
    ],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 30,
        "total_tokens": 50,
    },
}

# Max completion tokens request (new feature - alternative to max_tokens)
OPENAI_MAX_COMPLETION_TOKENS_REQUEST = {
    "model": "gpt-4",
    "messages": [
        {
            "role": "user",
            "content": "Write a short story",
        }
    ],
    "max_completion_tokens": 500,
}

