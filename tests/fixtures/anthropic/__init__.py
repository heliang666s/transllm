"""Anthropic test fixtures for format conversion testing"""

# Basic chat request (Anthropic format)
ANTHROPIC_CHAT_REQUEST = {
    "model": "claude-3-sonnet-20240229",
    "messages": [
        {
            "role": "user",
            "content": "Hello, how are you?",
        }
    ],
    "max_tokens": 100,
}

# Basic chat response (Anthropic format)
ANTHROPIC_CHAT_RESPONSE = {
    "id": "msg_123",
    "type": "message",
    "role": "assistant",
    "model": "claude-3-sonnet-20240229",
    "content": [
        {
            "type": "text",
            "text": "Hello! I'm doing well, thank you. How can I help you today?",
        }
    ],
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 10,
        "output_tokens": 20,
    },
}

# Request with system message
ANTHROPIC_SYSTEM_REQUEST = {
    "model": "claude-3-sonnet-20240229",
    "system": "You are a helpful assistant.",
    "messages": [
        {
            "role": "user",
            "content": "Hello, how are you?",
        }
    ],
    "max_tokens": 100,
}

# Tool-enabled request
ANTHROPIC_TOOL_REQUEST = {
    "model": "claude-3-sonnet-20240229",
    "messages": [
        {
            "role": "user",
            "content": "What's the weather like in Beijing?",
        }
    ],
    "tools": [
        {
            "name": "get_weather",
            "description": "Get weather information",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name",
                    }
                },
                "required": ["location"],
            },
        }
    ],
    "max_tokens": 100,
}

# Tool call response
ANTHROPIC_TOOL_RESPONSE = {
    "id": "msg_124",
    "type": "message",
    "role": "assistant",
    "model": "claude-3-sonnet-20240229",
    "content": [
        {
            "type": "tool_use",
            "id": "tool_123",
            "name": "get_weather",
            "input": {"location": "Beijing"},
        }
    ],
    "stop_reason": "tool_use",
    "usage": {
        "input_tokens": 100,
        "output_tokens": 50,
    },
}

# Multimodal request (with image)
ANTHROPIC_MULTIMODAL_REQUEST = {
    "model": "claude-3-sonnet-20240229",
    "messages": [
        {
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
    ],
    "max_tokens": 100,
}

# Full-featured request with all parameters
ANTHROPIC_FULL_REQUEST = {
    "model": "claude-3-opus-20240229",
    "system": "You are a helpful assistant.",
    "messages": [
        {
            "role": "user",
            "content": "Tell me a joke",
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Why did the developer go broke?",
                }
            ],
        },
        {
            "role": "user",
            "content": "Why?",
        },
    ],
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 50,
    "max_tokens": 150,
    "stop_sequences": ["\n", "END"],
    "metadata": {
        "request_id": "req-123",
        "user_id": "user-456",
    },
}

# Streaming events (Anthropic SSE format)
ANTHROPIC_STREAM_EVENTS = [
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "text",
        },
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {
            "type": "text_delta",
            "text": "Hello",
        },
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {
            "type": "text_delta",
            "text": "!",
        },
    },
    {
        "type": "content_block_stop",
        "index": 0,
    },
    {
        "type": "message_stop",
    },
]

# Response with cached tokens
ANTHROPIC_CACHE_RESPONSE = {
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
        "cache_read_input_tokens": 1000,
        "output_tokens": 20,
    },
}

# Tool result message (in conversation)
ANTHROPIC_TOOL_RESULT_MESSAGE = {
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": "tool_123",
            "content": "Beijing weather: Sunny, 20°C",
        }
    ],
}

# Multiple tool calls in response
ANTHROPIC_MULTI_TOOL_RESPONSE = {
    "id": "msg_126",
    "type": "message",
    "role": "assistant",
    "model": "claude-3-opus-20240229",
    "content": [
        {
            "type": "tool_use",
            "id": "tool_123",
            "name": "get_weather",
            "input": {"location": "Beijing"},
        },
        {
            "type": "tool_use",
            "id": "tool_124",
            "name": "get_news",
            "input": {"location": "Beijing"},
        },
    ],
    "stop_reason": "tool_use",
    "usage": {
        "input_tokens": 150,
        "output_tokens": 100,
    },
}

# Response with thinking (Claude 3.7+ with extended thinking)
ANTHROPIC_THINKING_RESPONSE = {
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

# Response with prompt caching (cache_creation and cache_read tokens)
ANTHROPIC_CACHED_REQUEST_RESPONSE = {
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
        "output_tokens": 50,
    },
}

# Response with cache read tokens
ANTHROPIC_CACHE_READ_RESPONSE = {
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

# Response with thinking blocks (Anthropic extended thinking)
ANTHROPIC_THINKING_BLOCKS_RESPONSE = {
    "id": "msg_130",
    "type": "message",
    "role": "assistant",
    "model": "claude-3-opus-20240229",
    "content": [
        {
            "type": "thinking",
            "thinking": "Let me think about this complex problem step by step. First, I need to understand the requirements...",
        },
        {
            "type": "text",
            "text": "Based on my analysis, here's the solution...",
        },
    ],
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 150,
        "output_tokens": 300,
    },
}

# Request with thinking enabled
ANTHROPIC_THINKING_REQUEST = {
    "model": "claude-3-opus-20240229",
    "messages": [
        {
            "role": "user",
            "content": "Solve this complex problem: ...",
        }
    ],
    "max_tokens": 16000,
    "thinking": {
        "type": "enabled",
        "budget_tokens": 10000,
    },
}

# Streaming events with text and tool call deltas
ANTHROPIC_STREAMING_EVENTS = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_1234567890",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-sonnet-20240229",
            "content": [],
            "stop_reason": None,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 0,
            },
        },
    },
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "text",
        },
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {
            "type": "text_delta",
            "text": "Hello",
        },
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {
            "type": "text_delta",
            "text": " world",
        },
    },
    {
        "type": "content_block_stop",
        "index": 0,
    },
    {
        "type": "message_delta",
        "delta": {
            "stop_reason": "end_turn",
        },
        "usage": {
            "output_tokens": 11,
        },
    },
    {
        "type": "message_stop",
    },
]

# Streaming events with tool call
ANTHROPIC_STREAMING_TOOL_EVENTS = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_tool123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-sonnet-20240229",
            "content": [],
            "stop_reason": None,
            "usage": {
                "input_tokens": 20,
                "output_tokens": 0,
            },
        },
    },
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "tool_use",
            "id": "tool_use_123",
            "name": "get_weather",
        },
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {
            "type": "input_json_delta",
            "partial_json": '{"location"',
        },
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {
            "type": "input_json_delta",
            "partial_json": ': "Beijing"}',
        },
    },
    {
        "type": "content_block_stop",
        "index": 0,
    },
    {
        "type": "message_stop",
    },
]

# Streaming events with thinking blocks
ANTHROPIC_STREAMING_THINKING_EVENTS = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_thinking123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-opus-20240229",
            "content": [],
            "stop_reason": None,
            "usage": {
                "input_tokens": 30,
                "output_tokens": 0,
            },
        },
    },
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "thinking",
        },
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {
            "type": "thinking_delta",
            "thinking": "Let me analyze this",
        },
    },
    {
        "type": "content_block_stop",
        "index": 0,
    },
    {
        "type": "content_block_start",
        "index": 1,
        "content_block": {
            "type": "text",
        },
    },
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {
            "type": "text_delta",
            "text": "Based on my analysis, the answer is...",
        },
    },
    {
        "type": "content_block_stop",
        "index": 1,
    },
    {
        "type": "message_stop",
    },
]
