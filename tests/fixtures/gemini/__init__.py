"""Gemini test fixtures"""

# Simple chat request
GEMINI_CHAT_REQUEST = {
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

# Simple chat response
GEMINI_CHAT_RESPONSE = {
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

# Tool-enabled request
GEMINI_TOOL_REQUEST = {
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

# Tool call response
GEMINI_TOOL_RESPONSE = {
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

# Tool response
GEMINI_FUNCTION_RESPONSE = {
    "role": "tool",
    "parts": [
        {
            "function_response": {
                "name": "get_weather",
                "response": "The weather in Beijing is 22°C with clear skies."
            }
        }
    ]
}

# Multimodal request with image
GEMINI_MULTIMODAL_REQUEST = {
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
        "maxOutputTokens": 200,
    }
}

# Multimodal response
GEMINI_MULTIMODAL_RESPONSE = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "text": "I can see a sunset over mountains in this image."
                    }
                ],
                "role": "model"
            },
            "finishReason": "stop",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 150,
        "candidatesTokenCount": 15,
        "totalTokenCount": 165,
    }
}

# System instruction request
GEMINI_SYSTEM_REQUEST = {
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

# JSON mode request (structured output)
GEMINI_JSON_REQUEST = {
    "contents": [
        {
            "role": "user",
            "parts": [
                {
                    "text": "Extract the name and age from this text: John Doe is 30 years old."
                }
            ]
        }
    ],
    "generationConfig": {
        "responseMimeType": "application/json",
        "maxOutputTokens": 100,
    }
}

# JSON mode response
GEMINI_JSON_RESPONSE = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "text": '{"name": "John Doe", "age": 30}'
                    }
                ],
                "role": "model"
            },
            "finishReason": "stop",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 30,
        "candidatesTokenCount": 20,
        "totalTokenCount": 50,
    }
}

# Thinking blocks request (Gemini 2.x/3.x)
GEMINI_THINKING_REQUEST = {
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

# Thinking blocks response
GEMINI_THINKING_RESPONSE = {
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

# Stop sequences request
GEMINI_STOP_REQUEST = {
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

# Empty response (no content)
GEMINI_EMPTY_RESPONSE = {
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

# Streaming chunk
GEMINI_STREAM_CHUNK = {
    "chunk": {
        "content": {
            "parts": [
                {
                    "text": "Hello"
                }
            ],
            "role": "model"
        }
    },
    "createTime": 1677652288,
    "id": "stream-123",
    "model": "gemini-1.5-pro",
    "chunkIndex": 0
}

# Streaming thinking chunk
GEMINI_STREAM_THINKING_CHUNK = {
    "chunk": {
        "content": {
            "parts": [
                {
                    "thinking": "Let me think about this..."
                }
            ],
            "role": "model"
        }
    },
    "createTime": 1677652288,
    "id": "stream-124",
    "model": "gemini-2.0-flash-exp",
    "chunkIndex": 0
}

# Complex request with multiple messages
GEMINI_COMPLEX_REQUEST = {
    "contents": [
        {
            "role": "user",
            "parts": [
                {
                    "text": "Tell me about dogs"
                }
            ]
        },
        {
            "role": "model",
            "parts": [
                {
                    "text": "Dogs are loyal animals that make great pets."
                }
            ]
        },
        {
            "role": "user",
            "parts": [
                {
                    "text": "What about cats?"
                }
            ]
        }
    ]
}

# Version detection - Gemini 3.x model
GEMINI_3_MODEL = "gemini-2.0-flash-exp"

# Version detection - Gemini 1.x model
GEMINI_1_MODEL = "gemini-1.5-pro"
