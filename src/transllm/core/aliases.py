# Provider-specific field alias mappings for TransLLM
# Each provider maps their native field names to the brand-neutral IR fields
# This enables bidirectional conversion between any two providers

from typing import Dict, Any


class ProviderAliases:
    """Field alias mappings for each LLM provider"""

    # OpenAI aliases - uses OpenAI-style field names as baseline
    OPENAI = {
        "model": "model",
        "messages": "messages",
        "temperature": "generation_params.temperature",
        "max_tokens": "generation_params.max_tokens",
        "top_p": "generation_params.top_p",
        "top_k": "generation_params.top_k",
        "stream": "generation_params.stream",
        "stop": "generation_params.stop_sequences",
        "tools": "tools",
        "tool_choice": "tool_choice",
        "system": "system_instruction",
        "user": "user",
        "assistant": "assistant",
        "content": "content",
        "role": "role",
        "tool": "tool",
        "id": "id",
        "object": "object",
        "created": "created",
        "choices": "choices",
        "usage": "usage",
        "prompt_tokens": "usage.prompt_tokens",
        "completion_tokens": "usage.completion_tokens",
        "total_tokens": "usage.total_tokens",
        "finish_reason": "finish_reason",
        "logprobs": "logprobs",
    }

    # Anthropic aliases - Claude-style field names
    ANTHROPIC = {
        # Request fields
        "model": "model",
        "messages": "messages",
        "max_tokens": "generation_params.max_tokens",
        "temperature": "generation_params.temperature",
        "top_p": "generation_params.top_p",
        "top_k": "generation_params.top_k",
        "stop_sequences": "generation_params.stop_sequences",
        "stream": "generation_params.stream",
        "tools": "tools",
        "tool_choice": "tool_choice",
        "system": "system_instruction",
        "metadata": "metadata",

        # Message fields
        "role": "role",
        "content": "content",

        # Response fields
        "id": "id",
        "type": "object",
        "role": "role",
        "content": "content",
        "usage": "usage",
        "input_tokens": "usage.input_tokens",
        "output_tokens": "usage.output_tokens",
        "stop_reason": "finish_reason",

        # Tool fields
        "name": "name",
        "description": "description",
        "input_schema": "parameters",
    }

    # Google Gemini aliases - Gemini-style field names
    GEMINI = {
        # Request fields
        "contents": "messages",
        "generationConfig": "generation_params",
        "temperature": "generation_params.temperature",
        "maxOutputTokens": "generation_params.max_tokens",
        "topP": "generation_params.top_p",
        "topK": "generation_params.top_k",
        "stopSequences": "generation_params.stop_sequences",
        "stream": "generation_params.stream",
        "tools": "tools",
        "toolConfig": "tool_choice",
        "systemInstruction": "system_instruction",
        "safetySettings": "metadata",

        # Content fields
        "parts": "content",
        "role": "role",
        "text": "text",

        # Response fields
        "candidates": "choices",
        "promptFeedback": "metadata",
        "usageMetadata": "usage",
        "promptTokenCount": "usage.prompt_tokens",
        "candidatesTokenCount": "usage.completion_tokens",
        "totalTokenCount": "usage.total_tokens",

        # Message roles (different enum values)
        "user": "user",
        "model": "assistant",
    }

    # Azure OpenAI aliases - similar to OpenAI with Azure-specific fields
    AZURE_OPENAI = {
        "model": "model",
        "messages": "messages",
        "temperature": "generation_params.temperature",
        "max_tokens": "generation_params.max_tokens",
        "top_p": "generation_params.top_p",
        "top_k": "generation_params.top_k",
        "stream": "generation_params.stream",
        "stop": "generation_params.stop_sequences",
        "tools": "tools",
        "tool_choice": "tool_choice",
        "user": "user",
        "assistant": "assistant",
        "content": "content",
        "role": "role",
        "id": "id",
        "object": "object",
        "created": "created",
        "choices": "choices",
        "usage": "usage",
        "prompt_tokens": "usage.prompt_tokens",
        "completion_tokens": "usage.completion_tokens",
        "total_tokens": "usage.total_tokens",
        "finish_reason": "finish_reason",
        "apim-Version": "metadata.azure_version",
        "enterprise-id": "metadata.enterprise_id",
    }

    # AWS Bedrock aliases - Bedrock-style field names
    AWS_BEDROCK = {
        # Request fields
        "modelId": "model",
        "messages": "messages",
        "inferenceConfig": "generation_params",
        "temperature": "generation_params.temperature",
        "maxTokens": "generation_params.max_tokens",
        "topP": "generation_params.top_p",
        "topK": "generation_params.top_k",
        "stopSequences": "generation_params.stop_sequences",
        "tools": "tools",
        "toolConfiguration": "tool_choice",
        "system": "system_instruction",

        # Content fields
        "role": "role",
        "content": "content",

        # Response fields
        "response": "choices",
        "usage": "usage",
        "inputTokenCount": "usage.prompt_tokens",
        "outputTokenCount": "usage.completion_tokens",
        "totalTokenCount": "usage.total_tokens",
        "stopReason": "finish_reason",
    }

    # Google Vertex AI aliases - Vertex AI specific
    GOOGLE_VERTEX_AI = {
        # Request fields
        "model": "model",
        "contents": "messages",
        "generationConfig": "generation_params",
        "temperature": "generation_params.temperature",
        "maxOutputTokens": "generation_params.max_tokens",
        "topP": "generation_params.top_p",
        "topK": "generation_params.top_k",
        "stopSequences": "generation_params.stop_sequences",
        "tools": "tools",
        "toolConfig": "tool_choice",
        "systemInstruction": "system_instruction",

        # Content fields
        "parts": "content",
        "role": "role",

        # Response fields
        "candidates": "choices",
        "usageMetadata": "usage",
        "promptTokenCount": "usage.prompt_tokens",
        "candidatesTokenCount": "usage.completion_tokens",
        "totalTokenCount": "usage.total_tokens",
        "finishReason": "finish_reason",
    }

    # Cohere aliases - Cohere-style field names
    COHERE = {
        # Request fields
        "message": "messages",
        "model": "model",
        "temperature": "generation_params.temperature",
        "max_tokens": "generation_params.max_tokens",
        "top_p": "generation_params.top_p",
        "stream": "generation_params.stream",
        "stop_sequences": "generation_params.stop_sequences",
        "tools": "tools",
        "tool_choice": "tool_choice",

        # Response fields
        "text": "content",
        "generation_id": "id",
        "token_count": "usage.total_tokens",
        "finish_reason": "finish_reason",
    }

    # HuggingFace aliases - HF Inference API style
    HUGGINGFACE = {
        # Request fields
        "inputs": "messages",
        "parameters": "generation_params",
        "temperature": "generation_params.temperature",
        "max_new_tokens": "generation_params.max_tokens",
        "top_p": "generation_params.top_p",
        "stream": "generation_params.stream",
        "stop": "generation_params.stop_sequences",
        "model": "model",
        "options": "metadata",
        "wait_for_model": "metadata.wait_for_model",
    }

    # VLLM aliases - vLLM server style
    VLLM = {
        # Request fields
        "model": "model",
        "messages": "messages",
        "temperature": "generation_params.temperature",
        "max_tokens": "generation_params.max_tokens",
        "top_p": "generation_params.top_p",
        "top_k": "generation_params.top_k",
        "stream": "generation_params.stream",
        "stop": "generation_params.stop_sequences",
        "tools": "tools",
        "tool_choice": "tool_choice",

        # Response fields
        "id": "id",
        "object": "object",
        "created": "created",
        "choices": "choices",
        "usage": "usage",
    }

    # NVIDIA NIM aliases - NIM style
    NVIDIA_NIM = {
        # Request fields
        "model": "model",
        "messages": "messages",
        "temperature": "generation_params.temperature",
        "max_tokens": "generation_params.max_tokens",
        "top_p": "generation_params.top_p",
        "stream": "generation_params.stream",
        "stop": "generation_params.stop_sequences",
        "tools": "tools",
        "tool_choice": "tool_choice",

        # Response fields
        "id": "id",
        "object": "object",
        "created": "created",
        "choices": "choices",
        "usage": "usage",
    }

    # Together AI aliases - Together style
    TOGETHER_AI = {
        # Request fields
        "model": "model",
        "messages": "messages",
        "temperature": "generation_params.temperature",
        "max_tokens": "generation_params.max_tokens",
        "top_p": "generation_params.top_p",
        "stream": "generation_params.stream",
        "stop": "generation_params.stop_sequences",
        "tools": "tools",
        "tool_choice": "tool_choice",

        # Response fields
        "id": "id",
        "object": "object",
        "created": "created",
        "choices": "choices",
        "usage": "usage",
    }

    # Fireworks AI aliases - Fireworks style
    FIREWORKS_AI = {
        # Request fields
        "model": "model",
        "messages": "messages",
        "temperature": "generation_params.temperature",
        "max_tokens": "generation_params.max_tokens",
        "top_p": "generation_params.top_p",
        "stream": "generation_params.stream",
        "stop": "generation_params.stop_sequences",
        "tools": "tools",
        "tool_choice": "tool_choice",

        # Response fields
        "id": "id",
        "object": "object",
        "created": "created",
        "choices": "choices",
        "usage": "usage",
    }

    # Mistral aliases - Mistral style
    MISTRAL = {
        # Request fields
        "model": "model",
        "messages": "messages",
        "temperature": "generation_params.temperature",
        "max_tokens": "generation_params.max_tokens",
        "top_p": "generation_params.top_p",
        "stream": "generation_params.stream",
        "stop": "generation_params.stop_sequences",
        "tools": "tools",
        "tool_choice": "tool_choice",

        # Response fields
        "id": "id",
        "object": "object",
        "created": "created",
        "choices": "choices",
        "usage": "usage",
    }

    # Groq aliases - Groq style
    GROQ = {
        # Request fields
        "model": "model",
        "messages": "messages",
        "temperature": "generation_params.temperature",
        "max_tokens": "generation_params.max_tokens",
        "top_p": "generation_params.top_p",
        "stream": "generation_params.stream",
        "stop": "generation_params.stop_sequences",
        "tools": "tools",
        "tool_choice": "tool_choice",

        # Response fields
        "id": "id",
        "object": "object",
        "created": "created",
        "choices": "choices",
        "usage": "usage",
    }

    @classmethod
    def get_provider_aliases(cls, provider: str) -> Dict[str, str]:
        """Get the alias mapping for a specific provider"""
        provider_key = provider.upper().replace("-", "_")
        aliases = getattr(cls, provider_key, None)

        if aliases is None:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: {cls.list_supported_providers()}")

        return aliases

    @classmethod
    def list_supported_providers(cls) -> list:
        """List all supported provider names"""
        providers = []
        for attr_name in dir(cls):
            if not attr_name.startswith("_") and attr_name.isupper():
                providers.append(attr_name.lower().replace("_", "-"))
        return providers

    @classmethod
    def get_reverse_mapping(cls, provider: str) -> Dict[str, str]:
        """Get reverse mapping (IR -> provider) for a specific provider"""
        aliases = cls.get_provider_aliases(provider)
        return {v: k for k, v in aliases.items()}


# Convenience function to get aliases
def get_provider_aliases(provider: str) -> Dict[str, str]:
    """Get field aliases for a provider"""
    return ProviderAliases.get_provider_aliases(provider)


def list_supported_providers() -> list:
    """List all supported providers"""
    return ProviderAliases.list_supported_providers()


def get_reverse_aliases(provider: str) -> Dict[str, str]:
    """Get reverse aliases (IR -> provider) for a provider"""
    return ProviderAliases.get_reverse_mapping(provider)
