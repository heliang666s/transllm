"""Provider capabilities matrix for compatibility checking"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.schema import Provider


@dataclass(frozen=True)
class ProviderCapabilities:
    """Capabilities and limitations of an LLM provider"""

    provider: Provider

    # Core features
    supports_streaming: bool = True
    supports_tools: bool = True
    supports_system_message: bool = True
    supports_multimodal: bool = False
    supports_thinking_mode: bool = False
    supports_json_mode: bool = False
    supports_web_search: bool = False
    supports_reasoning: bool = False

    # Tool calling capabilities
    max_concurrent_tools: int | None = None
    supports_tool_choice: bool = True
    supports_parallel_tool_calls: bool = True

    # Generation parameters
    max_stop_sequences: int | None = None
    max_context_length: int | None = None
    supports_temperature: bool = True
    supports_top_p: bool = True
    supports_top_k: bool = False
    supports_presence_penalty: bool = False
    supports_frequency_penalty: bool = False
    supports_seed: bool = False

    # Streaming capabilities
    supports_sse: bool = True
    supports_json_lines: bool = True
    supports_websocket: bool = False

    # Message format
    supports_content_blocks: bool = True
    supports_role_enumeration: bool = True

    # Version info
    api_version: str | None = None
    model_format_version: str | None = None


class ProviderCapabilityMatrix:
    """Registry of all provider capabilities"""

    _capabilities: dict[str, ProviderCapabilities] = {}

    @classmethod
    def register(cls, capabilities: ProviderCapabilities) -> None:
        """Register capabilities for a provider"""
        # Convert enum to string for storage
        provider_key = (
            capabilities.provider.value
            if hasattr(capabilities.provider, "value")
            else str(capabilities.provider)
        )
        cls._capabilities[provider_key.lower()] = capabilities

    @classmethod
    def get_capabilities(cls, provider: Provider) -> ProviderCapabilities:
        """Get capabilities for a provider"""
        # Convert enum to string for lookup
        provider_key = provider.value.lower()

        if provider_key not in cls._capabilities:
            # Return default capabilities if not registered
            return ProviderCapabilities(provider=provider)

        return cls._capabilities[provider_key]

    @classmethod
    def is_supported(cls, provider: Provider, feature: str) -> bool:
        """Check if a provider supports a specific feature

        Args:
            provider: Provider enum
            feature: Feature name (e.g., 'streaming', 'tools', 'multimodal')

        Returns:
            True if feature is supported
        """
        capabilities = cls.get_capabilities(provider)
        return getattr(capabilities, f"supports_{feature}", False)

    @classmethod
    def check_compatibility(
        cls,
        from_provider: Provider,
        to_provider: Provider,
        request_data: dict[str, Any],
    ) -> tuple[list[str], list[str]]:
        """Check conversion compatibility between providers

        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []

        to_caps = cls.get_capabilities(to_provider)

        # Check tool support
        if request_data.get("tools") and not to_caps.supports_tools:
            errors.append(
                f"Target provider '{to_provider}' does not support tool calling"
            )

        # Check concurrent tools
        if "tools" in request_data and to_caps.max_concurrent_tools:
            tool_count = len(request_data.get("tools", []))
            if tool_count > to_caps.max_concurrent_tools:
                warnings.append(
                    f"Target provider '{to_provider}' supports maximum "
                    f"{to_caps.max_concurrent_tools} tools, but {tool_count} provided"
                )

        # Check system message
        if (
            request_data.get("system_instruction")
            and not to_caps.supports_system_message
        ):
            warnings.append(
                f"Target provider '{to_provider}' does not support "
                "system messages (will be converted to user message)"
            )

        # Check streaming
        if (
            request_data.get("generation_params", {}).get("stream")
            and not to_caps.supports_streaming
        ):
            errors.append(f"Target provider '{to_provider}' does not support streaming")

        # Check thinking mode
        if any(
            isinstance(msg.get("content"), list)
            and any(cb.get("type") == "reasoning" for cb in msg.get("content", []))
            for msg in request_data.get("messages", [])
        ):
            if not to_caps.supports_thinking_mode:
                warnings.append(
                    f"Target provider '{to_provider}' does not support "
                    "thinking mode (will be ignored)"
                )

        # Check multimodal
        has_images = any(
            isinstance(msg.get("content"), list)
            and any(cb.get("type") == "image_url" for cb in msg.get("content", []))
            for msg in request_data.get("messages", [])
        )
        if has_images and not to_caps.supports_multimodal:
            errors.append(
                f"Target provider '{to_provider}' does not support multimodal content"
            )

        return errors, warnings


# Register known provider capabilities
ProviderCapabilityMatrix.register(
    ProviderCapabilities(
        provider=Provider.openai,
        supports_streaming=True,
        supports_tools=True,
        supports_system_message=True,
        supports_multimodal=True,
        supports_json_mode=True,
        max_concurrent_tools=128,
        max_stop_sequences=4,
        max_context_length=128000,
        supports_top_k=True,
        supports_presence_penalty=True,
        supports_frequency_penalty=True,
        supports_seed=True,
        api_version="2024-02-15-preview",
    )
)

ProviderCapabilityMatrix.register(
    ProviderCapabilities(
        provider=Provider.anthropic,
        supports_streaming=True,
        supports_tools=True,
        supports_system_message=True,
        supports_multimodal=True,
        supports_thinking_mode=True,
        supports_json_mode=True,
        max_concurrent_tools=10,
        max_stop_sequences=5,
        max_context_length=200000,
        supports_top_k=True,
        supports_seed=True,
        api_version="2023-06-01",
    )
)

ProviderCapabilityMatrix.register(
    ProviderCapabilities(
        provider=Provider.gemini,
        supports_streaming=True,
        supports_tools=True,
        supports_system_message=True,
        supports_multimodal=True,
        supports_web_search=True,
        max_concurrent_tools=64,
        max_stop_sequences=5,
        max_context_length=2000000,
        supports_top_k=True,
        supports_seed=True,
    )
)

ProviderCapabilityMatrix.register(
    ProviderCapabilities(
        provider=Provider.azure_openai,
        supports_streaming=True,
        supports_tools=True,
        supports_system_message=True,
        supports_multimodal=True,
        max_concurrent_tools=128,
        max_stop_sequences=4,
        max_context_length=128000,
        supports_top_k=True,
    )
)

ProviderCapabilityMatrix.register(
    ProviderCapabilities(
        provider=Provider.aws_bedrock,
        supports_streaming=True,
        supports_tools=True,
        supports_system_message=True,
        supports_multimodal=True,
        max_concurrent_tools=50,
        max_stop_sequences=10,
        max_context_length=300000,
    )
)
