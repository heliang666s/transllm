"""TransLLM - Universal LLM Format Converter

A brand-neutral intermediate representation (IR) for converting between
any two LLM API formats.
"""

from .adapters.openai import OpenAIAdapter
from .adapters.anthropic import AnthropicAdapter
from .adapters.gemini import GeminiAdapter
from .utils.provider_registry import ProviderRegistry
from .core.schema import Provider

# Register built-in adapters
ProviderRegistry.register(Provider.openai, OpenAIAdapter)
ProviderRegistry.register(Provider.anthropic, AnthropicAdapter)
ProviderRegistry.register(Provider.gemini, GeminiAdapter)

__version__ = "0.1.0"
__all__ = [
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
    "ProviderRegistry",
    "Provider",
]
