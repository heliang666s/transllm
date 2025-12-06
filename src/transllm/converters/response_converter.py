"""Response format converter"""

from __future__ import annotations

from typing import Any, Dict

from src.transllm.core.schema import Provider
from ..utils.provider_registry import ProviderRegistry
from ..core.exceptions import ConversionError, UnsupportedProviderError


class ResponseConverter:
    """Converts responses between different provider formats"""

    @staticmethod
    def convert(
        data: Dict[str, Any],
        from_provider: Provider,
        to_provider: Provider,
    ) -> Dict[str, Any]:
        """Convert response from one provider format to another

        Args:
            data: Response data in source provider format
            from_provider: Source provider (Provider enum)
            to_provider: Target provider (Provider enum)

        Returns:
            Response data in target provider format

        Examples:
            >>> # Using Provider enum (IDE autocomplete supported)
            >>> ResponseConverter.convert(data, Provider.openai, Provider.anthropic)

        Raises:
            UnsupportedProviderError: If provider is not supported
            ConversionError: If conversion fails
        """
        # Get adapters
        try:
            from_adapter = ProviderRegistry.get_adapter(from_provider)
        except UnsupportedProviderError:
            raise UnsupportedProviderError(
                from_provider,
                ProviderRegistry.list_supported_providers(),
            )

        try:
            to_adapter = ProviderRegistry.get_adapter(to_provider)
        except UnsupportedProviderError:
            raise UnsupportedProviderError(
                to_provider,
                ProviderRegistry.list_supported_providers(),
            )

        try:
            # Convert to unified IR
            unified_response = from_adapter.to_unified_response(data)

            # Convert to target format
            converted_response = to_adapter.from_unified_response(unified_response)

            return converted_response

        except Exception as e:
            if isinstance(e, ConversionError):
                raise
            raise ConversionError(
                f"Failed to convert response from {from_provider.value} to {to_provider.value}",
                from_provider,
                to_provider,
                {"original_error": str(e)},
            ) from e

    @staticmethod
    def check_idempotency(
        data: Dict[str, Any],
        provider: Provider,
    ) -> bool:
        """Check if conversion is idempotent (A -> IR -> A)

        Args:
            data: Response data
            provider: Provider name

        Returns:
            True if idempotent, False otherwise
        """
        try:
            converter = ResponseConverter()
            # Convert to IR and back
            intermediate = converter.convert(data, provider, provider)

            # Compare (simple deep comparison)
            return converter._deep_compare(data, intermediate)

        except Exception:
            return False

    @staticmethod
    def _deep_compare(obj1: Any, obj2: Any, path: str = "") -> bool:
        """Deep comparison of two objects with enum handling"""
        # Handle enum comparison
        if hasattr(obj1, 'value') and hasattr(obj2, 'value'):
            return obj1.value == obj2.value
        if hasattr(obj1, 'value') or hasattr(obj2, 'value'):
            # One is enum, one is not - compare enum value with the other
            val1 = obj1.value if hasattr(obj1, 'value') else obj1
            val2 = obj2.value if hasattr(obj2, 'value') else obj2
            return val1 == val2

        if type(obj1) != type(obj2):
            return False

        if isinstance(obj1, dict):
            # For dicts, compare values recursively
            if set(obj1.keys()) != set(obj2.keys()):
                return False
            return all(
                ResponseConverter._deep_compare(obj1[k], obj2[k], f"{path}.{k}")
                for k in obj1.keys()
            )

        if isinstance(obj1, (list, tuple)):
            if len(obj1) != len(obj2):
                return False
            # For lists, compare as multisets (order-independent for most cases)
            obj1_str = sorted(str(item) for item in obj1)
            obj2_str = sorted(str(item) for item in obj2)
            return obj1_str == obj2_str

        return obj1 == obj2
