"""Stream format converter"""

from __future__ import annotations

from typing import Any, Dict

from src.transllm.core.schema import Provider, StreamEvent
from ..core import BaseAdapter
from ..utils.provider_registry import ProviderRegistry
from ..core.exceptions import ConversionError, UnsupportedProviderError


class StreamConverter:
    """Converts stream events between different provider formats

    This class provides a unified interface for converting stream events from one
    provider format to another, similar to RequestConverter but for streaming data.

    The converter automatically manages state (sequence_id, timestamp) for each provider,
    ensuring proper ordering and timing of stream events.
    """

    def __init__(self) -> None:
        """Initialize StreamConverter with empty state dictionary"""
        # State tracking for each provider: {provider: {sequence_id, start_time}}
        self._adapter_states: Dict[Provider, Dict[str, Any]] = {}
        # Cache adapter instances to maintain state across stream events
        self._cached_adapters: Dict[Provider, BaseAdapter] = {}

    def convert_stream_event(
        self,
        data: Dict[str, Any],
        from_provider: Provider,
        to_provider: Provider,
    ) -> Dict[str, Any]:
        """Convert stream event from one provider format to another

        This is a convenience method that combines to_unified_event() and
        from_unified_event() in a single call.

        Args:
            data: Stream event data in source provider format
            from_provider: Source provider (Provider enum)
            to_provider: Target provider (Provider enum)

        Returns:
            Stream event data in target provider format

        Examples:
            >>> converter = StreamConverter()
            >>> event = converter.convert_stream_event(
            ...     {"choices": [{"delta": {"content": "Hello"}}]},
            ...     Provider.openai,
            ...     Provider.anthropic
            ... )

        Raises:
            UnsupportedProviderError: If provider is not supported
            ConversionError: If conversion fails
        """
        # Convert to unified format
        unified_event = self.to_unified_event(data, from_provider)

        # Convert to target format
        target_event = self.from_unified_event(unified_event, to_provider)

        return target_event

    def to_unified_event(
        self,
        data: Dict[str, Any],
        from_provider: Provider,
    ) -> StreamEvent:
        """Convert stream event from provider format to unified format

        Args:
            data: Stream event data in provider format
            from_provider: Source provider (Provider enum)

        Returns:
            Stream event in unified format (StreamEvent)

        Raises:
            UnsupportedProviderError: If provider is not supported
            ConversionError: If conversion fails
        """
        # Get or create cached adapter (maintains state across events)
        try:
            from_adapter = self._get_cached_adapter(from_provider)
        except UnsupportedProviderError:
            raise UnsupportedProviderError(
                from_provider,
                ProviderRegistry.list_supported_providers(),
            )

        try:
            # Ensure we have state for this provider
            self._ensure_provider_state(from_provider)

            # Let the adapter handle the conversion (it will manage state internally)
            unified_event = from_adapter.to_unified_stream_event(data)

            return unified_event

        except Exception as e:
            if isinstance(e, ConversionError):
                raise
            raise ConversionError(
                f"Failed to convert stream event from {from_provider.value} to unified format",
                from_provider,
                None,  # to_provider is not applicable for this step
                {"original_error": str(e)},
            ) from e

    def from_unified_event(
        self,
        unified_event: StreamEvent,
        to_provider: Provider,
    ) -> Dict[str, Any]:
        """Convert stream event from unified format to provider format

        Args:
            unified_event: Stream event in unified format
            to_provider: Target provider (Provider enum)

        Returns:
            Stream event data in target provider format

        Raises:
            UnsupportedProviderError: If provider is not supported
            ConversionError: If conversion fails
        """
        # Get or create cached adapter (maintains state across events)
        try:
            to_adapter = self._get_cached_adapter(to_provider)
        except UnsupportedProviderError:
            raise UnsupportedProviderError(
                to_provider,
                ProviderRegistry.list_supported_providers(),
            )

        try:
            # Convert from unified format
            target_event = to_adapter.from_unified_stream_event(unified_event)

            return target_event

        except Exception as e:
            if isinstance(e, ConversionError):
                raise
            raise ConversionError(
                f"Failed to convert stream event from unified format to {to_provider.value}",
                None,  # from_provider is not applicable for this step
                to_provider,
                {"original_error": str(e)},
            ) from e

    def reset_stream_state(self, provider: Provider) -> None:
        """Reset stream state for a specific provider

        This should be called when starting a new stream session with the same provider.

        Args:
            provider: Provider to reset state for
        """
        if provider in self._adapter_states:
            del self._adapter_states[provider]
        if provider in self._cached_adapters:
            # Reset the cached adapter's internal state
            self._cached_adapters[provider].reset_stream_state()
            # Remove from cache to get a fresh adapter next time
            del self._cached_adapters[provider]

    def reset_all_states(self) -> None:
        """Reset stream state for all providers"""
        self._adapter_states.clear()
        # Reset all cached adapters and clear cache
        for adapter in self._cached_adapters.values():
            adapter.reset_stream_state()
        self._cached_adapters.clear()

    def _ensure_provider_state(self, provider: Provider) -> None:
        """Ensure provider state is initialized

        Args:
            provider: Provider to check
        """
        if provider not in self._adapter_states:
            self._adapter_states[provider] = {
                "_stream_sequence_id": 0,
                "_stream_start_time": None,
            }

    def _get_cached_adapter(self, provider: Provider) -> BaseAdapter:
        """Get or create a cached adapter instance for a provider

        This ensures we use the same adapter instance across multiple stream events,
        which is critical for maintaining proper sequence_id and timestamp state.

        Args:
            provider: Provider to get adapter for

        Returns:
            Cached adapter instance

        Raises:
            UnsupportedProviderError: If provider is not supported
        """
        if provider not in self._cached_adapters:
            # Create new adapter and cache it
            if not ProviderRegistry.is_supported(provider):
                raise UnsupportedProviderError(
                    provider,
                    ProviderRegistry.list_supported_providers(),
                )
            adapter_class = ProviderRegistry._adapters[provider.value.lower()]
            self._cached_adapters[provider] = adapter_class(provider)

        return self._cached_adapters[provider]

    def get_provider_state(self, provider: Provider) -> Dict[str, Any]:
        """Get the current state for a provider

        Args:
            provider: Provider to get state for

        Returns:
            Dictionary containing sequence_id and start_time

        Raises:
            KeyError: If provider state hasn't been initialized
        """
        if provider not in self._adapter_states:
            raise KeyError(
                f"Provider state not initialized for {provider.value}. "
                "Call to_unified_event() first to initialize state."
            )
        return self._adapter_states[provider]

    def check_idempotency(
        self,
        data: Dict[str, Any],
        provider: Provider,
    ) -> bool:
        """Check if stream event conversion is idempotent (A -> IR -> A)

        Note: This checks if a stream event can be converted to unified format
        and back to the same provider format without loss of information.

        Args:
            data: Stream event data
            provider: Provider to test (Provider enum)

        Returns:
            True if idempotent, False otherwise
        """
        try:
            # Convert to unified format
            unified_event = self.to_unified_event(data, provider)

            # Convert back to provider format
            converted_back = self.from_unified_event(unified_event, provider)

            # Compare (simple deep comparison)
            return self._deep_compare(data, converted_back)

        except Exception:
            return False

    @staticmethod
    def _deep_compare(obj1: Any, obj2: Any, path: str = "") -> bool:
        """Deep comparison of two objects with enum handling

        This is a simplified comparison for stream events that may not preserve
        all metadata like sequence_id and timestamp.

        Args:
            obj1: First object
            obj2: Second object
            path: Current path in the object tree (for debugging)

        Returns:
            True if objects are equivalent, False otherwise
        """
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
            # For dicts, compare values recursively, ignoring metadata fields
            if set(obj1.keys()) != set(obj2.keys()):
                return False
            return all(
                StreamConverter._deep_compare(obj1[k], obj2[k], f"{path}.{k}")
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
