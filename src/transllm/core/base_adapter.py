"""Base adapter class for all provider adapters"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import (
        CoreRequest,
        CoreResponse,
        StreamEvent,
        Message,
        Provider,
    )
from .aliases import ProviderAliases


class BaseAdapter(ABC):
    """Base class for all provider adapters

    Each adapter must implement to_unified() and from_unified() methods
    to convert between provider-specific format and the brand-neutral IR.
    """

    def __init__(self, provider_name: Provider) -> None:
        self.provider_name_original = provider_name
        self.provider_name = provider_name.value.lower()
        self.aliases = ProviderAliases.get_provider_aliases(self.provider_name)
        self.reverse_aliases = ProviderAliases.get_reverse_mapping(self.provider_name)

        # Streaming state management (aligned with LiteLLM design)
        self._stream_sequence_id = 0
        self._stream_start_time: float | None = None

    @abstractmethod
    def to_unified_request(self, data: dict[str, Any]) -> CoreRequest:
        """Convert provider-specific request to unified IR format"""
        pass

    @abstractmethod
    def from_unified_request(self, unified_request: CoreRequest) -> dict[str, Any]:
        """Convert unified IR request to provider-specific format"""
        pass

    @abstractmethod
    def to_unified_response(self, data: dict[str, Any]) -> CoreResponse:
        """Convert provider-specific response to unified IR format"""
        pass

    @abstractmethod
    def from_unified_response(self, unified_response: CoreResponse) -> dict[str, Any]:
        """Convert unified IR response to provider-specific format"""
        pass

    def to_unified_message(self, data: dict[str, Any]) -> Message:
        """Convert provider-specific message to unified IR format"""
        # Default implementation - subclasses should override if needed
        role = data.get("role", "user")
        content = data.get("content", "")

        return Message(
            role=role,
            content=content,
            metadata=data.get("metadata"),
            tool_calls=data.get("tool_calls"),
            identifier=data.get("id"),
        )

    def from_unified_message(self, unified_message: Message) -> dict[str, Any]:
        """Convert unified IR message to provider-specific format"""
        # Default implementation - subclasses should override if needed
        return {
            "role": unified_message.role,
            "content": unified_message.content,
            "metadata": unified_message.metadata,
            "tool_calls": unified_message.tool_calls,
            "id": unified_message.identifier,
        }

    def reset_stream_state(self) -> None:
        """Reset streaming state for a new stream session.

        Call this method when starting a new stream to reset the
        sequence_id and timestamp counters.
        """
        self._stream_sequence_id = 0
        self._stream_start_time = None

    def to_unified_stream_event(self, data: dict[str, Any]) -> StreamEvent:
        """Convert provider-specific stream event to unified IR format.

        Automatically manages sequence_id and timestamp internally,
        following LiteLLM's design pattern for simplified streaming APIs.

        Args:
            data: Provider-specific stream event data

        Returns:
            Unified StreamEvent

        Example:
            # Simple usage - automatic state management
            for event in stream:
                unified = adapter.to_unified_stream_event(event)
        """
        # Auto-generate sequence_id
        sequence_id = self._stream_sequence_id
        self._stream_sequence_id += 1

        # Auto-generate timestamp
        if self._stream_start_time is None:
            self._stream_start_time = time.time()
        timestamp = time.time() - self._stream_start_time

        # Call implementation method (subclasses should override _to_unified_stream_event_impl)
        return self._to_unified_stream_event_impl(data, sequence_id, timestamp)

    def _to_unified_stream_event_impl(
        self,
        data: dict[str, Any],
        sequence_id: int,
        timestamp: float
    ) -> StreamEvent:
        """Internal implementation for stream event conversion.

        Subclasses should override this method instead of to_unified_stream_event()
        to maintain automatic state management.

        Args:
            data: Provider-specific stream event data
            sequence_id: Event sequence number (auto-generated)
            timestamp: Event timestamp (auto-generated)

        Returns:
            Unified StreamEvent
        """
        # Default implementation - extracts from data dict
        return StreamEvent(
            type=data.get("type", "content_delta"),
            sequence_id=sequence_id,
            timestamp=timestamp,
            content_delta=data.get("content_delta"),
            tool_call_delta=data.get("tool_call_delta"),
            tool_call=data.get("tool_call"),
            finish_reason=data.get("finish_reason"),
            content_index=data.get("content_index"),
            error=data.get("error"),
            metadata=data.get("metadata"),
        )

    def from_unified_stream_event(self, unified_event: StreamEvent) -> dict[str, Any]:
        """Convert unified IR stream event to provider-specific format"""
        # Default implementation - subclasses should override if needed
        return {
            "type": unified_event.type,
            "sequence_id": unified_event.sequence_id,
            "timestamp": unified_event.timestamp,
            "content_delta": unified_event.content_delta,
            "tool_call_delta": unified_event.tool_call_delta,
            "tool_call": unified_event.tool_call,
            "finish_reason": unified_event.finish_reason,
            "content_index": unified_event.content_index,
            "error": unified_event.error,
            "metadata": unified_event.metadata,
        }

    def map_field_to_unified(self, field_name: str) -> str:
        """Map provider field name to unified field name"""
        return self.aliases.get(field_name, field_name)

    def map_field_from_unified(self, unified_field_name: str) -> str:
        """Map unified field name to provider field name"""
        return self.reverse_aliases.get(unified_field_name, unified_field_name)

    def validate_conversion_feasibility(
        self,
        from_provider: Provider,
        to_provider: Provider,
        data: dict[str, Any],
    ) -> None:
        """Validate if conversion is feasible between providers

        This checks for capability mismatches and unsupported features.
        Subclasses should override to provide provider-specific checks.

        Args:
            from_provider: Source provider (Provider enum)
            to_provider: Target provider (Provider enum)
            data: Data to validate
        """
        # Default: no additional validation
        pass

    def check_idempotency(
        self,
        original_data: dict[str, Any],
        final_data: dict[str, Any],
        data_type: str,
    ) -> bool:
        """Check if conversion is idempotent (A -> IR -> A)"""
        # Simple recursive comparison
        return self._deep_compare(original_data, final_data)

    def _deep_compare(self, obj1: Any, obj2: Any, path: str = "") -> bool:
        """Deep comparison of two objects, ignoring ordering of lists"""
        if type(obj1) != type(obj2):
            return False

        if isinstance(obj1, dict):
            if set(obj1.keys()) != set(obj2.keys()):
                return False
            return all(
                self._deep_compare(obj1[k], obj2[k], f"{path}.{k}")
                for k in obj1.keys()
            )

        if isinstance(obj1, (list, tuple)):
            if len(obj1) != len(obj2):
                return False
            # For lists, we compare as multisets (order-independent)
            # Create a list of serialized forms for sorting
            obj1_serialized = []
            obj2_serialized = []
            for item1, item2 in zip(obj1, obj2):
                # For dict items, we need to serialize consistently
                if isinstance(item1, dict) and isinstance(item2, dict):
                    # Sort keys and serialize consistently
                    sorted_keys = sorted(set(item1.keys()) | set(item2.keys()))
                    serialized1 = json.dumps({k: item1.get(k) for k in sorted_keys}, sort_keys=True)
                    serialized2 = json.dumps({k: item2.get(k) for k in sorted_keys}, sort_keys=True)
                    obj1_serialized.append(serialized1)
                    obj2_serialized.append(serialized2)
                else:
                    # For non-dict items, use string representation
                    obj1_serialized.append(json.dumps(item1, sort_keys=True))
                    obj2_serialized.append(json.dumps(item2, sort_keys=True))
            return sorted(obj1_serialized) == sorted(obj2_serialized)

        return obj1 == obj2


class RequestAdapter(BaseAdapter):
    """Adapter specialized for request conversion"""

    @abstractmethod
    def to_unified_request(self, data: dict[str, Any]) -> CoreRequest:
        """Convert provider request to unified IR"""
        pass

    @abstractmethod
    def from_unified_request(self, unified_request: CoreRequest) -> dict[str, Any]:
        """Convert unified IR request to provider format"""
        pass


class ResponseAdapter(BaseAdapter):
    """Adapter specialized for response conversion"""

    @abstractmethod
    def to_unified_response(self, data: dict[str, Any]) -> CoreResponse:
        """Convert provider response to unified IR"""
        pass

    @abstractmethod
    def from_unified_response(self, unified_response: CoreResponse) -> dict[str, Any]:
        """Convert unified IR response to provider format"""
        pass


class StreamAdapter(BaseAdapter):
    """Adapter specialized for streaming event conversion"""

    @abstractmethod
    def to_unified_stream_event(self, data: dict[str, Any]) -> StreamEvent:
        """Convert provider stream event to unified IR"""
        pass

    @abstractmethod
    def from_unified_stream_event(self, unified_event: StreamEvent) -> dict[str, Any]:
        """Convert unified IR stream event to provider format"""
        pass
