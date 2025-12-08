"""TransLLM exception classes for conversion errors"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import Provider


class TransLLMError(Exception):
    """Base exception for all TransLLM errors"""

    pass


class ConversionError(TransLLMError):
    """Raised when conversion between formats fails"""

    def __init__(
        self,
        message: str,
        from_provider: "Provider",
        to_provider: "Provider",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.from_provider = from_provider.value
        self.to_provider = to_provider.value
        self.details = details or {}


class UnsupportedProviderError(TransLLMError):
    """Raised when an unsupported provider is requested"""

    def __init__(self, provider: "Provider", supported_providers: list[str]) -> None:
        self.provider = provider.value
        self.supported_providers = supported_providers
        message = (
            f"Unsupported provider: '{self.provider}'. "
            f"Supported providers: {', '.join(supported_providers)}"
        )
        super().__init__(message)


class UnsupportedFeatureError(TransLLMError):
    """Raised when a feature is not supported by the target provider"""

    def __init__(
        self,
        feature: str,
        provider: "Provider",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.feature = feature
        self.provider = provider.value
        self.details = details or {}
        message = f"Provider '{self.provider}' does not support feature: '{feature}'"
        super().__init__(message)


class ValidationError(TransLLMError):
    """Raised when data validation fails"""

    def __init__(
        self,
        message: str,
        validation_errors: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.validation_errors = validation_errors or []


class IdempotencyError(TransLLMError):
    """Raised when idempotency test fails (A -> IR -> A)"""

    def __init__(
        self,
        original_data: dict[str, Any],
        final_data: dict[str, Any],
        differences: list[str],
    ) -> None:
        self.original_data = original_data
        self.final_data = final_data
        self.differences = differences
        message = (
            "Idempotency test failed. Data changed after round-trip conversion:\n"
            + "\n".join(differences)
        )
        super().__init__(message)
