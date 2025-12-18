"""Lightweight helper for calling Anthropic's Messages APIs."""

from __future__ import annotations

import copy
from typing import Any

import httpx

from ...converters.request_converter import RequestConverter
from ...core.schema import Provider

DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_API_VERSION = "2023-06-01"


class AnthropicMessagesClient:
    """Synchronous client for Anthropic's Messages and count_tokens endpoints."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_version: str = DEFAULT_API_VERSION,
        timeout: float | httpx.Timeout | None = 30.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.api_version = api_version
        self.timeout = timeout
        self._client = http_client

    def send_message(
        self,
        request: dict[str, Any],
        *,
        from_provider: Provider = Provider.anthropic,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Call POST /v1/messages with an Anthropic-formatted request."""
        payload = self._normalize_request(request, from_provider)
        headers = self._build_headers(payload, extra_headers)

        return self._post_json(f"{self.base_url}/v1/messages", payload, headers)

    def count_message_tokens(
        self,
        request: dict[str, Any],
        *,
        from_provider: Provider = Provider.anthropic,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Call POST /v1/messages/count_tokens to estimate token usage."""
        payload = self._normalize_request(request, from_provider)
        headers = self._build_headers(payload, extra_headers)

        return self._post_json(
            f"{self.base_url}/v1/messages/count_tokens", payload, headers
        )

    def _normalize_request(
        self, request: dict[str, Any], from_provider: Provider
    ) -> dict[str, Any]:
        """Convert request to Anthropic format if it originated from another provider."""
        if from_provider != Provider.anthropic:
            return RequestConverter.convert(request, from_provider, Provider.anthropic)

        return copy.deepcopy(request)

    def _build_headers(
        self, payload: dict[str, Any], extra_headers: dict[str, str] | None
    ) -> dict[str, str]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "content-type": "application/json",
        }

        betas = payload.get("betas")
        if betas:
            if isinstance(betas, (list, tuple, set)):
                beta_header = ",".join(str(item) for item in betas if item)
            else:
                beta_header = str(betas)
            if beta_header:
                headers["anthropic-beta"] = beta_header

        if extra_headers:
            headers.update(extra_headers)

        return headers

    def _post_json(
        self, url: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
        requester = self._client.post if self._client else httpx.post
        response = requester(url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
