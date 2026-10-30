"""Provider-neutral HTTP formatting against pinned Harmony token IDs."""

from __future__ import annotations

import httpx

from beam_verifier.functional_tests import native_api


class Formatter:
    """Send chat inputs to an exact URL returning formatted token IDs.

    The caller owns the HTTP client. Requests use the endpoint's formatter;
    no template override or backend-specific field translation is applied.
    """

    def __init__(self, client: httpx.Client, url: str, model: str | None = None) -> None:
        self.client = client
        self.url = url
        self.model = model

    def chat_request(self, request: dict[str, object]) -> dict[str, object]:
        """Preserve fixture fields and add a model only when one was supplied."""
        return {**request, "model": self.model} if self.model is not None else dict(request)

    def render(self, request: dict[str, object]) -> list[int]:
        """Format chat messages using the supplied endpoint."""
        payload = native_api.post(self.client, self.url, self.chat_request(request))
        return native_api.Tokenization.model_validate(payload, strict=True).tokens
