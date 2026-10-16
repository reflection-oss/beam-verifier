"""Shared transport for non-generating conformance endpoints."""

from __future__ import annotations

from typing import cast

import httpx
import pydantic


class UnsupportedOperation(RuntimeError):
    """The server does not expose a requested conformance operation."""


def post(client: httpx.Client, url: str, body: dict[str, object]) -> dict[str, object]:
    """POST JSON, distinguishing absent routes from request and server failures."""
    response = client.post(url, json=body)
    if response.status_code in (404, 405, 501):
        raise UnsupportedOperation(f"UNSUPPORTED: {url} returned HTTP {response.status_code}")
    response.raise_for_status()
    return cast(dict[str, object], response.json())


class Tokenization(pydantic.BaseModel):
    """Validated token IDs for one nonempty prompt."""

    tokens: list[int] = pydantic.Field(min_length=1)
