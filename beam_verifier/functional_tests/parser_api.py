"""Provider-neutral HTTP parsing of fixed Harmony output tokens."""

from __future__ import annotations

import dataclasses
from typing import Literal

import httpx
import pydantic

from beam_verifier.functional_tests import native_api


@dataclasses.dataclass(frozen=True)
class ToolCall:
    """A parsed function name and decoded JSON argument object."""

    name: str
    arguments: dict[str, object]


@dataclasses.dataclass(frozen=True)
class ParsedMessage:
    """Assistant content, reasoning, and ordered tool calls."""

    reasoning_content: str
    content: str
    tool_calls: list[ToolCall]


def json_equal(left: object, right: object) -> bool:
    """Whether two decoded JSON values are equal with matching JSON types.

    Numbers compare numerically, while booleans remain distinct from numbers
    at every nesting level.
    """
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if isinstance(left, int | float) and isinstance(right, int | float):
        return left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(json_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(json_equal(a, b) for a, b in zip(left, right, strict=True))
    return type(left) is type(right) and left == right


class _Function(pydantic.BaseModel):  # pydantic: justified - nested external JSON response validation.
    """Validate function fields and decode JSON-encoded arguments."""

    name: str
    arguments: pydantic.Json[dict[str, object]]


class _ChatToolCall(pydantic.BaseModel):
    type: Literal["function"]
    function: _Function


class _ChatMessage(pydantic.BaseModel):
    role: Literal["assistant"]
    content: str | None
    reasoning_content: str | None = None
    tool_calls: list[_ChatToolCall] | None = None


class Parser:
    """Send tokens and tool schemas to an exact URL returning an assistant message.

    The caller owns the HTTP client. The endpoint owns detokenization, parsing,
    and argument coercion; this adapter only decodes the response JSON.
    """

    def __init__(self, client: httpx.Client, url: str, model: str | None = None) -> None:
        self.client = client
        self.url = url
        self.model = model

    def parse_message(self, tokens: list[int], tools: list[dict[str, object]]) -> ParsedMessage:
        """Parse pinned output IDs without generation or local wire repair."""
        body: dict[str, object] = {"tokens": tokens, "tools": tools}
        if self.model is not None:
            body["model"] = self.model
        payload = native_api.post(self.client, self.url, body)
        result = _ChatMessage.model_validate(payload, strict=True)
        return ParsedMessage(
            reasoning_content=result.reasoning_content or "",
            content=result.content or "",
            tool_calls=[
                ToolCall(name=call.function.name, arguments=call.function.arguments) for call in result.tool_calls or []
            ],
        )
