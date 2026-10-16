"""Check the parser contract using an httpx client over in-process HTTP."""

import dataclasses
import json
from collections.abc import Callable

import httpx
import pydantic
import pytest

from beam_verifier.functional_tests import native_api, parser_api


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer test-key"},
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://example.test/custom/parse?version=2",
        "https://example.test/proxy/parse_function_call?version=2",
        "https://example.test/proxy/v1/chat/completions/derender?version=2",
    ],
)
def test_parser_uses_exact_url_and_preserves_argument_types(url: str) -> None:
    requests: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        assert request.method == "POST"
        assert str(request.url) == url
        assert json.loads(request.content) == {"tokens": [151669, 42], "tools": []}
        return httpx.Response(
            200,
            json={
                "role": "assistant",
                "reasoning_content": "Think",
                "content": None,
                "tool_calls": [{"type": "function", "function": {"name": "compute", "arguments": '{"n": "1"}'}}],
            },
        )

    with _client(respond) as client:
        result = parser_api.Parser(client, url).parse_message([151669, 42], [])
    assert requests == [url]
    assert dataclasses.asdict(result) == {
        "reasoning_content": "Think",
        "content": "",
        "tool_calls": [{"name": "compute", "arguments": {"n": "1"}}],
    }


def test_parser_forwards_tools_and_optional_model() -> None:
    tools = [{"type": "function", "function": {"name": "compute"}}]

    def respond(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"tokens": [42], "tools": tools, "model": "test-model"}
        return httpx.Response(200, json={"role": "assistant", "content": "Answer"})

    with _client(respond) as client:
        result = parser_api.Parser(client, "https://example.test/parse", "test-model").parse_message([42], tools)
    assert result == parser_api.ParsedMessage(reasoning_content="", content="Answer", tool_calls=[])


@pytest.mark.parametrize("status", [404, 405, 501])
def test_missing_operation_is_unsupported(status: int) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "unavailable"})

    with _client(respond) as client, pytest.raises(native_api.UnsupportedOperation, match=f"HTTP {status}"):
        parser_api.Parser(client, "https://example.test/parse").parse_message([42], [])


@pytest.mark.parametrize("status", [400, 401, 500])
def test_request_errors_are_not_skipped(status: int) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "broken"})

    with _client(respond) as client, pytest.raises(httpx.HTTPStatusError) as caught:
        parser_api.Parser(client, "https://example.test/parse").parse_message([42], [])
    assert caught.value.response.status_code == status


@pytest.mark.parametrize(
    "message",
    [
        {"text": "detokenized wire is not a parsed message"},
        {"role": "user", "content": "wrong role"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": "{bad json"}}],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": "[]"}}],
        },
    ],
)
def test_invalid_assistant_response_fails(message: dict[str, object]) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=message)

    with _client(respond) as client, pytest.raises(pydantic.ValidationError):
        parser_api.Parser(client, "https://example.test/parse").parse_message([42], [])


def test_json_equal_accepts_matching_nested_json_types() -> None:
    assert parser_api.json_equal({"enabled": True}, {"enabled": True})
    assert parser_api.json_equal(1.0, 1)
    assert parser_api.json_equal(
        {"items": [{"score": 1.0, "label": "München 🧪"}]},
        {"items": [{"score": 1, "label": "München 🧪"}]},
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (
            {"obj": {"items": [{"label": "München 🧪", "enabled": True}]}},
            {"obj": {"items": [{"label": "München 🧪", "enabled": 1}]}},
        ),
        (0, False),
        ("1", 1),
        ({"enabled": True}, {"active": True}),
        ([1, 2], [1]),
    ],
)
def test_json_equal_rejects_different_json_types_or_shapes(left: object, right: object) -> None:
    assert not parser_api.json_equal(left, right)
