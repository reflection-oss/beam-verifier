"""The tool-choice probes reject incorrect partner responses over both transports."""

import copy
import json
from typing import Any

import httpx
import openai
import pytest

from beam_verifier.functional_tests.e2e import test_tool_protocol as protocol

_REASONING = "I should check the current weather before answering this question."


def _message(name: str | None = None, content: str = "", *, arguments: str = '{"location":"Paris"}') -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "reasoning_content": "think",
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": name, "arguments": arguments}}]
        if name
        else [],
    }


def _response(message: dict[str, Any], finish: str, *, stream: bool) -> httpx.Response:
    if not stream:
        return httpx.Response(200, json={"choices": [{"index": 0, "message": message, "finish_reason": finish}]})
    delta = {**message, "tool_calls": [{"index": i, **call} for i, call in enumerate(message["tool_calls"])]}
    events = [
        {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
    ]
    data = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=data)


def _client(message: dict[str, Any], finish: str, choice: object, *, stream: bool) -> openai.OpenAI:
    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert body["tool_choice"] == choice
        assert body["stream"] is stream
        return _response(message, finish, stream=stream)

    return openai.OpenAI(
        base_url="https://example.test/v1",
        api_key="test",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )


@pytest.mark.parametrize("stream", [False, True])
def test_named_probe_rejects_another_declared_function(stream: bool) -> None:
    case = protocol.tool_protocol_test_cases["named_function"][0]
    with _client(_message("get_weather"), "tool_calls", case["input"]["tool_choice"], stream=stream) as client:
        with pytest.raises(AssertionError, match="Expected only 'get_local_time'"):
            protocol.test_tool_choice_named(client, "test", stream, case)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("finish", ["tool_calls", "stop"])
def test_named_probe_accepts_the_selected_function(stream: bool, finish: str) -> None:
    case = protocol.tool_protocol_test_cases["named_function"][0]
    with _client(_message("get_local_time"), finish, case["input"]["tool_choice"], stream=stream) as client:
        protocol.test_tool_choice_named(client, "test", stream, case)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "response_reasoning",
    [
        {"reasoning": _REASONING},
        {"reasoning_content": _REASONING},
        {"thinking": _REASONING},
        {"reasoning": "", "reasoning_content": _REASONING},
        {"thinking": _REASONING, "reasoning_content": "think"},
    ],
    ids=["reasoning", "reasoning-content", "thinking", "empty-alias", "template-order"],
)
def test_roundtrip_replays_returned_reasoning(stream: bool, response_reasoning: dict[str, str]) -> None:
    """Whichever spelling comes back is replayed under the canonical one.

    The chat template renders from `thinking`, `reasoning_content` or
    `reasoning`, so the next turn needs only the name it prefers.
    """
    case = protocol.tool_protocol_test_cases["tool_roundtrip"][0]
    without_reasoning = {k: v for k, v in _message("get_weather").items() if k != "reasoning_content"}
    first = {**without_reasoning, **response_reasoning}
    expected = next(text for text in response_reasoning.values() if text)

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is stream
        if body["tool_choice"] == "required":
            return _response(first, "tool_calls", stream=stream)
        assert body["tool_choice"] == "none"
        assert body["messages"][-2] == {**without_reasoning, "reasoning_content": expected, "content": None}
        assert body["messages"][-1] == {"role": "tool", "tool_call_id": "call_1", "content": case["tool_result"]}
        return _response(_message(content=case["tool_result"]), "stop", stream=stream)

    with openai.OpenAI(
        base_url="https://example.test/v1",
        api_key="test",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        protocol.test_tool_roundtrip(client, "test", stream, case)


@pytest.mark.parametrize("stream", [False, True])
def test_turn_with_both_reasoning_fields_empty_is_rejected(stream: bool) -> None:
    """An endpoint that empties both reasoning spellings dropped the channel."""
    case = protocol.tool_protocol_test_cases["automatic_choice"][0]
    message = {**_message(content="Paris is in France."), "reasoning": "", "reasoning_content": ""}
    with _client(message, "stop", "auto", stream=stream) as client:
        with pytest.raises(AssertionError, match="no reasoning_content"):
            protocol.test_tool_choice_auto(client, "test", stream, case, {})


@pytest.mark.parametrize("stream", [False, True])
def test_empty_reasoning_alias_does_not_hide_leaked_reasoning(stream: bool) -> None:
    case = protocol.tool_protocol_test_cases["automatic_choice"][0]
    message = {**_message(content=_REASONING), "reasoning": "", "reasoning_content": _REASONING}
    with _client(message, "stop", "auto", stream=stream) as client:
        with pytest.raises(AssertionError, match="reasoning_content duplicated"):
            protocol.test_tool_choice_auto(client, "test", stream, case, {})


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("message", "finish"),
    [(_message(content="Paris is in France."), "stop"), (_message("get_weather"), "tool_calls")],
)
def test_auto_probe_accepts_text_or_tools(stream: bool, message: dict[str, Any], finish: str) -> None:
    case = protocol.tool_protocol_test_cases["automatic_choice"][0]
    with _client(message, finish, "auto", stream=stream) as client:
        protocol.test_tool_choice_auto(client, "test", stream, case, {})


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("message", "finish", "error"),
    [
        (_message(), "stop", "Expected text or tool calls"),
        (_message(content="Partial answer"), "length", "Incomplete generation"),
    ],
)
def test_auto_probe_rejects_empty_or_truncated_output(
    stream: bool, message: dict[str, Any], finish: str, error: str
) -> None:
    case = protocol.tool_protocol_test_cases["automatic_choice"][0]
    with _client(message, finish, "auto", stream=stream) as client, pytest.raises(AssertionError, match=error):
        protocol.test_tool_choice_auto(client, "test", stream, case, {})


@pytest.mark.parametrize("stream", [False, True])
def test_required_probe_rejects_text_only(stream: bool) -> None:
    case = protocol.tool_protocol_test_cases["test_tool_choice_required"]["test_params"]["case"][0]
    with _client(_message(content="Hello"), "stop", "required", stream=stream) as client:
        with pytest.raises(AssertionError, match="Expected tool call"):
            protocol.test_tool_choice_required(client, "test", stream, case)


@pytest.mark.parametrize("stream", [False, True])
def test_none_probe_rejects_tool_calls(stream: bool) -> None:
    case = protocol.tool_protocol_test_cases["test_tool_choice_none"]["test_params"]["case"][0]
    with _client(_message("get_weather"), "tool_calls", "none", stream=stream) as client:
        with pytest.raises(AssertionError, match="Expected no tool calls"):
            protocol.test_tool_choice_none(client, "test", stream, case)


@pytest.mark.parametrize("stream", [False, True])
def test_none_probe_reports_truncation_as_incomplete(stream: bool) -> None:
    case = protocol.tool_protocol_test_cases["test_tool_choice_none"]["test_params"]["case"][0]
    with _client(_message(content="Partial answer"), "length", "none", stream=stream) as client:
        with pytest.raises(AssertionError, match="Incomplete generation"):
            protocol.test_tool_choice_none(client, "test", stream, case)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "message",
    [_message(content="Partial answer"), _message("get_weather"), _message("get_weather", arguments='{"location":')],
    ids=["text", "complete-call", "partial-arguments"],
)
def test_required_probe_reports_truncation_as_incomplete(stream: bool, message: dict[str, Any]) -> None:
    case = protocol.tool_protocol_test_cases["test_tool_choice_required"]["test_params"]["case"][0]
    with _client(message, "length", "required", stream=stream) as client:
        with pytest.raises(AssertionError, match="Incomplete generation"):
            protocol.test_tool_choice_required(client, "test", stream, case)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "message",
    [
        _message(content="Partial answer"),
        _message("get_local_time"),
        _message("get_local_time", arguments='{"location":'),
    ],
    ids=["text", "complete-call", "partial-arguments"],
)
def test_named_probe_reports_truncation_as_incomplete(stream: bool, message: dict[str, Any]) -> None:
    case = protocol.tool_protocol_test_cases["named_function"][0]
    with _client(message, "length", case["input"]["tool_choice"], stream=stream) as client:
        with pytest.raises(AssertionError, match="Incomplete generation"):
            protocol.test_tool_choice_named(client, "test", stream, case)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "message",
    [_message(content="Partial answer"), _message("get_weather"), _message("get_weather", arguments='{"location":')],
    ids=["text", "complete-call", "partial-arguments"],
)
def test_roundtrip_probe_reports_first_turn_truncation_as_incomplete(stream: bool, message: dict[str, Any]) -> None:
    case = protocol.tool_protocol_test_cases["tool_roundtrip"][0]
    with _client(message, "length", "required", stream=stream) as client:
        with pytest.raises(AssertionError, match="Incomplete generation"):
            protocol.test_tool_roundtrip(client, "test", stream, case)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("placement", ["content", "nested_argument"])
def test_auto_probe_rejects_parameter_markup(stream: bool, placement: str) -> None:
    marker = '<rfl:parameter name="'
    case = copy.deepcopy(protocol.tool_protocol_test_cases["automatic_choice"][0])
    if placement == "content":
        message = _message(content=f"Leaked {marker}")
    else:
        case["input"]["tools"][0]["function"]["parameters"]["properties"]["location"] = {}
        value = {"nested": [marker]}
        message = _message("get_weather", arguments=json.dumps({"location": value}))
    with _client(message, "stop", "auto", stream=stream) as client:
        with pytest.raises(AssertionError, match="Expected every call parsed|Expected one parameter per value"):
            protocol.test_tool_choice_auto(client, "test", stream, case, {})


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "message", [_message(content="I will list the files."), _message("bash", arguments='{"command": "pwd"}')]
)
def test_harness_probe_accepts_text_or_undeclared_tool(stream: bool, message: dict[str, Any]) -> None:
    case = protocol.harness_test_cases["cases"][0]
    with _client(message, "stop", "auto", stream=stream) as client:
        protocol.test_tool_choice_auto(client, "test", stream, case, {})
