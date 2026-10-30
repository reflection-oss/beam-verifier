"""Chat-completion transport shared by the functional-test suites.

`generate_completion` sends one /v1/chat/completions request in either
streaming or non-streaming mode and normalizes the response to a
`Completion`, so each test is written once and parametrized over the
`stream` fixture.
"""

import dataclasses
import json
from typing import Any, Literal

import jsonschema
import openai
from openai.types import chat

from beam_verifier import options

DEFAULT_MAX_TOKENS = 4096

# Model request options for the run, set by the conftest.
MODEL_KWARGS: dict[str, Any] = {}

# Beam special tokens: must never be visible in API-level text.
SPECIAL_TOKENS = (
    "<|start|>",
    "<|end|>",
    "<|message|>",
    "<|channel|>",
    "<|constrain|>",
    "<|return|>",
    "<|call|>",
    "<|endoftext|>",
)
_MIN_REASONING_LEAK_LENGTH = 40
# Reasoning fields in the Beam chat template.
_REASONING_FIELDS = ("thinking", "reasoning_content", "reasoning")
# Incomplete finish reasons as per the OAI SDK.
_INCOMPLETE_FINISH_REASONS = ("length", "content_filter")


@dataclasses.dataclass
class Completion:
    """A finished assistant turn.

    Attributes:
        content: Assembled text content ("" when absent).
        tool_calls: Tool calls as plain dicts in the OpenAI wire shape
            {"id", "type", "function": {"name", "arguments"}}; empty when the
            response carries none.
        reasoning_content: Assembled reasoning-channel text, None when the
            response carries none.
        finish_reason: The choice's finish_reason (a stream's last reported one).
    """

    content: str
    tool_calls: list[dict[str, Any]]
    reasoning_content: str | None = None
    finish_reason: str | None = None


def generate_completion(
    http_client: openai.OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    *,
    stream: bool,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Literal["none", "auto", "required"] | chat.ChatCompletionNamedToolChoiceParam | None = None,
    response_format: dict[str, Any] | None = None,
    extra_params: dict[str, Any] | None = None,
) -> Completion:
    """Request one chat completion and normalize the response.

    Optional request params left as None are omitted from the request body
    (`max_tokens=None` lifts the default cap and lets the server default
    apply). `extra_params` is merged into the request body last, so
    spec-driven probes can send arbitrary decoding params, winning over the
    session's `MODEL_KWARGS`; both are named as the endpoint spells them,
    whether or not the OpenAI schema has the field. A streaming
    response is consumed here, so request errors surface inside this call in
    both modes. Every completion is checked against the shared
    response-contract invariants (see `assert_response_contract`); a
    non-streaming response is additionally checked for `role == "assistant"`,
    which a stream only carries in its first delta, untracked by the
    accumulator. Token-limit truncation is rejected before validating
    potentially incomplete response fields.
    """
    kwargs: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if response_format is not None:
        kwargs["response_format"] = response_format
    kwargs = options.merge_options(kwargs, MODEL_KWARGS)
    if extra_params:
        kwargs = options.merge_options(kwargs, extra_params)

    response = http_client.chat.completions.create(**kwargs)
    if stream:
        completion = _accumulate_stream(response)
    else:
        choice = response.choices[0]
        message = choice.message
        assert message.role == "assistant"
        completion = Completion(
            content=message.content or "",
            tool_calls=[tool_call.model_dump() for tool_call in message.tool_calls or []],
            reasoning_content=_get_reasoning_text(message),
            finish_reason=choice.finish_reason,
        )
    assert_response_contract(completion, declared_tools=tools, tool_choice=tool_choice)
    return completion


def canonical_tool_call(call: dict[str, Any]) -> tuple[str, str]:
    """Return a hashable identity for one tool call: its name and arguments.

    Arguments arrive as a JSON string, so two calls that differ only in
    whitespace or key order are the same call. Comparing the raw strings
    would count them as distinct; the arguments are parsed and re-emitted in
    a canonical form instead. List order is preserved, being semantic.

    Args:
        call: One tool call in the OpenAI wire shape. Its arguments must
            already satisfy the response contract, which parses them.

    Returns:
        The call's function name paired with its canonical arguments.
    """
    arguments = json.loads(call["function"]["arguments"])
    return call["function"]["name"], json.dumps(arguments, sort_keys=True, separators=(",", ":"))


def assert_response_contract(
    completion: Completion,
    declared_tools: list[dict[str, Any]] | None = None,
    tool_choice: Literal["none", "auto", "required"] | chat.ChatCompletionNamedToolChoiceParam | None = None,
) -> None:
    """Response-contract invariants every completion must satisfy.

    Shared assertions run on every generation the functional tests elicit:
      * `reasoning_content` is present and non-empty: Beam reasons on every
        turn, so a turn with no reasoning means the channel was dropped.
      * reasoning stays in `reasoning_content`: a non-trivial reasoning blob
        duplicated verbatim into `content` means the channels were not split
        correctly.
      * no wire-format special token is ever visible in `content`.
      * `finish_reason` is reported, and is not an incomplete outcome: a
        token-limit truncation or a filtered
        generation is rejected. Which completed value an endpoint reports is
        its own choice, and is not asserted.
      * every surfaced tool call is a fully formed function call (named,
        JSON-object arguments that name each key once): a `max_tokens`
        truncation must never yield a half-formed call.
      * tool-call ids are non-empty and unique within the message, so each
        result can be correlated back to its own call.
      * calls to declared tools satisfy their parameter schemas, including
        rules allowing extra properties. Under automatic choice, unknown tool
        names are permitted: this checks argument representation, not whether
        the client can execute the selected tool. Other choices require a
        declared tool. Values arrive in their declared types, not stringified.
    """
    assert completion.finish_reason is not None, "response contract: no finish_reason reported"
    assert completion.finish_reason not in _INCOMPLETE_FINISH_REASONS, (
        f"Incomplete generation: finish_reason {completion.finish_reason!r}"
    )
    assert completion.reasoning_content, "response contract: assistant turn with no reasoning_content"
    reasoning = completion.reasoning_content or ""
    if len(reasoning) >= _MIN_REASONING_LEAK_LENGTH:
        assert reasoning not in completion.content, "response contract: reasoning_content duplicated inside content"
    for token in SPECIAL_TOKENS:
        assert token not in completion.content, f"response contract: special token {token!r} visible in content"
    for call in completion.tool_calls:
        assert call["type"] == "function", f"response contract: tool call of type {call.get('type')!r}"
        assert call["function"]["name"], "response contract: tool call with no function name"
        try:
            arguments = json.loads(call["function"]["arguments"], object_pairs_hook=_reject_repeated_keys)
        except (TypeError, ValueError) as e:
            raise AssertionError(f"response contract: half-formed tool-call arguments: {e}") from e
        assert isinstance(arguments, dict), "response contract: tool-call arguments are not a JSON object"
    ids = [call["id"] for call in completion.tool_calls]
    assert all(ids), "response contract: tool call with an empty id"
    assert len(set(ids)) == len(ids), f"response contract: tool-call ids are not unique within the message: {ids}"
    if declared_tools is not None:
        declared_schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in declared_tools}
        for call in completion.tool_calls:
            name = call["function"]["name"]
            if name not in declared_schemas:
                assert tool_choice in (None, "auto"), f"response contract: undeclared tool called: {name!r}"
                continue
            schema = declared_schemas[name]
            arguments = json.loads(call["function"]["arguments"])
            try:
                jsonschema.validate(arguments, schema)
            except jsonschema.ValidationError as e:
                raise AssertionError(
                    f"response contract: tool-call arguments violate the declared schema for {name!r}: {e.message}"
                ) from e


def _reject_repeated_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Decodes one JSON object, rejecting a key it names twice.

    `json.loads` otherwise keeps the last value, hiding a parameter written twice.
    """
    keys = [key for key, _ in pairs]
    repeated = sorted({key for key in keys if keys.count(key) > 1})
    assert not repeated, f"response contract: tool-call arguments repeat keys {repeated}"
    return dict(pairs)


def _get_reasoning_text(
    message: chat.ChatCompletionMessage | chat.chat_completion_chunk.ChoiceDelta,
) -> str | None:
    """Reasoning text under whichever spelling the response used, else None."""
    for field in _REASONING_FIELDS:
        value = getattr(message, field, None)
        if value is not None:
            assert isinstance(value, str), f"response contract: {field} must be a string or null"
            if value:
                return value
    return None


def _accumulate_stream(stream: openai.Stream[chat.ChatCompletionChunk]) -> Completion:
    """Accumulates a streaming ChatCompletion response into a Completion."""
    tool_calls_buffer: dict[int, dict[str, Any]] = {}
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason = None
    for chunk in stream:
        if not chunk.choices:  # Usage-only chunks carry no message delta.
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        if choice.finish_reason:
            finish_reason = choice.finish_reason

        if delta.content:
            content_parts.append(delta.content)
        reasoning_delta = _get_reasoning_text(delta)
        if reasoning_delta is not None:
            reasoning_parts.append(reasoning_delta)

        if delta.tool_calls is None:
            continue

        for tool_call_delta in delta.tool_calls:
            call = tool_calls_buffer.setdefault(
                tool_call_delta.index,
                {"id": "", "type": None, "function": {"name": None, "arguments": ""}},
            )
            if tool_call_delta.id:
                call["id"] = tool_call_delta.id
            if tool_call_delta.type:
                call["type"] = tool_call_delta.type
            function_delta = tool_call_delta.function
            if function_delta:
                if function_delta.name:
                    call["function"]["name"] = function_delta.name
                if function_delta.arguments:
                    call["function"]["arguments"] += function_delta.arguments

    return Completion(
        content="".join(content_parts),
        tool_calls=[tool_calls_buffer[index] for index in sorted(tool_calls_buffer)],
        reasoning_content="".join(reasoning_parts) or None,
        finish_reason=finish_reason,
    )
