"""Offline tests for streamed tool-call accumulation (no network).

The streaming protocol identifies each tool call by its `index`: `id` and
`function.name` usually arrive only in a call's first fragment, and fragments
for parallel calls may interleave.
"""

import httpx
import openai
import pytest
from openai.types.chat import chat_completion_chunk

from beam_verifier.functional_tests import completions


def _fragment(
    index: int,
    *,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
    reported_type: str | None = "function",
) -> chat_completion_chunk.ChoiceDeltaToolCall:
    # model_construct, not the validating constructor: a nonconformant server
    # can report any `type`, and the SDK parses responses leniently, so tests
    # must be able to build the values verification is supposed to catch.
    return chat_completion_chunk.ChoiceDeltaToolCall.model_construct(
        index=index,
        id=call_id,
        type=reported_type,
        function=chat_completion_chunk.ChoiceDeltaToolCallFunction(name=name, arguments=arguments),
    )


def _stream(
    *fragments_per_chunk: list[chat_completion_chunk.ChoiceDeltaToolCall],
    content: str | None = None,
) -> completions.Completion:
    """Send SSE through the real SDK and the public completion gateway, offline."""
    chunks = []
    for index, fragments in enumerate(fragments_per_chunk):
        chunks.append(
            chat_completion_chunk.ChatCompletionChunk(
                id="chunk",
                created=0,
                model="test-model",
                object="chat.completion.chunk",
                choices=[
                    chat_completion_chunk.Choice(
                        index=0,
                        delta=chat_completion_chunk.ChoiceDelta(
                            role="assistant", content=content if index == 0 else None, tool_calls=list(fragments)
                        ),
                    )
                ],
            )
        )
    chunks.append(
        chat_completion_chunk.ChatCompletionChunk(
            id="chunk",
            created=0,
            model="test-model",
            object="chat.completion.chunk",
            choices=[
                chat_completion_chunk.Choice(
                    index=0,
                    delta=chat_completion_chunk.ChoiceDelta(reasoning_content="think"),
                    finish_reason="tool_calls",
                )
            ],
        )
    )
    payload = "".join(f"data: {chunk.model_dump_json()}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=payload)
    )
    with openai.OpenAI(
        base_url="http://example.test/v1", api_key="synthetic-key", http_client=httpx.Client(transport=transport)
    ) as client:
        return completions.generate_completion(
            client, "model", [{"role": "user", "content": "Use the tools."}], stream=True
        )


def test_parallel_calls_whose_later_fragments_omit_ids():
    completion = _stream(
        [_fragment(0, call_id="call_a", name="get_weather")],
        [_fragment(0, arguments='{"location":')],
        [_fragment(0, arguments=' "Paris"}')],
        [_fragment(1, call_id="call_b", name="get_local_time")],
        [_fragment(1, arguments='{"location":')],
        [_fragment(1, arguments=' "Tokyo"}')],
    )
    assert [call["id"] for call in completion.tool_calls] == ["call_a", "call_b"]
    assert [call["function"]["name"] for call in completion.tool_calls] == ["get_weather", "get_local_time"]
    assert completion.tool_calls[0]["function"]["arguments"] == '{"location": "Paris"}'
    assert completion.tool_calls[1]["function"]["arguments"] == '{"location": "Tokyo"}'


def test_interleaved_fragments_do_not_cross_contaminate():
    # Both calls announce themselves before either finishes streaming arguments.
    completion = _stream(
        [_fragment(0, call_id="call_a", name="get_weather")],
        [_fragment(1, call_id="call_b", name="get_local_time")],
        [_fragment(0, arguments='{"location": "Paris"}')],
        [_fragment(1, arguments='{"location": "Tokyo"}')],
    )
    assert completion.tool_calls[0]["id"] == "call_a"
    assert completion.tool_calls[0]["function"]["arguments"] == '{"location": "Paris"}'
    assert completion.tool_calls[1]["id"] == "call_b"
    assert completion.tool_calls[1]["function"]["arguments"] == '{"location": "Tokyo"}'


def test_calls_materialize_in_index_order():
    # A provider may open the second call's fragments first.
    completion = _stream(
        [_fragment(1, call_id="call_b", name="get_local_time", arguments="{}")],
        [_fragment(0, call_id="call_a", name="get_weather", arguments="{}")],
    )
    assert [call["id"] for call in completion.tool_calls] == ["call_a", "call_b"]


def test_stream_rejects_missing_call_ids() -> None:
    # Missing IDs must fail visibly, rather than silently dropping the calls.
    with pytest.raises(AssertionError, match="empty id"):
        _stream([_fragment(0, name="get_weather", arguments="{}"), _fragment(1, name="get_local_time", arguments="{}")])


def test_several_fragments_in_one_chunk():
    completion = _stream(
        [
            _fragment(0, call_id="call_a", name="f", arguments="{}"),
            _fragment(1, call_id="call_b", name="g", arguments="{}"),
        ]
    )
    assert [call["id"] for call in completion.tool_calls] == ["call_a", "call_b"]


def test_stream_preserves_early_content_and_final_finish_reason() -> None:
    completion = _stream([_fragment(0, call_id="call_a", name="f", arguments="{}")], [], content="thinking out loud")
    assert completion.content == "thinking out loud"
    assert completion.reasoning_content == "think"
    assert completion.finish_reason == "tool_calls"


def test_reported_tool_call_type_is_preserved():
    completion = _stream([_fragment(0, call_id="call_a", name="f", arguments="{}", reported_type="function")])
    assert completion.tool_calls[0]["type"] == "function"


@pytest.mark.parametrize("reported_type", ["custom_tool", None], ids=["invalid", "missing"])
def test_stream_rejects_invalid_tool_call_type(reported_type: str | None) -> None:
    with pytest.raises(AssertionError, match="tool call of type"):
        _stream([_fragment(0, call_id="call_a", name="f", arguments="{}", reported_type=reported_type)])
