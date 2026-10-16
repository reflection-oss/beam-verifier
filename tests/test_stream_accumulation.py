"""Offline tests for streamed tool-call accumulation (no network).

The streaming protocol identifies each tool call by its `index`: `id` and
`function.name` usually arrive only in a call's first fragment, and fragments
for parallel calls may interleave.
"""

import pytest
from openai.types.chat import chat_completion_chunk

from beam_verifier.functional_tests import completions


def _fragment(index, *, call_id=None, name=None, arguments=None, reported_type="function"):
    # model_construct, not the validating constructor: a nonconformant server
    # can report any `type`, and the SDK parses responses leniently, so tests
    # must be able to build the values verification is supposed to catch.
    return chat_completion_chunk.ChoiceDeltaToolCall.model_construct(
        index=index,
        id=call_id,
        type=reported_type,
        function=chat_completion_chunk.ChoiceDeltaToolCallFunction(name=name, arguments=arguments),
    )


def _stream(*fragments_per_chunk, finish_reason="tool_calls"):
    """One chunk per argument, each carrying that chunk's tool-call fragments."""
    chunks = []
    for fragments in fragments_per_chunk:
        chunks.append(
            chat_completion_chunk.ChatCompletionChunk(
                id="chunk",
                created=0,
                model="test-model",
                object="chat.completion.chunk",
                choices=[
                    chat_completion_chunk.Choice(
                        index=0,
                        delta=chat_completion_chunk.ChoiceDelta(role="assistant", tool_calls=list(fragments)),
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
                    index=0, delta=chat_completion_chunk.ChoiceDelta(), finish_reason=finish_reason
                )
            ],
        )
    )
    return chunks


def test_parallel_calls_whose_later_fragments_omit_ids():
    completion = completions._accumulate_stream(
        _stream(
            [_fragment(0, call_id="call_a", name="get_weather")],
            [_fragment(0, arguments='{"location":')],
            [_fragment(0, arguments=' "Paris"}')],
            [_fragment(1, call_id="call_b", name="get_local_time")],
            [_fragment(1, arguments='{"location":')],
            [_fragment(1, arguments=' "Tokyo"}')],
        )
    )
    assert [call["id"] for call in completion.tool_calls] == ["call_a", "call_b"]
    assert [call["function"]["name"] for call in completion.tool_calls] == ["get_weather", "get_local_time"]
    assert completion.tool_calls[0]["function"]["arguments"] == '{"location": "Paris"}'
    assert completion.tool_calls[1]["function"]["arguments"] == '{"location": "Tokyo"}'


def test_interleaved_fragments_do_not_cross_contaminate():
    # Both calls announce themselves before either finishes streaming arguments.
    completion = completions._accumulate_stream(
        _stream(
            [_fragment(0, call_id="call_a", name="get_weather")],
            [_fragment(1, call_id="call_b", name="get_local_time")],
            [_fragment(0, arguments='{"location": "Paris"}')],
            [_fragment(1, arguments='{"location": "Tokyo"}')],
        )
    )
    assert completion.tool_calls[0]["id"] == "call_a"
    assert completion.tool_calls[0]["function"]["arguments"] == '{"location": "Paris"}'
    assert completion.tool_calls[1]["id"] == "call_b"
    assert completion.tool_calls[1]["function"]["arguments"] == '{"location": "Tokyo"}'


def test_calls_materialize_in_index_order():
    # A provider may open the second call's fragments first.
    completion = completions._accumulate_stream(
        _stream(
            [_fragment(1, call_id="call_b", name="get_local_time", arguments="{}")],
            [_fragment(0, call_id="call_a", name="get_weather", arguments="{}")],
        )
    )
    assert [call["id"] for call in completion.tool_calls] == ["call_a", "call_b"]


def test_calls_without_ids_still_materialize():
    # An endpoint that never sends ids must not silently lose its calls: the
    # empty ids then fail the response contract loudly.
    completion = completions._accumulate_stream(
        _stream(
            [_fragment(0, name="get_weather", arguments="{}")],
            [_fragment(1, name="get_local_time", arguments="{}")],
        )
    )
    assert len(completion.tool_calls) == 2
    assert [call["id"] for call in completion.tool_calls] == ["", ""]


def test_several_fragments_in_one_chunk():
    completion = completions._accumulate_stream(
        _stream([_fragment(0, call_id="call_a", name="f", arguments="{}"), _fragment(1, call_id="call_b", name="g")])
    )
    assert [call["id"] for call in completion.tool_calls] == ["call_a", "call_b"]


def test_content_and_finish_reason_still_accumulate():
    chunks = _stream([_fragment(0, call_id="call_a", name="f", arguments="{}")], finish_reason="tool_calls")
    chunks[0].choices[0].delta.content = "thinking out loud"
    completion = completions._accumulate_stream(chunks)
    assert completion.content == "thinking out loud"
    assert completion.finish_reason == "tool_calls"


def test_reported_tool_call_type_is_preserved():
    completion = completions._accumulate_stream(
        _stream([_fragment(0, call_id="call_a", name="f", arguments="{}", reported_type="function")])
    )
    assert completion.tool_calls[0]["type"] == "function"


def test_invalid_reported_type_fails_verification():
    completion = completions._accumulate_stream(
        _stream([_fragment(0, call_id="call_a", name="f", arguments="{}", reported_type="custom_tool")])
    )
    assert completion.tool_calls[0]["type"] == "custom_tool"
    completion.reasoning_content = "think"
    with pytest.raises(AssertionError, match="tool call of type"):
        completions.assert_response_contract(completion)


def test_missing_reported_type_fails_verification():
    # A stream that never reports a type must not have one synthesized for it.
    completion = completions._accumulate_stream(
        _stream([_fragment(0, call_id="call_a", name="f", arguments="{}", reported_type=None)])
    )
    assert completion.tool_calls[0]["type"] is None
    completion.reasoning_content = "think"
    with pytest.raises(AssertionError, match="tool call of type"):
        completions.assert_response_contract(completion)
