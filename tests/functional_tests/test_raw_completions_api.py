"""Raw endpoint failures and truncation must not become conformance passes."""

import json
from typing import Any, Literal

import httpx
import pydantic
import pytest

from beam_verifier.functional_tests import raw_completions_api
from beam_verifier.functional_tests.raw_completions import test_raw_completions as raw
from beam_verifier.functional_tests.raw_completions.harmony_xml import probes, structure

_FINAL = "<|channel|>analysis<|message|>think<|end|><|start|>assistant<|channel|>final<|message|>4<|return|>"
_BUILTIN_CALL = (
    "<|channel|>analysis<|message|>think<|end|>"
    "<|start|>assistant to=bash<|channel|>analysis <|constrain|>xml<|message|>"
    '<rfl:parameter name="command">\nprintf 4\n</rfl:parameter><|call|>'
)


def test_generate_preserves_url_body_and_control_tokens() -> None:
    url = "https://example.test/custom/completions?revision=2"
    body: dict[str, object] = {"prompt": "<|start|>assistant", "model": "test-model", "skip_special_tokens": False}

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == url
        assert request.method == "POST"
        assert request.headers["authorization"] == "Bearer test-key"
        assert json.loads(request.content) == body
        return httpx.Response(200, json={"choices": [{"text": _FINAL, "finish_reason": "stop"}]})

    with httpx.Client(transport=httpx.MockTransport(respond), headers={"Authorization": "Bearer test-key"}) as client:
        completion = raw_completions_api.generate_completion(client, url, body)
    assert completion.text == _FINAL
    result = structure.validate_completion(completion.text, finish_reason=completion.finish_reason)
    assert result.ok
    assert result.shape == probes.CompletionShape.FINAL_ONLY


@pytest.mark.parametrize("options", [{"no_stop_trim": True}, {"include_stop_str_in_output": True}])
def test_raw_probe_sends_and_records_endpoint_options(options: dict[str, object]) -> None:
    case = raw.raw_completion_cases["cases"][0]
    expected = {**case["request"], **options, "model": "test-model"}
    metadata: dict[str, object] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == expected
        return httpx.Response(200, json={"choices": [{"text": _FINAL, "finish_reason": "stop"}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        raw.test_raw_completion(client, "https://example.test/raw", "test-model", case, metadata, options)
    assert metadata["request"] == expected


def test_validate_completion_accepts_builtin_tool_context() -> None:
    completion = raw_completions_api.Completion(text=_BUILTIN_CALL, finish_reason="stop")

    result = structure.validate_completion(
        completion.text, tools=["bash"], builtin_tools=("bash",), finish_reason=completion.finish_reason
    )

    assert result.ok
    assert result.shape == probes.CompletionShape.SINGLE_TOOL
    assert result.facets.builtin_tool


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": []},
        {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]},
        {"choices": [{"text": 123, "finish_reason": "stop"}]},
        {"choices": [{"text": _FINAL, "finish_reason": None}]},
    ],
)
def test_malformed_raw_response_cannot_pass(payload: dict[str, object]) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client, pytest.raises(pydantic.ValidationError):
        raw_completions_api.generate_completion(client, "https://example.test/raw", {"prompt": "test"})


@pytest.mark.parametrize("status", [401, 404, 500])
def test_endpoint_errors_are_not_skipped(status: int) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "unavailable"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client, pytest.raises(httpx.HTTPStatusError):
        raw_completions_api.generate_completion(client, "https://example.test/raw", {"prompt": "test"})


def test_token_limit_cannot_pass_even_with_a_complete_frame() -> None:
    completion = raw_completions_api.Completion(text=_FINAL, finish_reason="length")
    result = structure.validate_completion(completion.text, finish_reason=completion.finish_reason)
    assert not result.ok
    assert result.truncated
    assert result.shape == probes.CompletionShape.FINAL_ONLY


@pytest.mark.parametrize("text", ["", "<|start|>assistant", "<|start|>user<|message|>hello<|end|>"])
def test_empty_or_prompt_like_output_cannot_pass(text: str) -> None:
    result = structure.validate_completion(text, finish_reason="stop")
    assert not result.ok


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_first_failure_is_preserved(finish_reason: Literal["stop", "length"]) -> None:
    result = structure.validate_completion("<|channel|>final<|message|>4<|return|>", finish_reason=finish_reason)
    assert result.truncated == (finish_reason == "length")
    assert result.failure is not None
    assert result.failure.probe_id == probes.ProbeId.MANDATORY_FIRST_ANALYSIS
    assert result.failure.message == "The first assistant message must be analysis."


def _replay_client(answer: str, *, reasoning: str = "think") -> httpx.Client:
    def respond(request: httpx.Request) -> httpx.Response:
        text = _FINAL.replace("<|message|>think<|end|>", f"<|message|>{reasoning}<|end|>")
        text = text.replace("<|message|>4<|return|>", f"<|message|>{answer}<|return|>")
        return httpx.Response(200, json={"choices": [{"text": text, "finish_reason": "stop"}]})

    return httpx.Client(transport=httpx.MockTransport(respond))


def _recall_case() -> dict[str, Any]:
    """The fixture case whose final answer must repeat a word from its reasoning."""
    return next(case for case in raw.raw_completion_cases["cases"] if "expected_final_content" in case)


# The word itself lives in the fixture, so renaming it there needs no edit here.
_RECALL_WORD = _recall_case()["expected_final_content"][0]
# Shares no substring with any plausible codeword, so a near miss stays a miss.
_WRONG_WORD = "zzzz"


@pytest.mark.parametrize("answer", [_WRONG_WORD, "I cannot recall it.", ""], ids=["wrong", "refused", "empty"])
def test_raw_reasoning_replay_requires_the_expected_word(answer: str) -> None:
    with _replay_client(answer) as client, pytest.raises(AssertionError, match="Reasoning recall failed"):
        raw.test_raw_completion(client, "https://example.test/raw", "test", _recall_case(), {}, {})


def test_raw_reasoning_replay_cannot_pass_with_the_word_only_in_analysis() -> None:
    with (
        _replay_client("I cannot recall it.", reasoning=f"The codeword is {_RECALL_WORD}.") as client,
        pytest.raises(AssertionError, match="Reasoning recall failed"),
    ):
        raw.test_raw_completion(client, "https://example.test/raw", "test", _recall_case(), {}, {})


@pytest.mark.parametrize(
    "answer",
    [
        _RECALL_WORD,
        f"{_RECALL_WORD}.",
        f"The codeword is {_RECALL_WORD}.",
        _RECALL_WORD.capitalize(),
        _RECALL_WORD.upper(),
    ],
    ids=["bare", "punctuated", "in-a-sentence", "capitalised", "shouted"],
)
def test_raw_reasoning_replay_accepts_the_word_in_any_casing(answer: str) -> None:
    """The recalled word carries the casing of the sentence it lands in."""
    with _replay_client(answer) as client:
        raw.test_raw_completion(client, "https://example.test/raw", "test", _recall_case(), {}, {})
