"""`--model-kwargs` reaches every request the package sends, without displacing a probe's own params."""

import json
import pathlib
from collections.abc import Iterator
from typing import Any

import httpx
import openai
import pytest

from beam_verifier import options, records
from beam_verifier.benchmarks.scaffolds import base
from beam_verifier.benchmarks.scaffolds.inspect_ai import runner as inspect_ai_runner
from beam_verifier.functional_tests import completions
from beam_verifier.functional_tests.raw_completions import test_raw_completions as raw

_FINAL_ANSWER = (
    "<|channel|>analysis<|message|>Thinking.<|end|><|start|>assistant<|channel|>final<|message|>Done.<|return|>"
)


@pytest.fixture
def session_model_kwargs() -> Iterator[dict[str, Any]]:
    """Set the session's model kwargs the way the functional-test conftest does."""
    saved = completions.MODEL_KWARGS
    completions.MODEL_KWARGS = {}
    yield completions.MODEL_KWARGS
    completions.MODEL_KWARGS = saved


def _capturing_client(bodies: list[dict[str, Any]]) -> openai.OpenAI:
    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Paris is in France.",
                            "reasoning_content": "Recall the capital.",
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    return openai.OpenAI(
        base_url="https://example.test/v1",
        api_key="test",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )


@pytest.mark.parametrize(
    "option",
    [
        {"extra_body": {"no_stop_trim": True}},
        {"extra_body": {"include_stop_str_in_output": True}},
        {"extra_body": {"chat_template_kwargs": {"thinking": False}}},
    ],
    ids=["sglang", "vllm", "nested"],
)
def test_model_kwargs_are_sent_with_every_chat_completion(session_model_kwargs, option: dict[str, Any]) -> None:
    bodies: list[dict[str, Any]] = []
    completions.MODEL_KWARGS = option
    with _capturing_client(bodies) as client:
        completions.generate_completion(client, "test", [{"role": "user", "content": "Where is Paris?"}], stream=False)
    sent = option["extra_body"]
    assert bodies[0].items() >= sent.items(), f"Expected {sent} on the wire, got {bodies[0]}"


def test_a_probes_own_params_win_over_the_sessions_model_kwargs(session_model_kwargs) -> None:
    """A request-contract probe must observe the value it set, not a run-wide override."""
    bodies: list[dict[str, Any]] = []
    completions.MODEL_KWARGS = {"temperature": 0.0, "extra_body": {"custom_option": True}}
    with _capturing_client(bodies) as client:
        completions.generate_completion(
            client,
            "test",
            [{"role": "user", "content": "Where is Paris?"}],
            stream=False,
            extra_params={"temperature": 1.5, "extra_body": {"no_stop_trim": True}},
        )
    assert bodies[0]["temperature"] == 1.5, f"Expected the probe's temperature, got {bodies[0]['temperature']}"
    assert bodies[0]["custom_option"] is True, "Expected the session's extra_body key to survive the merge"
    assert bodies[0]["no_stop_trim"] is True, "Expected the probe's extra_body key in the request"


def test_merge_options_merges_nested_mappings_per_key() -> None:
    merged = options.merge_options(
        {"extra_body": {"custom_option": True, "no_stop_trim": True}, "stop": ["A"]},
        {"extra_body": {"custom_option": False}, "stop": ["B"]},
    )
    assert merged == {"extra_body": {"custom_option": False, "no_stop_trim": True}, "stop": ["B"]}


def test_merge_options_does_not_mutate_its_inputs() -> None:
    defaults = {"extra_body": {"custom_option": True}}
    options.merge_options(defaults, {"extra_body": {"custom_option": False}})
    assert defaults == {"extra_body": {"custom_option": True}}


@pytest.mark.parametrize(
    ("pairs", "message"),
    [
        (("no_stop_trim=true",), "not an OpenAI request field"),
        (("extra_body=null",), "extra_body must be a mapping"),
        (("extra_body=5",), "extra_body must be a mapping"),
    ],
    ids=["engine-option-named-flat", "null-extra-body", "scalar-extra-body"],
)
def test_an_option_named_the_wrong_way_is_refused_with_the_spelling_to_use(
    pairs: tuple[str, ...], message: str
) -> None:
    """One spelling each, checked once at the boundary, so no layer has to guess."""
    with pytest.raises(ValueError, match=message):
        options.parse_model_kwargs(pairs)


def test_an_option_named_the_right_way_is_accepted() -> None:
    parsed = options.parse_model_kwargs(("temperature=0.6", "extra_body={no_stop_trim: true}"))
    assert parsed == {"temperature": 0.6, "extra_body": {"no_stop_trim": True}}


def test_an_openai_field_is_accepted_in_extra_body_too() -> None:
    """A library may not implement one and ask for it there; refusing would leave no way to pass it."""
    assert options.parse_model_kwargs(("extra_body={stop: [END]}",)) == {"extra_body": {"stop": ["END"]}}


def test_engine_options_reach_a_flat_body_transport() -> None:
    """A transport that posts the body itself gets `extra_body` unwrapped into it."""
    case = raw.raw_completion_cases["cases"][0]
    bodies: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"text": _FINAL_ANSWER, "finish_reason": "stop"}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        raw.test_raw_completion(
            client, "https://example.test/raw", "test", case, {}, {"extra_body": {"no_stop_trim": True}}
        )
    assert bodies[0]["no_stop_trim"] is True, f"engine option did not reach the wire: {bodies[0]}"
    assert "extra_body" not in bodies[0], "extra_body must not be posted as a literal field"


@pytest.mark.parametrize(
    ("option", "expected"),
    [
        ({"stop": ["END"]}, {"stop_seqs": ["END"]}),
        ({"stop": "END"}, {"stop_seqs": ["END"]}),
        ({"n": 4}, {"num_choices": 4}),
        ({"max_completion_tokens": 256}, {"max_tokens": 256}),
    ],
    ids=["stop-list", "stop-string", "n", "max-completion-tokens"],
)
def test_inspect_ai_renames_the_openai_fields_it_spells_differently(
    option: dict[str, Any], expected: dict[str, Any]
) -> None:
    """`--model-kwargs` speaks the OpenAI schema; InspectAI names a few its own way."""
    runner = inspect_ai_runner.InspectAIRunner(dataset_id="gpqa-diamond", task="inspect_evals/gpqa_diamond")
    kwargs = runner.build_eval_kwargs(
        base.RunRequest(
            base_url="https://example.test/v1",
            model_name="test",
            run_dir=pathlib.Path("/tmp"),
            settings=records.EvalSettings(model_kwargs=option),
        )
    )
    for name, value in expected.items():
        assert kwargs[name] == value, f"expected {name}={value!r}, got {kwargs.get(name)!r}"
    assert not set(option) & set(kwargs), f"the OpenAI name must not survive: {sorted(set(option) & set(kwargs))}"


def test_engine_options_are_not_renamed_on_their_way_to_the_engine() -> None:
    """`extra_body` goes to the engine verbatim, so nothing in it is translated."""
    runner = inspect_ai_runner.InspectAIRunner(dataset_id="gpqa-diamond", task="inspect_evals/gpqa_diamond")
    kwargs = runner.build_eval_kwargs(
        base.RunRequest(
            base_url="https://example.test/v1",
            model_name="test",
            run_dir=pathlib.Path("/tmp"),
            settings=records.EvalSettings(model_kwargs={"extra_body": {"stop": ["END"], "no_stop_trim": True}}),
        )
    )
    assert kwargs["extra_body"]["stop"] == ["END"], "an engine option must keep its own name"
    assert kwargs["extra_body"]["no_stop_trim"] is True
