"""Tool-protocol tests: tool-call shape, tool_choice handling, and tool-result grounding."""

import json
import pathlib

import pytest

from beam_verifier.functional_tests import completions, load

tool_protocol_test_cases = load.load_test_cases("tool_protocol", pathlib.Path(__file__).parent / "fixtures")

harness_test_cases = load.load_test_cases("harness_tool_calls", pathlib.Path(__file__).parent / "fixtures")

# Beam's argument markup, which a parsed call never shows.
PARAMETER_MARKUP = ('<rfl:parameter name="', "</rfl:parameter>")


def _strings(value: object) -> list[str]:
    """Every string inside a decoded JSON value."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        strings: list[str] = []
        for item in value:
            strings.extend(_strings(item))
        return strings
    return []


@pytest.mark.parametrize(
    "case",
    tool_protocol_test_cases["test_tool_call_shape"]["test_params"]["case"],
    ids=load.case_id_generator,
)
def test_tool_call_shape(http_client, model, stream, case):
    """The prompt calls for a tool call: output expect a structurally valid call (schema-valid arguments)."""
    completion = completions.generate_completion(
        http_client,
        model,
        case["input"]["messages"],
        stream=stream,
        tools=case["input"]["tools"],
    )

    assert completion.tool_calls, "Expected the prompt to elicit at least one tool call"


@pytest.mark.parametrize(
    "case",
    tool_protocol_test_cases["test_parallel_tool_calls"]["test_params"]["case"],
    ids=load.case_id_generator,
)
def test_parallel_tool_calls(http_client, model, stream, case):
    """The prompt ask for multiple tasks — the output expects 2+ parallel distinct tool calls."""
    completion = completions.generate_completion(
        http_client,
        model,
        case["input"]["messages"],
        stream=stream,
        tools=case["input"]["tools"],
    )

    calls = completion.tool_calls
    min_tool_calls = case["expected"]["min_tool_calls"]
    assert len(calls) >= min_tool_calls, f"Expected at least {min_tool_calls} tool calls in one turn, got {len(calls)}"
    distinct_calls = {completions.canonical_tool_call(call) for call in calls}
    assert len(distinct_calls) >= min_tool_calls, (
        f"Expected {min_tool_calls} distinct calls, but the turn repeats the same task: {calls}"
    )


@pytest.mark.parametrize(
    "case",
    tool_protocol_test_cases["test_tool_choice_required"]["test_params"]["case"],
    ids=load.case_id_generator,
)
def test_tool_choice_required(http_client, model, stream, case):
    """tool_choice=required produces at least one declared, schema-valid tool call."""
    completion = completions.generate_completion(
        http_client,
        model,
        case["input"]["messages"],
        stream=stream,
        tools=case["input"]["tools"],
        tool_choice=case["input"]["tool_choice"],
    )

    assert completion.tool_calls, "Expected tool call when tool_choice='required'"


@pytest.mark.parametrize(
    "case",
    tool_protocol_test_cases["test_tool_choice_none"]["test_params"]["case"],
    ids=load.case_id_generator,
)
def test_tool_choice_none(http_client, model, stream, case):
    """The request forbids tool calls (tool_choice=none) — the output expects an answer with content only."""
    completion = completions.generate_completion(
        http_client,
        model,
        case["input"]["messages"],
        stream=stream,
        tools=case["input"]["tools"],
        tool_choice=case["input"]["tool_choice"],
    )

    assert not completion.tool_calls, "Expected no tool calls when tool_choice='none'"
    assert completion.content, "Expected content when tool_choice='none'"


@pytest.mark.parametrize("case", tool_protocol_test_cases["named_function"], ids=load.case_id_generator)
def test_tool_choice_named(http_client, model, stream, case):
    """A named tool_choice produces calls only to the selected function."""
    completion = completions.generate_completion(
        http_client,
        model,
        case["input"]["messages"],
        stream=stream,
        tools=case["input"]["tools"],
        tool_choice=case["input"]["tool_choice"],
    )
    selected = case["input"]["tool_choice"]["function"]["name"]
    assert completion.tool_calls, f"Expected a call to the selected function {selected!r}"
    actual_names = [call["function"]["name"] for call in completion.tool_calls]
    assert all(name == selected for name in actual_names), f"Expected only {selected!r}, got {actual_names!r}"


@pytest.mark.parametrize(
    ("case", "generation_options"),
    [pytest.param(case, {}, id=load.case_id_generator(case)) for case in tool_protocol_test_cases["automatic_choice"]]
    + [
        pytest.param(
            case,
            {
                "max_tokens": harness_test_cases["request"]["max_tokens"],
                "extra_params": {"reasoning_effort": harness_test_cases["request"]["reasoning_effort"]},
            },
            id=load.case_id_generator(case),
        )
        for case in harness_test_cases["cases"]
    ],
)
def test_tool_choice_auto(http_client, model, stream, case, generation_options):
    """Automatic choice yields text or schema-valid tool calls without leaked parameter markup."""
    completion = completions.generate_completion(
        http_client,
        model,
        case["input"]["messages"],
        stream=stream,
        tools=case["input"]["tools"],
        tool_choice="auto",
        **generation_options,
    )

    assert completion.tool_calls or completion.content.strip(), "Expected text or tool calls when tool_choice='auto'"

    for marker in PARAMETER_MARKUP:
        assert marker not in completion.content, f"Expected every call parsed, got {marker!r} in content"
    for call in completion.tool_calls:
        for value in _strings(json.loads(call["function"]["arguments"])):
            if any(token in value for token in completions.SPECIAL_TOKENS):
                # A fake turn the model wrote into the value; no grammar rules it out.
                continue
            for marker in PARAMETER_MARKUP:
                assert marker not in value, (
                    f"Expected one parameter per value, got {marker!r} inside a {call['function']['name']!r} argument"
                )


@pytest.mark.parametrize("case", tool_protocol_test_cases["tool_roundtrip"], ids=load.case_id_generator)
def test_tool_roundtrip(http_client, model, stream, case):
    """Smoke-test continuation with returned call IDs and one city's supplied weather.

    This does not test correlation between distinct results for multiple cities.
    """
    first = completions.generate_completion(
        http_client,
        model,
        case["input"]["messages"],
        stream=stream,
        tools=case["input"]["tools"],
        tool_choice="required",
    )
    assert first.tool_calls, "Expected a generated tool call before submitting its result"
    assistant = {"role": "assistant", "content": first.content or None, "tool_calls": first.tool_calls}
    if first.reasoning_content is not None:
        # Always the canonical name: engines drop fields outside their schema.
        assistant["reasoning_content"] = first.reasoning_content
    replies = [
        {"role": "tool", "tool_call_id": call["id"], "content": case["tool_result"]} for call in first.tool_calls
    ]
    followup = completions.generate_completion(
        http_client,
        model,
        [*case["input"]["messages"], assistant, *replies],
        stream=stream,
        tools=case["input"]["tools"],
        tool_choice="none",
    )
    assert not followup.tool_calls, "Expected an answer after supplying the tool result with tool_choice='none'"
    assert all(value.lower() in followup.content.lower() for value in case["expected"]["answer"]), (
        f"Expected tool-result facts {case['expected']['answer']!r}, got {followup.content!r}"
    )


@pytest.mark.parametrize(
    "case",
    tool_protocol_test_cases["test_tool_result_grounding"]["test_params"]["case"],
    ids=load.case_id_generator,
)
def test_tool_result_grounding(http_client, model, stream, case):
    """The history carries a tool call and its result — the output is expected to answer from that result."""
    completion = completions.generate_completion(
        http_client,
        model,
        case["input"]["messages"],
        stream=stream,
        tools=case["input"]["tools"],
    )

    content = completion.content.lower()
    expected_answers = case["expected"]["answer"]
    assert any(answer.lower() in content for answer in expected_answers), (
        f"Expected one of {expected_answers} in content, but got: '{completion.content}'"
    )
