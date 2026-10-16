"""Offline tests for the shared response-contract assertions (no network)."""

import copy
import json

import pytest

from beam_verifier.functional_tests import completions


def _completion(**overrides):
    fields = {"content": "fine", "tool_calls": [], "reasoning_content": "think first", "finish_reason": "stop"}
    fields.update(overrides)
    return completions.Completion(**fields)


def test_contract_accepts_clean_completion():
    completions.assert_response_contract(_completion())


def test_contract_rejects_missing_reasoning():
    with pytest.raises(AssertionError, match="no reasoning_content"):
        completions.assert_response_contract(_completion(reasoning_content=None))


def test_contract_rejects_reasoning_duplicated_into_content():
    reasoning = "step by step reasoning that is long enough to be a real chain of thought"
    bad = _completion(reasoning_content=reasoning, content=f"{reasoning}\nThe answer is 4.")
    with pytest.raises(AssertionError, match="reasoning_content duplicated"):
        completions.assert_response_contract(bad)


def test_contract_short_reasoning_echo_is_not_a_leak():
    completions.assert_response_contract(_completion(reasoning_content="It is Earth.", content="It is Earth."))


def test_contract_rejects_special_token_in_content():
    with pytest.raises(AssertionError, match="special token"):
        completions.assert_response_contract(_completion(content="the answer<|return|>"))


def test_contract_rejects_incomplete_generation():
    with pytest.raises(AssertionError, match="Incomplete generation"):
        completions.assert_response_contract(_completion(finish_reason="length"))


def test_contract_rejects_a_filtered_generation():
    with pytest.raises(AssertionError, match="Incomplete generation"):
        completions.assert_response_contract(_completion(finish_reason="content_filter"))


def test_contract_rejects_a_response_that_never_reported_how_it_ended():
    with pytest.raises(AssertionError, match="no finish_reason reported"):
        completions.assert_response_contract(_completion(finish_reason=None))


@pytest.mark.parametrize("finish_reason", ["stop", "tool_calls"])
def test_contract_accepts_either_completed_finish_reason(finish_reason: str):
    """A finished tool call is spec-legal as either value; engines differ."""
    completions.assert_response_contract(_completion(finish_reason=finish_reason))


def test_contract_rejects_half_formed_tool_call():
    truncated = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "SF'}}
    with pytest.raises(AssertionError, match="half-formed"):
        completions.assert_response_contract(_completion(finish_reason="tool_calls", tool_calls=[truncated]))


def test_contract_rejects_unnamed_tool_call():
    unnamed = {"id": "c1", "type": "function", "function": {"name": None, "arguments": "{}"}}
    with pytest.raises(AssertionError, match="no function name"):
        completions.assert_response_contract(_completion(finish_reason="tool_calls", tool_calls=[unnamed]))


def test_contract_accepts_well_formed_tool_call():
    call = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "SF"}'}}
    completions.assert_response_contract(_completion(finish_reason="tool_calls", tool_calls=[call]))


def _call(call_id="c1", name="get_weather", arguments='{"location": "SF"}'):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}


def test_contract_rejects_empty_tool_call_id():
    with pytest.raises(AssertionError, match="empty id"):
        completions.assert_response_contract(_completion(finish_reason="tool_calls", tool_calls=[_call(call_id="")]))


def test_contract_rejects_duplicate_tool_call_ids():
    calls = [_call(), _call(arguments='{"location": "Tokyo"}')]
    with pytest.raises(AssertionError, match="not unique"):
        completions.assert_response_contract(_completion(finish_reason="tool_calls", tool_calls=calls))


@pytest.mark.parametrize(
    "tool_choice",
    ["none", "required", {"type": "function", "function": {"name": "get_weather"}}],
    ids=["none", "required", "named"],
)
def test_contract_rejects_undeclared_tool(tool_choice):
    with pytest.raises(AssertionError, match="undeclared tool"):
        completions.assert_response_contract(
            _completion(finish_reason="tool_calls", tool_calls=[_call(name="get_time")]),
            declared_tools=[WEATHER_TOOL],
            tool_choice=tool_choice,
        )


@pytest.mark.parametrize("tool_choice", [None, "auto"])
def test_contract_accepts_undeclared_tool_under_automatic_choice(tool_choice):
    completions.assert_response_contract(
        _completion(tool_calls=[_call(name="bash", arguments='{"command": "pwd"}')]),
        declared_tools=[WEATHER_TOOL],
        tool_choice=tool_choice,
    )


@pytest.mark.parametrize(
    ("schema_options", "extra_arguments", "valid"),
    [
        ({"additionalProperties": {"type": "string"}}, {"unit": "celsius"}, True),
        ({"additionalProperties": False}, {"unit": "celsius"}, False),
        (
            {"additionalProperties": False, "patternProperties": {"^x_": {"type": "string"}}},
            {"x_unit": "celsius"},
            True,
        ),
    ],
)
def test_contract_respects_schema_rules_for_extra_arguments(schema_options, extra_arguments, valid):
    tool = copy.deepcopy(WEATHER_TOOL)
    tool["function"]["parameters"].update(schema_options)
    completion = _completion(tool_calls=[_call(arguments=json.dumps({"location": "SF", **extra_arguments}))])
    if valid:
        completions.assert_response_contract(completion, declared_tools=[tool])
    else:
        with pytest.raises(AssertionError, match="violate the declared schema"):
            completions.assert_response_contract(completion, declared_tools=[tool])


def test_contract_rejects_repeated_parameter():
    repeated = _call(arguments='{"location": "SF", "location": "Tokyo"}')
    with pytest.raises(AssertionError, match=r"repeat keys \['location'\]"):
        completions.assert_response_contract(_completion(finish_reason="tool_calls", tool_calls=[repeated]))


def test_contract_rejects_arguments_violating_declared_schema():
    wrongly_typed = _call(arguments='{"location": 42}')
    with pytest.raises(AssertionError, match="violate the declared schema"):
        completions.assert_response_contract(
            _completion(finish_reason="tool_calls", tool_calls=[wrongly_typed]), declared_tools=[WEATHER_TOOL]
        )


def test_contract_accepts_declared_schema_valid_call():
    completions.assert_response_contract(
        _completion(finish_reason="tool_calls", tool_calls=[_call()]), declared_tools=[WEATHER_TOOL]
    )
