"""Offline tests for the shared response-contract assertions (no network)."""

import dataclasses
import json
import pathlib
from typing import Any

import pytest
import yaml

from beam_verifier.functional_tests import completions

_CASES = yaml.safe_load((pathlib.Path(__file__).parent / "fixtures/chat.yaml").read_text())["invalid_responses"]


@pytest.mark.parametrize("case", _CASES, ids=[case["id"] for case in _CASES])
def test_contract_rejects_invalid_response(case: dict[str, Any], completion: completions.Completion) -> None:
    with pytest.raises(AssertionError, match=case["error"]):
        completions.assert_response_contract(
            dataclasses.replace(completion, **case["response"]), declared_tools=case.get("declared_tools")
        )


def test_contract_short_reasoning_echo_is_not_a_leak(completion: completions.Completion):
    completions.assert_response_contract(
        dataclasses.replace(completion, reasoning_content="It is Earth.", content="It is Earth.")
    )


@pytest.mark.parametrize("finish_reason", ["stop", "tool_calls"])
def test_contract_accepts_either_completed_finish_reason(finish_reason: str, completion: completions.Completion):
    """A finished tool call is spec-legal as either value; engines differ."""
    completions.assert_response_contract(dataclasses.replace(completion, finish_reason=finish_reason))


def test_contract_accepts_well_formed_tool_call(completion: completions.Completion, tool_call: dict[str, Any]):
    completions.assert_response_contract(
        dataclasses.replace(completion, finish_reason="tool_calls", tool_calls=[tool_call])
    )


@pytest.mark.parametrize(
    "tool_choice",
    ["none", "required", {"type": "function", "function": {"name": "get_weather"}}],
    ids=["none", "required", "named"],
)
def test_contract_rejects_undeclared_tool(
    tool_choice: str | dict[str, dict[str, str]] | None,
    completion: completions.Completion,
    tool_call: dict[str, Any],
    weather_tool: dict[str, Any],
):
    tool_call["function"]["name"] = "get_time"
    completion.tool_calls = [tool_call]
    completion.finish_reason = "tool_calls"
    with pytest.raises(AssertionError, match="undeclared tool"):
        completions.assert_response_contract(completion, declared_tools=[weather_tool], tool_choice=tool_choice)


@pytest.mark.parametrize("tool_choice", [None, "auto"])
def test_contract_accepts_undeclared_tool_under_automatic_choice(
    tool_choice: str | dict[str, dict[str, str]] | None,
    completion: completions.Completion,
    tool_call: dict[str, Any],
    weather_tool: dict[str, Any],
):
    tool_call["function"] = {"name": "bash", "arguments": '{"command": "pwd"}'}
    completion.tool_calls = [tool_call]
    completions.assert_response_contract(completion, declared_tools=[weather_tool], tool_choice=tool_choice)


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
def test_contract_respects_schema_rules_for_extra_arguments(
    schema_options: dict[str, object],
    extra_arguments: dict[str, str],
    valid: bool,
    completion: completions.Completion,
    tool_call: dict[str, Any],
    weather_tool: dict[str, Any],
):
    weather_tool["function"]["parameters"].update(schema_options)
    tool_call["function"]["arguments"] = json.dumps({"location": "SF", **extra_arguments})
    completion.tool_calls = [tool_call]
    if valid:
        completions.assert_response_contract(completion, declared_tools=[weather_tool])
    else:
        with pytest.raises(AssertionError, match="violate the declared schema"):
            completions.assert_response_contract(completion, declared_tools=[weather_tool])


def test_contract_accepts_declared_schema_valid_call(
    completion: completions.Completion, tool_call: dict[str, Any], weather_tool: dict[str, Any]
):
    completions.assert_response_contract(
        dataclasses.replace(completion, finish_reason="tool_calls", tool_calls=[tool_call]),
        declared_tools=[weather_tool],
    )
