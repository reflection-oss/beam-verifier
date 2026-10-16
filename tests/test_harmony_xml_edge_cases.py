"""Additional shape, schema, and framing edge cases for raw completions."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from beam_verifier.functional_tests.raw_completions.harmony_xml import probes, structure

with (pathlib.Path(__file__).parent / "fixtures/harmony_xml_edge_cases.json").open() as _file:
    _EDGE_CASES: list[dict[str, Any]] = json.load(_file)


_ANALYSIS = "<|channel|>analysis<|message|>think<|end|>"
_FINAL = "<|start|>assistant<|channel|>final<|message|>answer<|return|>"
_WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
            },
        },
    },
}
_NOARGS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "noargs",
        "parameters": {"type": "object", "properties": dict[str, Any]()},
    },
}
_WEATHER_REQUIRED_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
            },
            "required": ["city"],
        },
    },
}


def _tool_call(
    name: str = "weather",
    *,
    body: str = "",
    end: str = "<|call|>",
    channel: str = "analysis ",
    codec: str = "xml",
) -> str:
    return f"<|start|>assistant to={name}<|channel|>{channel}<|constrain|>{codec}<|message|>{body}{end}"


def _eval(
    text: str,
    *,
    tools: list[str] | list[dict[str, Any]] | None = None,
    builtin_tools: tuple[str, ...] = (),
) -> probes.ValidationResult:
    return structure.validate_completion(
        text,
        tools=tools,
        builtin_tools=builtin_tools,
    )


@pytest.mark.parametrize(
    ("completion", "tools", "shape"),
    [
        (_ANALYSIS + _FINAL, [], probes.CompletionShape.FINAL_ONLY),
        (_ANALYSIS + _FINAL, ["weather"], probes.CompletionShape.NO_CALL),
        (
            _ANALYSIS + _tool_call(),
            ["weather"],
            probes.CompletionShape.SINGLE_TOOL,
        ),
        (
            _ANALYSIS + _tool_call(end="<|end|>") + _tool_call(),
            ["weather"],
            probes.CompletionShape.PARALLEL_TOOL,
        ),
    ],
)
def test_shape_classification(completion: str, tools: list[str], shape: probes.CompletionShape) -> None:
    result = _eval(completion, tools=tools)
    assert result.ok
    assert result.shape == shape


def test_shape_is_reported_on_failure() -> None:
    completion = _ANALYSIS + _tool_call(end="<|end|>") + _tool_call(end="<|end|>")
    result = _eval(completion, tools=["weather"])

    assert not result.ok
    assert result.shape == probes.CompletionShape.PARALLEL_TOOL


@pytest.mark.parametrize(
    ("completion", "shape"),
    [
        (
            _ANALYSIS + _tool_call(name="unknown"),
            probes.CompletionShape.SINGLE_TOOL,
        ),
        (
            _ANALYSIS + _tool_call(end="<|end|>") + _tool_call(name="unknown"),
            probes.CompletionShape.PARALLEL_TOOL,
        ),
    ],
)
def test_shape_counts_attempted_frames_not_valid_ones(completion: str, shape: probes.CompletionShape) -> None:
    """An undeclared `to=NAME` still shapes as a tool call, never as unknown."""
    result = _eval(completion, tools=["weather"])

    assert result.failure is not None
    assert result.failure.probe_id == probes.ProbeId.TOOL_CALL_NAME_EXISTS
    assert result.shape == shape


def test_facets_report_builtin_and_schema_tools() -> None:
    builtin = _eval(
        _ANALYSIS + _tool_call(name="bash"),
        tools=["bash"],
        builtin_tools=("bash",),
    )
    assert builtin.facets.builtin_tool
    assert not builtin.facets.schema_tool

    schema_bearing = _eval(
        _ANALYSIS + _tool_call(body='<rfl:parameter name="city">\nLondon\n</rfl:parameter>'),
        tools=[_WEATHER_TOOL],
    )
    assert schema_bearing.facets.schema_tool
    assert not schema_bearing.facets.builtin_tool


def test_required_parameter_may_omit_optional_ones() -> None:
    body = '<rfl:parameter name="city">\nLondon\n</rfl:parameter>'
    assert _eval(
        _ANALYSIS + _tool_call(body=body),
        tools=[_WEATHER_REQUIRED_TOOL],
    ).ok


def test_parameter_values_may_contain_lt() -> None:
    """Well-formedness must not reject raw '<' inside parameter text."""
    assert _eval(
        _ANALYSIS + _tool_call(body='<rfl:parameter name="city">\nx < y\n</rfl:parameter>'),
        tools=["weather"],
    ).ok


def test_parameter_elements_may_be_separated_by_whitespace() -> None:
    """Whitespace between `rfl:parameter` elements is still a legal body."""
    body = '<rfl:parameter name="city">\nLondon\n</rfl:parameter> \n\t <rfl:parameter name="days">\n3\n</rfl:parameter>'
    assert _eval(
        _ANALYSIS + _tool_call(body=body),
        tools=[_WEATHER_TOOL],
    ).ok


def test_whitespace_only_xml_body_is_empty_arguments() -> None:
    assert _eval(
        _ANALYSIS + _tool_call(name="noargs", body="  \n  "),
        tools=[_NOARGS_TOOL],
    ).ok


@pytest.mark.parametrize("case", _EDGE_CASES, ids=[case["id"] for case in _EDGE_CASES])
def test_additional_conformance_cases(case: dict[str, Any]) -> None:
    result = _eval(case["completion"], tools=case["tools"], builtin_tools=case["builtins"])
    assert result.ok == case["ok"]
    assert result.truncated == case["truncated"]
    assert (result.failure.probe_id if result.failure else None) == case["probe"]


def test_free_text_to_is_not_a_tool_call() -> None:
    completion = (
        "<|channel|>analysis<|message|>see to=unknown<|end|>"
        "<|start|>assistant"
        "<|channel|>final<|message|>to=unknown<|return|>"
    )
    assert _eval(completion, tools=["weather"]).ok


def test_empty_schema_rejects_parameters() -> None:
    completion = _ANALYSIS + _tool_call(
        name="noargs",
        body='<rfl:parameter name="extra">\n1\n</rfl:parameter>',
    )
    result = _eval(completion, tools=[_NOARGS_TOOL])
    assert result.failure is not None
    assert result.failure.probe_id == probes.ProbeId.TOOL_CALL_PARAMETER_EXISTS


def test_parallel_calls_have_no_artificial_limit() -> None:
    completion = _ANALYSIS + _tool_call(end="<|end|>") * 20 + _tool_call()
    assert _eval(completion, tools=["weather"]).ok


def test_consecutive_analysis_messages_have_no_artificial_limit() -> None:
    completion = _ANALYSIS + ("<|start|>assistant" + _ANALYSIS) * 2000 + _FINAL
    assert _eval(completion, tools=[]).ok
