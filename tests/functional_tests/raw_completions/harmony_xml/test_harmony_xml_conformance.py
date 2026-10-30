"""Probe attribution, valid generations, and truncation for the portable checker."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest
import yaml

from beam_verifier.functional_tests.raw_completions.harmony_xml import probes, structure

_EXAMPLES = yaml.safe_load((pathlib.Path(__file__).parent / "fixtures/examples.yaml").read_text())

with (pathlib.Path(__file__).parent / "fixtures/harmony_xml_edge_cases.json").open() as _file:
    _EDGE_CASES: list[dict[str, Any]] = json.load(_file)

_ANALYSIS = _EXAMPLES["analysis"]


_FINAL = _EXAMPLES["final"]


_WEATHER_TOOL = _EXAMPLES["weather_tool"]


_NOARGS_TOOL = _EXAMPLES["noargs_tool"]


_WEATHER_REQUIRED_TOOL = _EXAMPLES["weather_required_tool"]


def _tool_call(
    name: str = "weather",
    *,
    body: str = "",
    end: str = "<|call|>",
    channel: str = "analysis ",
    codec: str = "xml",
) -> str:
    return f"<|start|>assistant to={name}<|channel|>{channel}<|constrain|>{codec}<|message|>{body}{end}"


@pytest.mark.parametrize(
    "completion",
    [
        _ANALYSIS + _FINAL,
        "<|start|>assistant" + _ANALYSIS + _FINAL,
        _ANALYSIS + "<|start|>assistant" + _ANALYSIS + _FINAL,
    ],
)
def test_legal_final(completion: str) -> None:
    assert structure.validate_completion(completion, tools=[]).ok


def test_legal_tool_call() -> None:
    parameter = '<rfl:parameter name="city">\nLondon\n</rfl:parameter>'
    completion = _ANALYSIS + _tool_call(body=parameter)
    assert structure.validate_completion(completion, tools=["weather"]).ok


def test_legal_parallel_and_builtin_calls() -> None:
    parallel = (
        _ANALYSIS + _tool_call(end="<|end|>") + _tool_call(body='<rfl:parameter name="city">\nParis\n</rfl:parameter>')
    )
    assert structure.validate_completion(parallel, tools=["weather"]).ok

    builtin = _ANALYSIS + _tool_call(name="bash")
    assert structure.validate_completion(
        builtin,
        tools=["bash"],
        builtin_tools=("bash",),
    ).ok


def test_typed_tool_parameters() -> None:
    body = '<rfl:parameter name="city">\nLondon\n</rfl:parameter>\n<rfl:parameter name="days">\n3\n</rfl:parameter>'
    assert structure.validate_completion(
        _ANALYSIS + _tool_call(body=body),
        tools=[_WEATHER_TOOL],
    ).ok


@pytest.mark.parametrize(
    "completion",
    _EXAMPLES["truncation_is_a_status"],
)
def test_truncation_is_a_status(completion: str) -> None:
    result = structure.validate_completion(completion, tools=["weather"])
    assert not result.ok
    assert result.truncated
    assert result.failure is None


@pytest.mark.parametrize(
    ("completion", "tools", "builtins", "probe"),
    _EXAMPLES["probe_attribution"],
)
def test_probe_attribution(
    completion: str,
    tools: list[str] | list[dict[str, Any]],
    builtins: tuple[str, ...],
    probe: str,
) -> None:
    result = structure.validate_completion(
        completion,
        tools=tools,
        builtin_tools=builtins,
    )
    assert not result.ok and result.failure is not None
    assert result.failure.probe_id == probes.ProbeId(probe)


@pytest.mark.parametrize(
    ("suffix", "probe"),
    [
        ("assistant<|channel|><|message|>answer<|return|>", probes.ProbeId.CHANNEL_PRESENCE_VALIDITY),
        ("assistant to=weather<|channel|><|constrain|>xml<|message|><|call|>", probes.ProbeId.TOOL_CALL_HEADER_FORM),
        ("assistant to=weather<|channel|>analysis <|constrain|><|message|><|call|>", probes.ProbeId.CODEC_CORRECTNESS),
    ],
)
def test_missing_header_fields_are_invalid_not_truncated(suffix: str, probe: str) -> None:
    result = structure.validate_completion(_ANALYSIS + "<|start|>" + suffix, tools=["weather"])

    assert not result.ok
    assert not result.truncated
    assert result.failure is not None
    assert result.failure.probe_id == probes.ProbeId(probe)


@pytest.mark.parametrize("token", ["<|message|>", "<|constrain|>"])
@pytest.mark.parametrize(
    ("prefix", "suffix", "probe"),
    [
        ("<|channel|>analysis<|message|>", "<|end|>" + _FINAL, probes.ProbeId.REASONING_MESSAGE_FORM),
        (_ANALYSIS + "<|start|>assistant<|channel|>final<|message|>", "<|return|>", probes.ProbeId.FINAL_TERMINATOR),
    ],
)
def test_reserved_tokens_in_message_bodies_fail(token: str, prefix: str, suffix: str, probe: str) -> None:
    result = structure.validate_completion(prefix + "before" + token + "after" + suffix)

    assert not result.ok
    assert result.failure is not None
    assert result.failure.probe_id == probes.ProbeId(probe)


@pytest.mark.parametrize(
    "completion",
    _EXAMPLES["eof_after_valid_header_fields_is_truncated"],
)
def test_eof_after_valid_header_fields_is_truncated(completion: str) -> None:
    result = structure.validate_completion(completion, tools=["weather"])

    assert not result.ok
    assert result.truncated
    assert result.failure is None


@pytest.mark.parametrize(("value", "valid"), [("3.5", True), ("later", False)])
def test_numeric_oneof_parameters(value: str, valid: bool) -> None:
    tool = {
        "name": "weather",
        "parameters": {"properties": {"cursor": {"oneOf": [{"type": "number"}, {"type": "integer"}]}}},
    }
    result = structure.validate_completion(
        _ANALYSIS + _tool_call(body=f'<rfl:parameter name="cursor">\n{value}\n</rfl:parameter>'), tools=[tool]
    )

    assert result.ok == valid
    assert (result.failure.probe_id if result.failure else None) == (
        None if valid else probes.ProbeId.TOOL_CALL_PARAMETER_TYPES
    )


@pytest.mark.parametrize("parameters", [{"required": ["city"]}, {"properties": {}, "required": ["city"]}])
def test_required_parameter_is_enforced_without_property_declaration(parameters: dict[str, object]) -> None:
    result = structure.validate_completion(
        _ANALYSIS + _tool_call(), tools=[{"name": "weather", "parameters": parameters}]
    )

    assert not result.ok
    assert result.failure is not None
    assert result.failure.probe_id == probes.ProbeId.TOOL_CALL_PARAMETER_REQUIRED


def test_parameter_value_ends_at_next_opener_without_a_closer() -> None:
    body = '<rfl:parameter name="days">\n3\n<rfl:parameter name="city">\nLondon\n</rfl:parameter>'
    assert structure.validate_completion(_ANALYSIS + _tool_call(body=body), tools=[_WEATHER_REQUIRED_TOOL]).ok


def test_opener_without_a_preceding_newline_stays_in_the_value() -> None:
    body = '<rfl:parameter name="city">\nLondon<rfl:parameter name="days">\n3\n</rfl:parameter>'
    assert structure.validate_completion(_ANALYSIS + _tool_call(body=body), tools=[_WEATHER_REQUIRED_TOOL]).ok


@pytest.mark.parametrize("name", ["bash", "web.search"])
def test_builtin_calls_preserve_declared_names_and_facets(name: str) -> None:
    tool = {"name": name, "parameters": {"properties": {"query": {"type": "string"}}, "required": ["query"]}}
    result = structure.validate_completion(
        _ANALYSIS + _tool_call(name=name, body='<rfl:parameter name="query">\nexample\n</rfl:parameter>'),
        tools=[tool],
        builtin_tools=(name,),
    )

    assert result.ok
    assert result.shape == probes.CompletionShape.SINGLE_TOOL
    assert result.facets.builtin_tool
    assert result.facets.schema_tool


@pytest.mark.parametrize(
    ("completion", "tools", "shape"),
    _EXAMPLES["shape_classification"],
)
def test_shape_classification(completion: str, tools: list[str], shape: str) -> None:
    result = structure.validate_completion(completion, tools=tools)
    assert result.ok
    assert result.shape == probes.CompletionShape(shape)


def test_shape_is_reported_on_failure() -> None:
    completion = _ANALYSIS + _tool_call(end="<|end|>") + _tool_call(end="<|end|>")
    result = structure.validate_completion(completion, tools=["weather"])

    assert not result.ok
    assert result.shape == probes.CompletionShape.PARALLEL_TOOL


@pytest.mark.parametrize(
    ("completion", "shape"),
    _EXAMPLES["shape_counts_attempted_frames_not_valid_ones"],
)
def test_shape_counts_attempted_frames_not_valid_ones(completion: str, shape: str) -> None:
    """An undeclared `to=NAME` still shapes as a tool call, never as unknown."""
    result = structure.validate_completion(completion, tools=["weather"])

    assert result.failure is not None
    assert result.failure.probe_id == probes.ProbeId.TOOL_CALL_NAME_EXISTS
    assert result.shape == probes.CompletionShape(shape)


def test_facets_report_builtin_and_schema_tools() -> None:
    builtin = structure.validate_completion(
        _ANALYSIS + _tool_call(name="bash"),
        tools=["bash"],
        builtin_tools=("bash",),
    )
    assert builtin.facets.builtin_tool
    assert not builtin.facets.schema_tool

    schema_bearing = structure.validate_completion(
        _ANALYSIS + _tool_call(body='<rfl:parameter name="city">\nLondon\n</rfl:parameter>'),
        tools=[_WEATHER_TOOL],
    )
    assert schema_bearing.facets.schema_tool
    assert not schema_bearing.facets.builtin_tool


def test_required_parameter_may_omit_optional_ones() -> None:
    body = '<rfl:parameter name="city">\nLondon\n</rfl:parameter>'
    assert structure.validate_completion(
        _ANALYSIS + _tool_call(body=body),
        tools=[_WEATHER_REQUIRED_TOOL],
    ).ok


def test_parameter_values_may_contain_lt() -> None:
    """Well-formedness must not reject raw '<' inside parameter text."""
    assert structure.validate_completion(
        _ANALYSIS + _tool_call(body='<rfl:parameter name="city">\nx < y\n</rfl:parameter>'),
        tools=["weather"],
    ).ok


def test_parameter_elements_may_be_separated_by_whitespace() -> None:
    """Whitespace between `rfl:parameter` elements is still a legal body."""
    body = '<rfl:parameter name="city">\nLondon\n</rfl:parameter> \n\t <rfl:parameter name="days">\n3\n</rfl:parameter>'
    assert structure.validate_completion(
        _ANALYSIS + _tool_call(body=body),
        tools=[_WEATHER_TOOL],
    ).ok


def test_whitespace_only_xml_body_is_empty_arguments() -> None:
    assert structure.validate_completion(
        _ANALYSIS + _tool_call(name="noargs", body="  \n  "),
        tools=[_NOARGS_TOOL],
    ).ok


@pytest.mark.parametrize("case", _EDGE_CASES, ids=[case["id"] for case in _EDGE_CASES])
def test_additional_conformance_cases(case: dict[str, Any]) -> None:
    result = structure.validate_completion(case["completion"], tools=case["tools"], builtin_tools=case["builtins"])
    assert result.ok == case["ok"]
    assert result.truncated == case["truncated"]
    assert (result.failure.probe_id if result.failure else None) == case["probe"]


def test_free_text_to_is_not_a_tool_call() -> None:
    completion = (
        "<|channel|>analysis<|message|>see to=unknown<|end|>"
        "<|start|>assistant"
        "<|channel|>final<|message|>to=unknown<|return|>"
    )
    assert structure.validate_completion(completion, tools=["weather"]).ok


def test_empty_schema_rejects_parameters() -> None:
    completion = _ANALYSIS + _tool_call(
        name="noargs",
        body='<rfl:parameter name="extra">\n1\n</rfl:parameter>',
    )
    result = structure.validate_completion(completion, tools=[_NOARGS_TOOL])
    assert result.failure is not None
    assert result.failure.probe_id == probes.ProbeId.TOOL_CALL_PARAMETER_EXISTS


def test_parallel_calls_have_no_artificial_limit() -> None:
    completion = _ANALYSIS + _tool_call(end="<|end|>") * 20 + _tool_call()
    assert structure.validate_completion(completion, tools=["weather"]).ok


def test_consecutive_analysis_messages_have_no_artificial_limit() -> None:
    completion = _ANALYSIS + ("<|start|>assistant" + _ANALYSIS) * 2000 + _FINAL
    assert structure.validate_completion(completion, tools=[]).ok
