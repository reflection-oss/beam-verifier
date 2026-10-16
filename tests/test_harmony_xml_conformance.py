"""Probe attribution, valid generations, and truncation for the portable checker."""

from __future__ import annotations

from typing import Any

import pytest

from beam_verifier.functional_tests.raw_completions.harmony_xml import probes, structure

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
    "completion",
    [
        _ANALYSIS + _FINAL,
        "<|start|>assistant" + _ANALYSIS + _FINAL,
        _ANALYSIS + "<|start|>assistant" + _ANALYSIS + _FINAL,
    ],
)
def test_legal_final(completion: str) -> None:
    assert _eval(completion, tools=[]).ok


def test_legal_tool_call() -> None:
    parameter = '<rfl:parameter name="city">\nLondon\n</rfl:parameter>'
    completion = _ANALYSIS + _tool_call(body=parameter)
    assert structure.validate_completion(completion, tools=["weather"]).ok


def test_legal_parallel_and_builtin_calls() -> None:
    parallel = (
        _ANALYSIS + _tool_call(end="<|end|>") + _tool_call(body='<rfl:parameter name="city">\nParis\n</rfl:parameter>')
    )
    assert _eval(parallel, tools=["weather"]).ok

    builtin = _ANALYSIS + _tool_call(name="bash")
    assert _eval(
        builtin,
        tools=["bash"],
        builtin_tools=("bash",),
    ).ok


def test_typed_tool_parameters() -> None:
    body = '<rfl:parameter name="city">\nLondon\n</rfl:parameter>\n<rfl:parameter name="days">\n3\n</rfl:parameter>'
    assert _eval(
        _ANALYSIS + _tool_call(body=body),
        tools=[_WEATHER_TOOL],
    ).ok


@pytest.mark.parametrize(
    "completion",
    [
        "",
        "<|channel|>",
        "<|channel|>analysis<|message|>partial",
        _ANALYSIS + "<|start|>",
        _ANALYSIS + "<|start|>assistant<|channel|>final<|message|>partial",
        _ANALYSIS + _tool_call(end=""),
    ],
)
def test_truncation_is_a_status(completion: str) -> None:
    result = structure.validate_completion(completion, tools=["weather"])
    assert not result.ok
    assert result.truncated
    assert result.failure is None


@pytest.mark.parametrize(
    ("completion", "tools", "builtins", "probe"),
    [
        (
            "<|channel|>commentary<|message|>x<|end|>",
            [],
            (),
            probes.ProbeId.CHANNEL_PRESENCE_VALIDITY,
        ),
        (
            "<|channel|>final<|message|>x<|return|>",
            [],
            (),
            probes.ProbeId.MANDATORY_FIRST_ANALYSIS,
        ),
        (
            _ANALYSIS,
            [],
            (),
            probes.ProbeId.NO_TRAILING_REASONING,
        ),
        (
            _ANALYSIS + "<|channel|>final<|message|>x<|return|>",
            [],
            (),
            probes.ProbeId.MESSAGE_START_AFTER_END,
        ),
        (
            "<|channel|>analysis<|message|>x<|return|>",
            [],
            (),
            probes.ProbeId.REASONING_MESSAGE_FORM,
        ),
        (
            "<|channel|>analysis<|message|><|end|>" + _FINAL,
            [],
            (),
            probes.ProbeId.REASONING_MESSAGE_FORM,
        ),
        (
            _ANALYSIS + "<|start|>assistant<|channel|>final<|message|>x<|end|>",
            [],
            (),
            probes.ProbeId.FINAL_TERMINATOR,
        ),
        (
            _ANALYSIS + "<|start|>assistant to=weather<|message|>x<|call|>",
            ["weather"],
            (),
            probes.ProbeId.TOOL_CALL_HEADER_FORM,
        ),
        (
            _ANALYSIS + _tool_call(name="unknown"),
            ["weather"],
            (),
            probes.ProbeId.TOOL_CALL_NAME_EXISTS,
        ),
        (
            _ANALYSIS + _tool_call(name="refl.bash"),
            ["bash"],
            ("bash",),
            probes.ProbeId.TOOL_CALL_NAME_EXISTS,
        ),
        (
            _ANALYSIS + _tool_call(codec="json"),
            ["weather"],
            (),
            probes.ProbeId.CODEC_CORRECTNESS,
        ),
        (
            _ANALYSIS + _tool_call(channel="analysis"),
            ["weather"],
            (),
            probes.ProbeId.CODEC_CORRECTNESS,
        ),
        (
            "<|channel|>analysis <|message|>x<|end|>" + _FINAL,
            [],
            (),
            probes.ProbeId.CHANNEL_PRESENCE_VALIDITY,
        ),
        (
            _ANALYSIS + _tool_call(name="1weather"),
            ["weather"],
            (),
            probes.ProbeId.TOOL_CALL_HEADER_FORM,
        ),
        (
            _ANALYSIS
            + _tool_call(
                body=(
                    '<rfl:parameter name="city">\na\n</rfl:parameter>\n<rfl:parameter name="city">\nb\n</rfl:parameter>'
                ),
            ),
            ["weather"],
            (),
            probes.ProbeId.TOOL_CALL_XML_WELL_FORMED,
        ),
        (
            _ANALYSIS
            + _tool_call(
                body='<rfl:parameter name="ci ty">\nx\n</rfl:parameter>',
            ),
            ["weather"],
            (),
            probes.ProbeId.TOOL_CALL_XML_WELL_FORMED,
        ),
        (
            _ANALYSIS + _tool_call(body='<rfl:parameter name="city">London</rfl:parameter>'),
            ["weather"],
            (),
            probes.ProbeId.TOOL_CALL_XML_WELL_FORMED,
        ),
        (
            _ANALYSIS
            + _tool_call(
                body='<parameter name="city">London</parameter>',
            ),
            ["weather"],
            (),
            probes.ProbeId.TOOL_CALL_XML_WELL_FORMED,
        ),
        (
            _ANALYSIS
            + _tool_call(
                body='<rfl:parameter name="city">London',
            ),
            ["weather"],
            (),
            probes.ProbeId.TOOL_CALL_XML_WELL_FORMED,
        ),
        (
            _ANALYSIS
            + _tool_call(
                body='<rfl:parameter name="days">\n3\n</rfl:parameter>',
            ),
            [_WEATHER_REQUIRED_TOOL],
            (),
            probes.ProbeId.TOOL_CALL_PARAMETER_REQUIRED,
        ),
        (
            _ANALYSIS
            + _tool_call(
                body='<rfl:parameter name="units">\nC\n</rfl:parameter>',
            ),
            [_WEATHER_TOOL],
            (),
            probes.ProbeId.TOOL_CALL_PARAMETER_EXISTS,
        ),
        (
            _ANALYSIS
            + _tool_call(
                body='<rfl:parameter name="days">\nsoon\n</rfl:parameter>',
            ),
            [_WEATHER_TOOL],
            (),
            probes.ProbeId.TOOL_CALL_PARAMETER_TYPES,
        ),
        (
            _ANALYSIS + _tool_call(end="<|end|>") + "<|start|>assistant<|channel|>final<|message|>x<|return|>",
            ["weather"],
            (),
            probes.ProbeId.GROUPING_PLACEMENT,
        ),
        (
            _ANALYSIS + _tool_call() + _FINAL,
            ["weather"],
            (),
            probes.ProbeId.GROUPING_PLACEMENT,
        ),
        (
            _ANALYSIS + _tool_call() + _tool_call(),
            ["weather"],
            (),
            probes.ProbeId.PARALLEL_TERMINATION,
        ),
        (
            _ANALYSIS + _FINAL + "trailing",
            [],
            (),
            probes.ProbeId.UNCLASSIFIED,
        ),
    ],
)
def test_probe_attribution(
    completion: str,
    tools: list[str] | list[dict[str, Any]],
    builtins: tuple[str, ...],
    probe: probes.ProbeId,
) -> None:
    result = _eval(
        completion,
        tools=tools,
        builtin_tools=builtins,
    )
    assert not result.ok and result.failure is not None
    assert result.failure.probe_id == probe


@pytest.mark.parametrize(
    ("suffix", "probe"),
    [
        ("assistant<|channel|><|message|>answer<|return|>", probes.ProbeId.CHANNEL_PRESENCE_VALIDITY),
        ("assistant to=weather<|channel|><|constrain|>xml<|message|><|call|>", probes.ProbeId.TOOL_CALL_HEADER_FORM),
        ("assistant to=weather<|channel|>analysis <|constrain|><|message|><|call|>", probes.ProbeId.CODEC_CORRECTNESS),
    ],
)
def test_missing_header_fields_are_invalid_not_truncated(suffix: str, probe: probes.ProbeId) -> None:
    result = _eval(_ANALYSIS + "<|start|>" + suffix, tools=["weather"])

    assert not result.ok
    assert not result.truncated
    assert result.failure is not None
    assert result.failure.probe_id == probe


@pytest.mark.parametrize("token", ["<|message|>", "<|constrain|>"])
@pytest.mark.parametrize(
    ("prefix", "suffix", "probe"),
    [
        ("<|channel|>analysis<|message|>", "<|end|>" + _FINAL, probes.ProbeId.REASONING_MESSAGE_FORM),
        (_ANALYSIS + "<|start|>assistant<|channel|>final<|message|>", "<|return|>", probes.ProbeId.FINAL_TERMINATOR),
    ],
)
def test_reserved_tokens_in_message_bodies_fail(token: str, prefix: str, suffix: str, probe: probes.ProbeId) -> None:
    result = _eval(prefix + "before" + token + "after" + suffix)

    assert not result.ok
    assert result.failure is not None
    assert result.failure.probe_id == probe


@pytest.mark.parametrize(
    "completion",
    [
        "<|channel|>analysis",
        _ANALYSIS + "<|start|>assistant<|channel|>final",
        _ANALYSIS + "<|start|>assistant to=weather",
        _ANALYSIS + "<|start|>assistant to=weather<|channel|>",
        _ANALYSIS + "<|start|>assistant to=weather<|channel|>analysis",
        _ANALYSIS + "<|start|>assistant to=weather<|channel|>analysis ",
        _ANALYSIS + "<|start|>assistant to=weather<|channel|>analysis <|constrain|>",
        _ANALYSIS + "<|start|>assistant to=weather<|channel|>analysis <|constrain|>xml",
    ],
)
def test_eof_after_valid_header_fields_is_truncated(completion: str) -> None:
    result = _eval(completion, tools=["weather"])

    assert not result.ok
    assert result.truncated
    assert result.failure is None


@pytest.mark.parametrize(("value", "valid"), [("3.5", True), ("later", False)])
def test_numeric_oneof_parameters(value: str, valid: bool) -> None:
    tool = {
        "name": "weather",
        "parameters": {"properties": {"cursor": {"oneOf": [{"type": "number"}, {"type": "integer"}]}}},
    }
    result = _eval(
        _ANALYSIS + _tool_call(body=f'<rfl:parameter name="cursor">\n{value}\n</rfl:parameter>'), tools=[tool]
    )

    assert result.ok == valid
    assert (result.failure.probe_id if result.failure else None) == (
        None if valid else probes.ProbeId.TOOL_CALL_PARAMETER_TYPES
    )


@pytest.mark.parametrize("parameters", [{"required": ["city"]}, {"properties": {}, "required": ["city"]}])
def test_required_parameter_is_enforced_without_property_declaration(parameters: dict[str, object]) -> None:
    result = _eval(_ANALYSIS + _tool_call(), tools=[{"name": "weather", "parameters": parameters}])

    assert not result.ok
    assert result.failure is not None
    assert result.failure.probe_id == probes.ProbeId.TOOL_CALL_PARAMETER_REQUIRED


def test_parameter_value_ends_at_next_opener_without_a_closer() -> None:
    body = '<rfl:parameter name="days">\n3\n<rfl:parameter name="city">\nLondon\n</rfl:parameter>'
    assert _eval(_ANALYSIS + _tool_call(body=body), tools=[_WEATHER_REQUIRED_TOOL]).ok


def test_opener_without_a_preceding_newline_stays_in_the_value() -> None:
    body = '<rfl:parameter name="city">\nLondon<rfl:parameter name="days">\n3\n</rfl:parameter>'
    assert _eval(_ANALYSIS + _tool_call(body=body), tools=[_WEATHER_REQUIRED_TOOL]).ok


@pytest.mark.parametrize("name", ["bash", "web.search"])
def test_builtin_calls_preserve_declared_names_and_facets(name: str) -> None:
    tool = {"name": name, "parameters": {"properties": {"query": {"type": "string"}}, "required": ["query"]}}
    result = _eval(
        _ANALYSIS + _tool_call(name=name, body='<rfl:parameter name="query">\nexample\n</rfl:parameter>'),
        tools=[tool],
        builtin_tools=(name,),
    )

    assert result.ok
    assert result.shape == probes.CompletionShape.SINGLE_TOOL
    assert result.facets.builtin_tool
    assert result.facets.schema_tool
