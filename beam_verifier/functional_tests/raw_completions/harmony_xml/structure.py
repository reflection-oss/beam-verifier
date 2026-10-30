"""Walk one Harmony XML generation and report the first structural violation."""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, cast

from beam_verifier.functional_tests.raw_completions.harmony_xml import probes
from beam_verifier.functional_tests.raw_completions.harmony_xml import (
    tools as tool_defs,
)

_START = "<|start|>"
_END = "<|end|>"
_MESSAGE = "<|message|>"
_CHANNEL = "<|channel|>"
_CONSTRAIN = "<|constrain|>"
_RETURN = "<|return|>"
_CALL = "<|call|>"
# The tokens a complete generation ends on.
TERMINATORS = (_RETURN, _CALL)
_START_ASSISTANT = f"{_START}assistant"
_TOOL_HEADER_PREFIX = "assistant to="
# Tool recipients preserve declared names, including dotted names such as web.search.
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")
# XML parameter names may also begin with a digit.
_PARAMETER_NAME_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
_RFL_TAG_RE = re.compile(
    r'<rfl:parameter name="([^"]*)">\n(.*?)\n(?:</rfl:parameter>|(?=<rfl:parameter name="))',
    re.DOTALL,
)
_JSON_TYPES = frozenset(
    {
        "integer",
        "number",
        "boolean",
        "null",
        "object",
        "array",
    }
)

_SPECIAL_RE = re.compile(r"<\|(?:start|end|message|channel|constrain|return|call)\|>")


def normalize_completion(completion: str) -> str:
    """Strip a leading generation-prompt `<|start|>assistant` (no `to=`)."""
    if completion.startswith(_START_ASSISTANT) and not completion.startswith(f"{_START_ASSISTANT} to="):
        return completion[len(_START_ASSISTANT) :]
    return completion


def _lex(text: str) -> list[tuple[bool, str]]:
    atoms: list[tuple[bool, str]] = []
    position = 0
    for match in _SPECIAL_RE.finditer(text):
        if match.start() > position:
            atoms.append((False, text[position : match.start()]))
        atoms.append((True, match.group()))
        position = match.end()
    if position < len(text):
        atoms.append((False, text[position:]))
    return atoms


def _parse_rfl_body(body: str) -> dict[str, str] | None:
    """Parse `rfl:parameter` elements, or None if the body is illegal."""
    arguments: dict[str, str] = {}
    position = 0
    length = len(body)
    while position < length:
        while position < length and body[position].isspace():
            position += 1
        if position == length:
            break
        tag_match = _RFL_TAG_RE.match(body, position)
        if tag_match is None:
            return None
        name = tag_match.group(1)
        if not _PARAMETER_NAME_RE.fullmatch(name) or name in arguments:
            return None
        arguments[name] = tag_match.group(2)
        position = tag_match.end()
    return arguments


def _schema_types(schema: object) -> tuple[str, ...]:
    if not isinstance(schema, dict):
        return ()
    schema_fields = cast(dict[str, object], schema)
    declared = schema_fields.get("type")
    types = [declared] if isinstance(declared, str) else []
    if isinstance(declared, list):
        types.extend(t for t in cast(list[object], declared) if isinstance(t, str))
    for union_name in ("anyOf", "oneOf"):
        for choice in cast(Iterable[object], schema_fields.get(union_name) or []):
            types.extend(_schema_types(choice))
    if schema_fields.get("nullable") and "null" not in types:
        types.append("null")
    return tuple(types)


def _coerce(raw: str, types: tuple[str, ...]) -> object:
    for type_name in types:
        try:
            if type_name == "integer":
                return int(raw)
            if type_name == "number":
                number = float(raw)
                if number != number or number in (float("inf"), float("-inf")):
                    continue  # NaN / inf are not valid JSON numbers
                # Preserve ints when the source text had no fractional part.
                return int(number) if number.is_integer() and "." not in raw else number
            if type_name == "boolean" and raw.strip().lower() in ("true", "1", "false", "0"):
                return raw.strip().lower() in ("true", "1")
            if type_name == "null" and raw.strip() in ("null", "None"):
                return None
            if type_name in ("object", "array"):
                decoded = json.loads(raw)
                if isinstance(decoded, dict if type_name == "object" else list):
                    return cast(dict[str, object] | list[object], decoded)
        except ValueError:
            continue
    return raw  # `string` params, unknown types and failed casts all keep the original text


def _cast_ok(raw: str, schema: Mapping[str, Any]) -> bool:
    """Whether raw parameter text can represent one of the declared types."""
    types = _schema_types(dict(schema))
    if not types or "string" in types:
        return True
    if not any(name in _JSON_TYPES for name in types):
        return True
    return _coerce(raw, types) is not raw


class _Checker:
    """State machine for one normalized generation."""

    def __init__(self, text: str, tools: tool_defs.ToolSet) -> None:
        self._atoms = _lex(text)
        self._index = 0
        self._tools = tools
        self._shape = probes.CompletionShape.UNKNOWN
        self._tool_calls = 0
        self._builtin_tool = False
        self._schema_tool = False

    def run(self) -> probes.ValidationResult:
        """Validate the required analysis followed by final or grouped tools."""
        result = self._analysis()
        if result is None:
            result = self._after_analysis()
        return result.classified(self._shape, self._facets())

    def _facets(self) -> probes.CompletionFacets:
        return probes.CompletionFacets(
            builtin_tool=self._builtin_tool,
            schema_tool=self._schema_tool,
        )

    def _done(self) -> bool:
        return self._index == len(self._atoms)

    def _special(self, value: str) -> bool:
        if self._done():
            return False
        special, atom = self._atoms[self._index]
        return special and atom == value

    def _consume_special(self, value: str) -> bool:
        if not self._special(value):
            return False
        self._index += 1
        return True

    def _consume_text(self) -> str | None:
        if self._done() or self._atoms[self._index][0]:
            return None
        value = self._atoms[self._index][1]
        self._index += 1
        return value

    def _analysis(self) -> probes.ValidationResult | None:
        if self._done():
            return probes.ValidationResult.incomplete()
        if not self._consume_special(_CHANNEL):
            return probes.ValidationResult.fail(probes.ProbeId.MANDATORY_FIRST_ANALYSIS)
        label = self._consume_text()
        if label is None:
            return (
                probes.ValidationResult.incomplete()
                if self._done()
                else probes.ValidationResult.fail(probes.ProbeId.CHANNEL_PRESENCE_VALIDITY)
            )
        if label == "final":
            return probes.ValidationResult.fail(probes.ProbeId.MANDATORY_FIRST_ANALYSIS)
        if label != "analysis":
            return probes.ValidationResult.fail(probes.ProbeId.CHANNEL_PRESENCE_VALIDITY)
        return self._analysis_body()

    def _analysis_body(self) -> probes.ValidationResult | None:
        if not self._consume_special(_MESSAGE):
            return (
                probes.ValidationResult.incomplete()
                if self._done()
                else probes.ValidationResult.fail(probes.ProbeId.REASONING_MESSAGE_FORM)
            )

        has_text = False
        while not self._done():
            special, value = self._atoms[self._index]
            if special and value == _END:
                if not has_text:
                    return probes.ValidationResult.fail(probes.ProbeId.REASONING_MESSAGE_FORM)
                self._index += 1
                return None
            if special:
                return probes.ValidationResult.fail(probes.ProbeId.REASONING_MESSAGE_FORM)
            has_text = True
            self._index += 1
        return probes.ValidationResult.incomplete()

    def _after_analysis(self) -> probes.ValidationResult:
        while True:
            if self._done():
                return probes.ValidationResult.fail(probes.ProbeId.NO_TRAILING_REASONING)
            if not self._consume_special(_START):
                return probes.ValidationResult.fail(probes.ProbeId.MESSAGE_START_AFTER_END)
            header = self._consume_text()
            if header is None:
                return (
                    probes.ValidationResult.incomplete()
                    if self._done()
                    else probes.ValidationResult.fail(probes.ProbeId.TOOL_CALL_HEADER_FORM)
                )
            if header == "assistant":
                result = self._final()
                if result is not None:
                    return result
            elif header.startswith(_TOOL_HEADER_PREFIX):
                return self._tool_group(header.removeprefix(_TOOL_HEADER_PREFIX))
            else:
                return probes.ValidationResult.fail(probes.ProbeId.TOOL_CALL_HEADER_FORM)

    def _final(self) -> probes.ValidationResult | None:
        if not self._consume_special(_CHANNEL):
            return (
                probes.ValidationResult.incomplete()
                if self._done()
                else probes.ValidationResult.fail(probes.ProbeId.TOOL_CALL_HEADER_FORM)
            )
        label = self._consume_text()
        if label is None:
            return (
                probes.ValidationResult.incomplete()
                if self._done()
                else probes.ValidationResult.fail(probes.ProbeId.CHANNEL_PRESENCE_VALIDITY)
            )
        if label == "analysis":
            return self._analysis_body()
        if label != "final":
            return probes.ValidationResult.fail(probes.ProbeId.CHANNEL_PRESENCE_VALIDITY)
        # FINAL_ONLY vs NO_CALL: whether any tools were advertised.
        self._shape = probes.CompletionShape.NO_CALL if self._tools.tools else probes.CompletionShape.FINAL_ONLY
        if not self._consume_special(_MESSAGE):
            return (
                probes.ValidationResult.incomplete()
                if self._done()
                else probes.ValidationResult.fail(probes.ProbeId.FINAL_TERMINATOR)
            )

        while not self._done():
            special, value = self._atoms[self._index]
            if special and value == _RETURN:
                self._index += 1
                return (
                    probes.ValidationResult.success()
                    if self._done()
                    else probes.ValidationResult.fail(probes.ProbeId.UNCLASSIFIED)
                )
            if special:
                return probes.ValidationResult.fail(probes.ProbeId.FINAL_TERMINATOR)
            self._index += 1
        return probes.ValidationResult.incomplete()

    def _tool_group(self, name: str) -> probes.ValidationResult:
        while True:
            result, terminator = self._tool_call(name)
            if result is not None:
                return result
            if terminator == _CALL:
                if self._done():
                    return probes.ValidationResult.success()
                return self._after_last_tool()
            followup, name = self._next_tool()
            if followup is not None:
                return followup

    def _tool_call(self, name: str) -> tuple[probes.ValidationResult | None, str | None]:
        # Count attempted frames so unknown-name failures still shape as tool calls.
        self._tool_calls += 1
        self._shape = (
            probes.CompletionShape.SINGLE_TOOL if self._tool_calls == 1 else probes.CompletionShape.PARALLEL_TOOL
        )
        if _NAME_RE.fullmatch(name) is None:
            return probes.ValidationResult.fail(probes.ProbeId.TOOL_CALL_HEADER_FORM), None
        if not self._tools.is_known(name):
            return probes.ValidationResult.fail(probes.ProbeId.TOOL_CALL_NAME_EXISTS), None
        self._builtin_tool = self._builtin_tool or (name in self._tools.builtin_names)
        self._schema_tool = self._schema_tool or (self._tools.schema_for(name) is not None)
        if not self._consume_special(_CHANNEL):
            return (
                probes.ValidationResult.incomplete()
                if self._done()
                else probes.ValidationResult.fail(probes.ProbeId.TOOL_CALL_HEADER_FORM)
            ), None
        label = self._consume_text()
        if label is None:
            return (
                probes.ValidationResult.incomplete()
                if self._done()
                else probes.ValidationResult.fail(probes.ProbeId.TOOL_CALL_HEADER_FORM)
            ), None
        if label.strip() != "analysis":
            return probes.ValidationResult.fail(probes.ProbeId.TOOL_CALL_HEADER_FORM), None
        if label == "analysis" and self._done():
            return probes.ValidationResult.incomplete(), None
        if label != "analysis ":
            return probes.ValidationResult.fail(probes.ProbeId.CODEC_CORRECTNESS), None
        if not self._consume_special(_CONSTRAIN):
            return (
                probes.ValidationResult.incomplete()
                if self._done()
                else probes.ValidationResult.fail(probes.ProbeId.CODEC_CORRECTNESS)
            ), None
        codec = self._consume_text()
        if codec is None:
            return (
                probes.ValidationResult.incomplete()
                if self._done()
                else probes.ValidationResult.fail(probes.ProbeId.CODEC_CORRECTNESS)
            ), None
        if codec != "xml":
            return probes.ValidationResult.fail(probes.ProbeId.CODEC_CORRECTNESS), None
        if not self._consume_special(_MESSAGE):
            return (
                probes.ValidationResult.incomplete()
                if self._done()
                else probes.ValidationResult.fail(probes.ProbeId.CODEC_CORRECTNESS)
            ), None

        body_parts: list[str] = []
        while not self._done():
            special, value = self._atoms[self._index]
            if special and value in (_END, _CALL):
                self._index += 1
                body = "".join(body_parts)
                failure = self._validate_tool_body(name, body)
                return (probes.ValidationResult.fail(failure), None) if failure is not None else (None, value)
            if special:
                return probes.ValidationResult.fail(probes.ProbeId.TOOL_CALL_XML_WELL_FORMED), None
            body_parts.append(value)
            self._index += 1

        body = "".join(body_parts)
        if _parse_rfl_body(body) is None:
            return probes.ValidationResult.fail(probes.ProbeId.TOOL_CALL_XML_WELL_FORMED), None
        return probes.ValidationResult.incomplete(), None

    def _validate_tool_body(self, name: str, body: str) -> probes.ProbeId | None:
        arguments = _parse_rfl_body(body)
        if arguments is None:
            return probes.ProbeId.TOOL_CALL_XML_WELL_FORMED
        schema = self._tools.schema_for(name)
        if schema is None:
            return (
                probes.ProbeId.TOOL_CALL_PARAMETER_REQUIRED
                if self._tools.required_for(name) - arguments.keys()
                else None
            )
        for parameter, value in arguments.items():
            declaration = schema.get(parameter)
            if declaration is None:
                return probes.ProbeId.TOOL_CALL_PARAMETER_EXISTS
            if isinstance(declaration, Mapping) and not _cast_ok(value, cast(Mapping[str, Any], declaration)):
                return probes.ProbeId.TOOL_CALL_PARAMETER_TYPES
        if self._tools.required_for(name) - arguments.keys():
            return probes.ProbeId.TOOL_CALL_PARAMETER_REQUIRED
        return None

    def _next_tool(self) -> tuple[probes.ValidationResult | None, str]:
        if self._done():
            return probes.ValidationResult.incomplete(), ""
        if not self._consume_special(_START):
            return probes.ValidationResult.fail(probes.ProbeId.MESSAGE_START_AFTER_END), ""
        header = self._consume_text()
        if header is None:
            return (
                probes.ValidationResult.incomplete()
                if self._done()
                else probes.ValidationResult.fail(probes.ProbeId.TOOL_CALL_HEADER_FORM),
                "",
            )
        if header == "assistant":
            return probes.ValidationResult.fail(probes.ProbeId.GROUPING_PLACEMENT), ""
        if not header.startswith(_TOOL_HEADER_PREFIX):
            return probes.ValidationResult.fail(probes.ProbeId.TOOL_CALL_HEADER_FORM), ""
        return None, header.removeprefix(_TOOL_HEADER_PREFIX)

    def _after_last_tool(self) -> probes.ValidationResult:
        if not self._consume_special(_START):
            return probes.ValidationResult.fail(probes.ProbeId.GROUPING_PLACEMENT)
        header = self._consume_text()
        if header is not None and header.startswith(_TOOL_HEADER_PREFIX):
            return probes.ValidationResult.fail(probes.ProbeId.PARALLEL_TERMINATION)
        return probes.ValidationResult.fail(probes.ProbeId.GROUPING_PLACEMENT)


def validate_completion(
    completion: str,
    *,
    tools: Sequence[tool_defs.ToolSpec] | None = None,
    builtin_tools: Iterable[str] = (),
    finish_reason: Literal["stop", "length"] = "stop",
) -> probes.ValidationResult:
    """Check one generated completion with its advertised tools.

    Args:
        completion: Raw generated text, with Harmony control tokens preserved.
        tools: Explicit tool names or OpenAI-style function specifications.
        builtin_tools: Advertised tool names that represent built-ins.
        finish_reason: A token-limit finish marks the result as incomplete,
            preserving any structural failure already found.

    Returns:
        Validity, first failure, structural shape, and truncation status.
    """
    result = _Checker(
        normalize_completion(completion),
        tool_defs.ToolSet.from_specs(tools, builtin_tools),
    ).run()
    if finish_reason == "length":
        return dataclasses.replace(result, ok=False, truncated=True)
    return result
