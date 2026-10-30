"""Tool context for Harmony XML conformance checks."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

type ToolSpec = Mapping[str, Any] | str


@dataclasses.dataclass(frozen=True, slots=True)
class Tool:
    """One advertised tool and its optional parameter schema."""

    name: str
    builtin: bool = False
    # `None` means unknown schema; an empty mapping means no parameters allowed.
    schema: Mapping[str, Any] | None = None
    required: frozenset[str] = frozenset()

    @property
    def wire_name(self) -> str:
        """Declared name expected after `to=`, including for built-ins."""
        return self.name


@dataclasses.dataclass(frozen=True, slots=True)
class ToolSet:
    """Canonical, deduplicated tool context used by the evaluator."""

    tools: tuple[Tool, ...] = ()

    @classmethod
    def from_specs(
        cls,
        tools: Sequence[ToolSpec] | None = None,
        builtin_tools: Iterable[str] = (),
    ) -> ToolSet:
        """Build a tool set from OpenAI-style specs or names."""
        builtins = frozenset(name for name in builtin_tools if name)
        resolved: dict[str, Tool] = {name: Tool(name=name, builtin=True) for name in sorted(builtins)}
        for spec in tools or ():
            name, schema, required = _name_and_schema(spec)
            if name is None:
                continue
            previous = resolved.get(name)
            if schema is None and previous is not None:
                schema = previous.schema
                required = previous.required | required
            resolved[name] = Tool(
                name=name,
                builtin=name in builtins or (previous.builtin if previous else False),
                schema=schema,
                required=required,
            )
        return cls(tuple(resolved.values()))

    @property
    def builtin_names(self) -> frozenset[str]:
        """Bare names of built-in tools."""
        return frozenset(tool.name for tool in self.tools if tool.builtin)

    @property
    def wire_names(self) -> frozenset[str]:
        """Names valid after `to=`."""
        return frozenset(tool.wire_name for tool in self.tools)

    def is_known(self, wire_name: str) -> bool:
        """Whether `wire_name` names an advertised tool."""
        return wire_name in self.wire_names

    def _tool_for(self, wire_name: str) -> Tool | None:
        for tool in self.tools:
            if tool.wire_name == wire_name:
                return tool
        return None

    def schema_for(self, wire_name: str) -> Mapping[str, Any] | None:
        """Return the declared property schema for a wire tool name."""
        tool = self._tool_for(wire_name)
        return tool.schema if tool is not None else None

    def required_for(self, wire_name: str) -> frozenset[str]:
        """Return advertised required parameter names for a wire tool name."""
        tool = self._tool_for(wire_name)
        return tool.required if tool is not None else frozenset()


def _name_and_schema(
    spec: ToolSpec,
) -> tuple[str | None, Mapping[str, Any] | None, frozenset[str]]:
    if isinstance(spec, str):
        return spec or None, None, frozenset()
    function_raw = spec.get("function", spec)
    if not isinstance(function_raw, Mapping):
        return None, None, frozenset()
    function = cast(Mapping[str, Any], function_raw)
    name_raw = function.get("name")
    if not name_raw:
        return None, None, frozenset()
    name = str(name_raw)
    parameters_raw = function.get("parameters")
    if not isinstance(parameters_raw, Mapping):
        return name, None, frozenset()
    parameters = cast(Mapping[str, Any], parameters_raw)
    required: frozenset[str] = frozenset()
    required_raw = parameters.get("required")
    if isinstance(required_raw, Sequence) and not isinstance(required_raw, (str, bytes)):
        required = frozenset(str(item) for item in cast(Sequence[object], required_raw))
    properties_raw = parameters.get("properties")
    if not isinstance(properties_raw, Mapping):
        return name, None, required
    properties = cast(Mapping[str, Any], properties_raw)
    resolved: dict[str, Any] = {}
    for key, value in properties.items():
        if isinstance(value, Mapping):
            resolved[str(key)] = dict(cast(Mapping[str, Any], value))
        else:
            resolved[str(key)] = {}
    return name, resolved, required
