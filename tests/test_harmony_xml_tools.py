"""Explicit tool declarations used to check generated calls."""

import pytest

from beam_verifier.functional_tests.raw_completions.harmony_xml import tools


def test_function_schema_preserves_required_and_optional_parameters() -> None:
    declared = tools.ToolSet.from_specs(
        [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                        "required": ["city"],
                    },
                },
            }
        ]
    )

    assert declared.is_known("weather")
    assert not declared.is_known("unknown")
    assert declared.schema_for("weather") == {"city": {"type": "string"}, "days": {"type": "integer"}}
    assert declared.required_for("weather") == frozenset({"city"})


def test_empty_schema_differs_from_unknown_schema() -> None:
    declared = tools.ToolSet.from_specs(
        ["unknown_schema", {"name": "noargs", "parameters": {"type": "object", "properties": {}}}]
    )

    assert declared.schema_for("unknown_schema") is None
    assert declared.schema_for("noargs") == {}
    assert declared.required_for("noargs") == frozenset()


def test_builtin_names_are_used_verbatim() -> None:
    declared = tools.ToolSet.from_specs(["bash", "weather"], builtin_tools=["bash"])

    assert declared.builtin_names == frozenset({"bash"})
    assert declared.wire_names == frozenset({"bash", "weather"})
    assert declared.is_known("bash")
    assert not declared.is_known("refl.bash")


def test_repeated_name_preserves_declared_schema() -> None:
    declared = tools.ToolSet.from_specs(
        [
            {"name": "weather", "parameters": {"properties": {"city": {"type": "string"}}, "required": ["city"]}},
            "weather",
        ]
    )

    assert len(declared.tools) == 1
    assert declared.schema_for("weather") == {"city": {"type": "string"}}
    assert declared.required_for("weather") == frozenset({"city"})


def test_no_declared_tools() -> None:
    declared = tools.ToolSet.from_specs()

    assert declared.wire_names == frozenset()
    assert not declared.is_known("weather")
    assert declared.schema_for("weather") is None
    assert declared.required_for("weather") == frozenset()


@pytest.mark.parametrize("parameters", [{"required": ["city"]}, {"properties": {}, "required": ["city"]}])
def test_required_names_do_not_depend_on_properties(parameters: dict[str, object]) -> None:
    declared = tools.ToolSet.from_specs([{"name": "weather", "parameters": parameters}])

    assert declared.required_for("weather") == frozenset({"city"})


@pytest.mark.parametrize(
    ("builtins", "wire_name", "invalid_alias"),
    [((), "weather", "refl.weather"), (("weather",), "weather", "refl.weather")],
)
def test_schema_lookups_only_accept_canonical_wire_names(
    builtins: tuple[str, ...], wire_name: str, invalid_alias: str
) -> None:
    declared = tools.ToolSet.from_specs(
        [{"name": "weather", "parameters": {"properties": {"city": {"type": "string"}}, "required": ["city"]}}],
        builtin_tools=builtins,
    )

    assert declared.schema_for(wire_name) == {"city": {"type": "string"}}
    assert declared.required_for(wire_name) == frozenset({"city"})
    assert not declared.is_known(invalid_alias)
    assert declared.schema_for(invalid_alias) is None
    assert declared.required_for(invalid_alias) == frozenset()


@pytest.mark.parametrize("name", ["web.search", "refl.custom"])
def test_declared_dotted_names_are_not_rewritten(name: str) -> None:
    declared = tools.ToolSet.from_specs([name], builtin_tools=[name])

    assert declared.wire_names == frozenset({name})
    assert declared.builtin_names == frozenset({name})
