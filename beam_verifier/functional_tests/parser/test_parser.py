"""Deterministic extraction checks using pinned Harmony XML token IDs."""

import dataclasses
import pathlib
from typing import Any

import httpx
import jsonschema
import pytest

from beam_verifier.functional_tests import load, native_api, parser_api

parser_cases = load.load_test_cases("harmony_xml", pathlib.Path(__file__).parent / "fixtures")


def _assert_argument_schemas(result: parser_api.ParsedMessage, tools: list[dict[str, Any]]) -> None:
    """Check declared types, including boolean versus integer equality in Python."""
    schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
    for call in result.tool_calls:
        jsonschema.validate(call.arguments, schemas[call.name])


@pytest.mark.parametrize("case", parser_cases["message_cases"], ids=load.case_id_generator)
def test_parse_assistant_message(
    parser_client: parser_api.Parser,
    case: dict[str, Any],
    json_metadata: dict[str, object],
) -> None:
    """Pinned output IDs recover the expected assistant fields with schema coercion."""
    json_metadata.update(request=case["input"], expected=case["expected"])
    try:
        result = parser_client.parse_message(case["input"]["tokens"], case["input"]["tools"])
    except native_api.UnsupportedOperation as error:
        json_metadata["skip_reason"] = str(error)
        pytest.skip(str(error))
    actual = dataclasses.asdict(result)
    json_metadata["actual"] = actual
    assert parser_api.json_equal(actual, case["expected"]), (
        f"Assistant parsing mismatch: expected {case['expected']!r}, got {actual!r}"
    )
    _assert_argument_schemas(result, case["input"]["tools"])


@pytest.mark.parametrize("case", parser_cases["rejection_cases"], ids=load.case_id_generator)
def test_reject_malformed_output(
    parser_client: parser_api.Parser,
    case: dict[str, Any],
    json_metadata: dict[str, object],
) -> None:
    """Malformed or truncated tool wire fails instead of returning a partial call."""
    json_metadata.update(request=case["input"], expected_status=400)
    try:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            result = parser_client.parse_message(case["input"]["tokens"], case["input"]["tools"])
            json_metadata["actual"] = dataclasses.asdict(result)
    except native_api.UnsupportedOperation as error:
        json_metadata["skip_reason"] = str(error)
        pytest.skip(str(error))
    assert caught.value.response.status_code == 400, (
        f"Malformed parser output must return HTTP 400, got {caught.value.response.status_code}"
    )
