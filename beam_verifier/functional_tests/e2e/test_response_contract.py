"""Response-contract tests: responses follow the requested output format."""

import json
import pathlib

import jsonschema
import pytest

from beam_verifier.functional_tests import completions, load

response_contract_test_cases = load.load_test_cases("response_contract", pathlib.Path(__file__).parent / "fixtures")


@pytest.mark.parametrize(
    "case",
    response_contract_test_cases["test_chat_structured_output"]["test_params"]["case"],
    ids=load.case_id_generator,
)
def test_chat_structured_output(http_client, model, stream, case):
    """Input is a constrained decoding request, specifying a JSON schema - output expects a valid JSON satisfying it."""
    completion = completions.generate_completion(
        http_client,
        model,
        case["input"]["messages"],
        stream=stream,
        response_format=case["input"]["response_format"],
    )
    parsed = json.loads(completion.content)
    jsonschema.validate(parsed, case["input"]["response_format"]["json_schema"]["schema"])
    for key, value in case["expected"].items():
        assert parsed.get(key) == value, f"expected {key}={value!r}, got {parsed.get(key)!r}"
