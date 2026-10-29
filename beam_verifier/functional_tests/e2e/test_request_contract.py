"""Request-contract tests: what the endpoint accepts, rejects, and carries into the prompt."""

import pathlib
from typing import Any

import openai
import pytest

from beam_verifier.functional_tests import completions, load

request_contract_test_cases = load.load_test_cases("request_contract", pathlib.Path(__file__).parent / "fixtures")


def _name_value_pairs(rows: list[dict[str, Any]], key: str) -> list[tuple[object, object]]:
    """Flattens fixture rows into (name, value) parametrize pairs."""
    pairs: list[tuple[object, object]] = []
    for row in rows:
        pairs.extend((row["name"], value) for value in row[key])
    return pairs


accepted_params = _name_value_pairs(request_contract_test_cases["input_parameters_values"], "accepted")
rejected_params = _name_value_pairs(request_contract_test_cases["input_parameters_values"], "rejected")
prior_turn_reasoning_cases = request_contract_test_cases["prior_turn_reasoning_cases"]

# Param probes are request-validation checks (shape, not content): use a short
# prompt and non-streaming transport while sharing the bounded generation budget.
PARAM_PROBE_MESSAGES = [{"role": "user", "content": "Say 'OK' and nothing else."}]


def test_minimal_request_accepted(http_client, model):
    """A minimal request (only required fields, no decoding params) succeeds."""
    completions.generate_completion(http_client, model, PARAM_PROBE_MESSAGES, stream=False)


@pytest.mark.parametrize(("param_name", "value"), accepted_params)
def test_param_value_accepted(http_client, model, param_name, value):
    """Each valid decoding param value succeeds."""
    completions.generate_completion(
        http_client,
        model,
        PARAM_PROBE_MESSAGES,
        stream=False,
        extra_params={param_name: value},
    )


@pytest.mark.parametrize(("param_name", "value"), rejected_params)
def test_param_value_rejected(http_client, model, param_name, value):
    """A non-valid decoding param value is refused."""
    with pytest.raises(openai.BadRequestError):
        completions.generate_completion(
            http_client,
            model,
            PARAM_PROBE_MESSAGES,
            stream=False,
            extra_params={param_name: value},
        )


@pytest.mark.parametrize("case", prior_turn_reasoning_cases, ids=load.case_id_generator)
def test_prior_turn_thinking_is_replayed(http_client, model, stream, case):
    """A value stated only in a prior turn's reasoning_content is available to the next turn."""
    completion = completions.generate_completion(
        http_client,
        model,
        case["input"]["messages"],
        stream=stream,
    )
    answer = case["expected"]["answer"]
    assert answer.lower() in completion.content.lower(), (
        f"Expected {answer!r}, stated only in the prior turn's reasoning_content, got {completion.content!r}"
    )


@pytest.mark.parametrize(
    "case",
    request_contract_test_cases["test_malformed_request_rejected"]["test_params"]["case"],
    ids=load.case_id_generator,
)
def test_malformed_request_rejected(http_client, model, stream, case):
    """A malformed request is refused."""
    with pytest.raises(openai.BadRequestError):
        completions.generate_completion(
            http_client,
            model,
            case["input"]["messages"],
            stream=stream,
            tools=case["input"].get("tools"),
            tool_choice=case["input"].get("tool_choice"),
        )
