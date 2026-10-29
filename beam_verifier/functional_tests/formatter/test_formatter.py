"""Fixed Harmony prompts check formatting independently of generation."""

import pathlib
from typing import Any

import httpx
import pytest

from beam_verifier.functional_tests import formatter_api, load, native_api

formatter_cases = load.load_test_cases("harmony_xml", pathlib.Path(__file__).parent / "fixtures")


@pytest.mark.parametrize("case", formatter_cases["render_cases"], ids=load.case_id_generator)
def test_render_chat(
    formatter_client: formatter_api.Formatter, case: dict[str, Any], json_metadata: dict[str, object]
) -> None:
    """Chat messages render to the exact token sequence of an independent Harmony prompt."""
    json_metadata.update(
        request=case["input"],
        expected_prompt=case["expected_prompt"],
    )
    json_metadata["transmitted_request"] = formatter_client.chat_request(case["input"])
    try:
        actual = formatter_client.render(case["input"])
        expected = case["expected_tokens"]
    except native_api.UnsupportedOperation as error:
        json_metadata["skip_reason"] = str(error)
        pytest.skip(str(error))
    json_metadata.update(actual_tokens=actual, expected_tokens=expected)
    assert actual == expected, f"Formatter token mismatch for {case['case_id']}: expected {expected!r}, got {actual!r}"


@pytest.mark.parametrize("case", formatter_cases["rejection_cases"], ids=load.case_id_generator)
def test_reject_invalid_chat(
    formatter_client: formatter_api.Formatter, case: dict[str, Any], json_metadata: dict[str, object]
) -> None:
    """Invalid Harmony formatter inputs return HTTP 400 without generation."""
    json_metadata.update(request=case["input"], expected_status=400)
    json_metadata["transmitted_request"] = formatter_client.chat_request(case["input"])
    try:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            formatter_client.render(case["input"])
    except native_api.UnsupportedOperation as error:
        json_metadata["skip_reason"] = str(error)
        pytest.skip(str(error))
    assert caught.value.response.status_code == 400, (
        f"Invalid formatter input must return HTTP 400, got {caught.value.response.status_code}"
    )
