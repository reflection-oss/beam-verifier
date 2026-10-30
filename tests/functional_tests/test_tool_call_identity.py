"""Tool-call identity ignores JSON formatting but preserves argument meaning."""

import pathlib
from typing import Any

import pytest
import yaml

from beam_verifier.functional_tests import completions

_CASES = yaml.safe_load((pathlib.Path(__file__).parent / "fixtures/tool_identity.yaml").read_text())


@pytest.mark.parametrize("case", _CASES, ids=[case["id"] for case in _CASES])
def test_canonical_tool_call_distinguishes_arguments(case: dict[str, Any], tool_call: dict[str, Any]) -> None:
    calls = [{**tool_call, "function": function} for function in case["functions"]]
    assert len({completions.canonical_tool_call(call) for call in calls}) == case["unique_count"]
