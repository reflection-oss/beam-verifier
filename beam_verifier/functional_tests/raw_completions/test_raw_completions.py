"""Generate raw Harmony completions and check the applicable probes."""

import dataclasses
import pathlib
from typing import Any

import httpx
import pytest

from beam_verifier import options
from beam_verifier.functional_tests import load, raw_completions_api
from beam_verifier.functional_tests.raw_completions.harmony_xml import probes, structure

raw_completion_cases = load.load_test_cases("harmony_xml", pathlib.Path(__file__).parent / "fixtures")


@pytest.mark.parametrize("case", raw_completion_cases["cases"], ids=load.case_id_generator)
def test_raw_completion(
    endpoint_client: httpx.Client,
    completions_url: str,
    model: str | None,
    case: dict[str, Any],
    json_metadata: dict[str, object],
    model_kwargs: dict[str, object],
) -> None:
    """Unconstrained generated text obeys Harmony framing, channels, and tool-call syntax."""
    # The posted body is the wire body, so engine options come up out of `extra_body`.
    sent: dict[str, Any] = dict(model_kwargs)
    engine_options = options.merge_options(
        {key: value for key, value in sent.items() if key != "extra_body"}, sent.get("extra_body", {})
    )
    body = options.merge_options(case["request"], engine_options)
    if model is not None:
        body["model"] = model
    builtin_tools = case.get("builtin_tools", [])
    json_metadata.update(
        profile=raw_completion_cases["profile"],
        request=body,
    )
    completion = raw_completions_api.generate_completion(endpoint_client, completions_url, body)
    result = structure.validate_completion(
        completion.text, tools=case["tools"], builtin_tools=builtin_tools, finish_reason=completion.finish_reason
    )
    json_metadata.update(
        completion=completion.model_dump(),
        conformance=dataclasses.asdict(result),
        declared_tools=case["tools"],
        builtin_tools=builtin_tools,
    )
    detail = f"{result.failure.probe_id}: {result.failure.message}" if result.failure else "incomplete completion"
    if not result.ok and not completion.text.rstrip().endswith(structure.TERMINATORS):
        detail += (
            "; the completion carries no terminating Harmony token, which is what an engine that trims"
            " stop strings returns. Pass the preservation option your engine spells, e.g."
            " --model-kwargs 'extra_body={no_stop_trim: true}' (SGLang) or"
            " --model-kwargs 'extra_body={include_stop_str_in_output: true}' (vLLM)"
        )
    assert not result.truncated, f"Raw completion truncated (finish_reason={completion.finish_reason}): {detail}"
    assert result.ok, f"Raw Harmony conformance failed: {detail}; text={completion.text!r}"
    if expected_shape := case.get("expected_shape"):
        assert result.shape == probes.CompletionShape(expected_shape)
    if case.get("expected_builtin_tool"):
        assert result.facets.builtin_tool
    if expected_content := case.get("expected_final_content"):
        final = completion.text.rsplit("<|message|>", 1)[-1].removesuffix("<|return|>")
        # A recalled word carries the casing of the sentence it lands in.
        assert all(value.lower() in final.lower() for value in expected_content), (
            f"Reasoning recall failed (Harmony syntax passed): expected {expected_content!r} "
            f"in the final answer, got {final!r}"
        )
