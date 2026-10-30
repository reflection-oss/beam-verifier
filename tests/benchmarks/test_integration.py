"""Opt-in benchmark smoke test; the ordinary suite makes no endpoint requests.

Run with: `uv run pytest tests/benchmarks/test_integration.py --base-url <url>/v1 --model <id>`.
Set `REFLECTION_API_KEY` if the endpoint requires authentication.
"""

import os
import pathlib

import pytest

from beam_verifier import records
from beam_verifier.benchmarks import registry
from beam_verifier.benchmarks.scaffolds import base


def test_dataset_scores_against_a_live_endpoint(request: pytest.FixtureRequest, tmp_path: pathlib.Path) -> None:
    base_url = request.config.getoption("--base-url")
    if not base_url:
        pytest.skip("pass --base-url and --model to run the integration test")
    model = request.config.getoption("--model")
    if not model:
        pytest.fail("--base-url requires --model")
    outcome = registry.DATASETS["aime2026"].runner(
        base.RunRequest(
            base_url=base_url,
            model_name=model,
            api_key=request.config.getoption("--api-key") or os.environ.get("REFLECTION_API_KEY") or "EMPTY",
            run_dir=tmp_path,
            settings=records.EvalSettings(num_examples=2, parallel_tasks=2, max_tokens=4096),
        )
    )
    assert outcome.result.n == 2
    assert outcome.result.num_error == 0, outcome.result.error_types
    assert outcome.result.metrics
    assert len(outcome.traces) == 2
    assert all(trace.score is not None for trace in outcome.traces)
