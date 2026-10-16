"""Integration test: one registered dataset, run end to end against a live endpoint.

Every other test here is a unit test over pure logic, so this one is opt-in:

    uv run pytest tests/test_integration.py \
        --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct
"""

import pathlib

from beam_verifier import records
from beam_verifier.benchmarks import registry
from beam_verifier.benchmarks.scaffolds import base


def test_dataset_scores_against_a_live_endpoint(live_endpoint: str, model: str, api_key: str, tmp_path: pathlib.Path):
    """The real eval path: a registered dataset, called through the runner it carries."""
    outcome = registry.DATASETS["aime2026"].runner(
        base.RunRequest(
            base_url=live_endpoint,
            model_name=model,
            api_key=api_key,
            run_dir=tmp_path,
            settings=records.EvalSettings(num_examples=2, parallel_tasks=2, max_tokens=4096),
        )
    )
    assert outcome.result.n == 2
    assert outcome.result.metrics
    assert len(outcome.traces) == 2
    assert all(trace.score is not None or trace.error_type for trace in outcome.traces)
