"""Offline tests for the dataset table (no network; datasets are not loaded here)."""

import subprocess
import sys

import pytest

from beam_verifier.benchmarks import registry as benchmarks_registry
from beam_verifier.benchmarks.scaffolds.inspect_ai import runner as inspect_ai

# The task each registered InspectAI dataset resolves to. `inspect_evals` names its
# tasks after the datasets, so the mapping is mechanical — but a dataset whose
# upstream task is renamed has to say so on its row rather than silently run another.
INSPECT_AI_TASKS = {
    "gpqa-diamond": "inspect_evals/gpqa_diamond",
    "aime2026": "inspect_evals/aime2026",
}


@pytest.mark.parametrize("dataset_id", ["aime2026", "gpqa-diamond"])
def test_the_reported_datasets_are_registered(dataset_id: str):
    """The datasets whose scores this package publishes have to be runnable: these are
    the ids the reproduction runs and the README quote."""
    assert dataset_id in benchmarks_registry.DATASETS


@pytest.mark.parametrize(
    "dataset_id", [name for name, dataset in benchmarks_registry.DATASETS.items() if dataset.run_by_default]
)
def test_eval_datasets_have_packaged_defaults(dataset_id: str):
    dataset = benchmarks_registry.DATASETS[dataset_id]
    assert dataset.config is not None
    assert dataset.resolve_settings() is not None


@pytest.mark.parametrize(
    "dataset_id", [name for name, dataset in benchmarks_registry.DATASETS.items() if not dataset.run_by_default]
)
def test_optional_datasets_keep_native_prompts_without_yaml_configs(dataset_id: str):
    dataset = benchmarks_registry.DATASETS[dataset_id]
    assert dataset.config is None
    assert dataset.resolve_settings().prompt_factory is None


@pytest.mark.parametrize(("dataset_id", "task"), sorted(INSPECT_AI_TASKS.items()))
def test_inspect_ai_dataset_wiring(dataset_id: str, task: str):
    """Every InspectAI row names the task it runs, so a renamed upstream task shows up
    here rather than as a dataset quietly scoring something else."""
    runner = benchmarks_registry.DATASETS[dataset_id].runner
    assert isinstance(runner, inspect_ai.InspectAIRunner)
    assert runner.resolved_task == task


def test_default_evaluation_runs_the_reference_datasets():
    assert {name for name, dataset in benchmarks_registry.DATASETS.items() if dataset.run_by_default} == {
        "gpqa-diamond",
        "aime2026",
    }


def test_table_is_keyed_by_dataset_id():
    """The key is the id the command line names, and the runner owns that id."""
    assert all(dataset_id == dataset.dataset_id for dataset_id, dataset in benchmarks_registry.DATASETS.items())


def test_dataset_id_comes_from_its_runner():
    """The runner owns the dataset's identity, so the two can never disagree."""
    dataset = benchmarks_registry.Dataset(runner=inspect_ai.InspectAIRunner(dataset_id="somewhere"))
    assert dataset.dataset_id == "somewhere"


def test_registry_import_does_not_load_eval_libraries() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from beam_verifier.benchmarks import registry
assert 'inspect_ai' not in sys.modules
""",
        ],
        check=True,
    )
