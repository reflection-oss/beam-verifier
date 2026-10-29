import pathlib
from typing import Any

import pytest
import yaml

from beam_verifier import records
from beam_verifier.reporting import run_store


@pytest.fixture
def report_examples() -> dict[str, Any]:
    """Loads fresh records so tests can change them independently."""
    return yaml.safe_load((pathlib.Path(__file__).parent / "fixtures/records.yaml").read_text())


@pytest.fixture
def run_manifest(report_examples: dict[str, Any]) -> records.RunManifest:
    return records.RunManifest.model_validate(report_examples["manifest"])


@pytest.fixture
def failed_dataset(report_examples: dict[str, Any]) -> records.DatasetResult:
    return records.DatasetResult.model_validate(report_examples["failed_dataset"])


@pytest.fixture
def trace_examples(report_examples: dict[str, Any]) -> dict[str, records.Trace]:
    return {name: records.Trace.model_validate(value) for name, value in report_examples["traces"].items()}


@pytest.fixture
def run_dir(tmp_path: pathlib.Path, run_manifest: records.RunManifest) -> pathlib.Path:
    path = tmp_path / "run"
    run_store.RunStore(path).write_manifest(run_manifest)
    return path
