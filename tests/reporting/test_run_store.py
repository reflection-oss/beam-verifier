"""Offline tests for records + run store (no network)."""

import json
import pathlib

import pytest

from beam_verifier import records
from beam_verifier.reporting import run_store


def test_request_hash_deterministic():
    m = [{"role": "user", "content": "hi"}]
    assert run_store.request_hash(m) == run_store.request_hash([{"role": "user", "content": "hi"}])
    assert run_store.request_hash(m) != run_store.request_hash([{"role": "user", "content": "bye"}])


def test_run_store_roundtrip(
    tmp_path: pathlib.Path, run_manifest: records.RunManifest, trace_examples: dict[str, records.Trace]
) -> None:
    store = run_store.RunStore(tmp_path / "run")
    traces = [trace_examples["graded_answer"]]
    store.write_traces("gpqa", traces)
    run_manifest.dataset_results = [
        records.DatasetResult(dataset="gpqa", topline_metric_name="grade", metrics={"grade": 100.0}, n=1, num_ok=1)
    ]
    store.write_manifest(run_manifest)

    m2, t2 = run_store.load_run(tmp_path / "run")
    assert m2 == run_manifest
    assert m2.dataset_results[0].status == "ok"
    assert t2["gpqa"][0].score == 1.0 and t2["gpqa"][0].content == "The answer is B"
    assert t2["gpqa"][0].finish_reason == "stop"


def test_verify_manifest_preserves_endpoint_urls(tmp_path):
    store = run_store.RunStore(tmp_path / "run")
    store.write_manifest(
        records.RunManifest(
            run_id="r",
            kind="verify",
            created_at="t",
            version="0.1.0",
            model="m",
            base_url="https://example.test/parse",
            config=records.EvalSettings(
                parse_url="https://example.test/parse",
                format_url="https://example.test/format",
            ),
        )
    )

    manifest, _ = run_store.load_run(tmp_path / "run")
    assert manifest.config.parse_url == "https://example.test/parse"
    assert manifest.config.format_url == "https://example.test/format"


def test_jsonl_unicode_line_separators_roundtrip(tmp_path):
    # Content with Unicode line separators that JSON leaves unescaped but str.splitlines() breaks on.
    store = run_store.RunStore(tmp_path / "run")
    nasty = "answer with\x85weird separators \\sin(\\theta)"
    store.write_traces("x", [records.Trace(index=0, request_hash="h", messages=[], content=nasty, score=1.0)])
    store.write_manifest(
        records.RunManifest(run_id="r", kind="eval", created_at="t", version="0.1.0", model="m", base_url="e")
    )
    _, t2 = run_store.load_run(tmp_path / "run")
    assert len(t2["x"]) == 1
    assert t2["x"][0].content == nasty


def test_load_manifest_renames_keys_written_under_an_earlier_schema(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "r",
                "kind": "eval",
                "created_at": "t",
                "reflection_eval_version": "0.1.0",
                "model": "m",
                "endpoint": "http://endpoint/v1",
                "evalsets": [{"evalset": "gpqa-diamond", "topline_metric_name": "accuracy"}],
            }
        )
    )
    manifest = run_store.load_manifest(run_dir)
    assert manifest.version == "0.1.0"
    assert manifest.base_url == "http://endpoint/v1"
    assert [result.dataset for result in manifest.dataset_results] == ["gpqa-diamond"]


@pytest.mark.source_checkout
def test_default_runs_dir_is_dot_runs_at_the_package_root():
    root = run_store.package_root()
    assert root is not None
    assert (root / "pyproject.toml").is_file()
    assert run_store.default_runs_dir() == root / run_store.RUNS_DIR_NAME


def test_new_run_dir_honors_an_explicit_base(tmp_path):
    run_dir = run_store.new_run_dir("zai-org/GLM-5.2-FP8", "eval", base=tmp_path)
    assert run_dir.parent == tmp_path
    assert run_dir.name.endswith("-zai-org-GLM-5.2-FP8-eval")
    assert (run_dir / "traces").is_dir()


def test_failed_dataset_is_saved_with_its_error(
    tmp_path: pathlib.Path, run_manifest: records.RunManifest, failed_dataset: records.DatasetResult
) -> None:
    """A dataset whose runner raised stays in the saved run, with the reason why."""
    store = run_store.RunStore(tmp_path / "run")
    run_manifest.dataset_results = [
        failed_dataset,
        records.DatasetResult(dataset="math500", topline_metric_name="grade", metrics={"grade": 42.0}, n=1, num_ok=1),
    ]
    store.write_manifest(run_manifest)
    manifest, _ = run_store.load_run(tmp_path / "run")
    failed = manifest.dataset_results[0]
    assert failed.dataset == "aime2026" and failed.status == "failed"
    assert failed.error == "task loading failed"
    assert not failed.metrics
