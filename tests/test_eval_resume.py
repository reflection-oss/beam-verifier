"""Resume the CLI across an interrupted runner without calling an endpoint."""

import pathlib
from collections.abc import Collection

import click.testing
import pytest

from beam_verifier import cli, records
from beam_verifier.benchmarks import registry
from beam_verifier.benchmarks.scaffolds import base
from beam_verifier.benchmarks.scaffolds.inspect_ai import runner as inspect_ai_runner
from beam_verifier.reporting import run_store


@pytest.fixture(autouse=True)
def available_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CLI state and argument tests independent of a live endpoint."""
    monkeypatch.setattr(cli, "check_endpoint", lambda *args: True)


class InterruptibleRunner(base.DatasetRunner):
    MODULE = "offline"

    def __init__(self, dataset_id: str, *, interrupt: bool = False) -> None:
        self.dataset_id = dataset_id
        self.interrupt = interrupt
        self.requests: list[base.RunRequest] = []

    def unsupported_resume_overrides(self, settings: records.EvalSettings, names: Collection[str]) -> list[str]:
        del settings, names  # This fake applies the entire request on every invocation.
        return []

    def __call__(self, request: base.RunRequest) -> base.RunOutcome:
        self.requests.append(request)
        if self.interrupt:
            raise KeyboardInterrupt
        return base.RunOutcome(
            result=records.DatasetResult(
                dataset=self.dataset_id, topline_metric_name="accuracy", metrics={"accuracy": 0.0}, n=1, num_ok=1
            ),
            traces=[records.Trace(index=0, request_hash="sample", messages=[], score=0.0)],
        )


def test_resume_preserves_finished_datasets_and_runs_the_remaining_plan(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    complete = InterruptibleRunner("complete")
    interrupted = InterruptibleRunner("interrupted", interrupt=True)
    unstarted = InterruptibleRunner("unstarted")
    monkeypatch.setitem(registry.DATASETS, "complete", registry.Dataset(runner=complete))
    monkeypatch.setitem(registry.DATASETS, "interrupted", registry.Dataset(runner=interrupted))
    monkeypatch.setitem(registry.DATASETS, "unstarted", registry.Dataset(runner=unstarted))
    endpoint = ["--base-url", "http://unused/v1", "--model-name", "model"]
    command = click.testing.CliRunner()
    first = command.invoke(
        cli.cli,
        [
            "eval",
            "complete",
            "interrupted",
            "unstarted",
            *endpoint,
            "--temperature",
            "0.3",
            "--harness-kwargs",
            "epochs=25",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert first.exit_code == 1
    manifest = run_store.load_manifest(tmp_path)
    created_at = manifest.created_at
    assert manifest.dataset_ids == ["complete", "interrupted", "unstarted"]
    assert [result.dataset for result in manifest.dataset_results] == ["complete"]
    assert manifest.finished_at is None
    completed_traces = (tmp_path / "traces" / "complete.jsonl").read_bytes()

    interrupted.interrupt = False
    resumed = command.invoke(cli.cli, ["eval", *endpoint, "--resume", str(tmp_path)])
    assert resumed.exit_code == 0, resumed.output
    manifest, traces = run_store.load_run(tmp_path)
    assert [result.dataset for result in manifest.dataset_results] == ["complete", "interrupted", "unstarted"]
    assert len(complete.requests) == 1
    assert len(interrupted.requests) == 2
    assert len(unstarted.requests) == 1
    assert interrupted.requests[1].resume
    assert unstarted.requests[0].settings.temperature == 0.3
    assert unstarted.requests[0].settings.harness_kwargs == {"epochs": 25}
    assert manifest.config.harness_kwargs == {"epochs": 25}
    assert (tmp_path / "traces" / "complete.jsonl").read_bytes() == completed_traces
    assert traces["interrupted"][0].score == 0.0
    assert manifest.finished_at is not None
    assert manifest.created_at == created_at

    # A second resume of the completed run performs no evaluation.
    again = command.invoke(cli.cli, ["eval", *endpoint, "--resume", str(tmp_path)])
    assert again.exit_code == 0, again.output
    assert len(interrupted.requests) == 2
    assert len(unstarted.requests) == 1


def test_interruption_during_first_dataset_leaves_a_run_plan(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interrupted = InterruptibleRunner("interrupted", interrupt=True)
    monkeypatch.setitem(registry.DATASETS, "interrupted", registry.Dataset(runner=interrupted))
    result = click.testing.CliRunner().invoke(
        cli.cli,
        [
            "eval",
            "interrupted",
            "--base-url",
            "http://unused/v1",
            "--model-name",
            "model",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    manifest = run_store.load_manifest(tmp_path)
    assert manifest.dataset_ids == ["interrupted"]
    assert manifest.dataset_results == []


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--model-name", "different"], "saved model"),
        (["--output-dir", "/unused"], "writes into the saved run"),
        (["math500"], "cannot change the saved dataset list"),
    ],
)
def test_resume_rejects_changes_before_touching_the_run(
    tmp_path: pathlib.Path, options: list[str], message: str
) -> None:
    run_store.RunStore(tmp_path).write_manifest(
        records.RunManifest(
            run_id="run",
            kind="eval",
            created_at="2026-09-01T00:00:00",
            version="0.1.0",
            model="model",
            base_url="http://unused/v1",
            config=records.EvalSettings(temperature=0.3),
            dataset_ids=["aime2026"],
        )
    )
    original = (tmp_path / "run.json").read_bytes()
    result = click.testing.CliRunner().invoke(
        cli.cli,
        ["eval", "--base-url", "http://unused/v1", "--model-name", "model", "--resume", str(tmp_path), *options],
    )
    assert result.exit_code == 1
    assert message in result.output
    assert (tmp_path / "run.json").read_bytes() == original


def test_resume_records_overrides_beside_the_settings_it_was_launched_with(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`config` keeps the original settings; a resume's overrides are recorded beside it."""
    interrupted = InterruptibleRunner("interrupted", interrupt=True)
    monkeypatch.setitem(registry.DATASETS, "interrupted", registry.Dataset(runner=interrupted))
    endpoint = ["--base-url", "http://unused/v1", "--model-name", "model"]
    command = click.testing.CliRunner()
    command.invoke(
        cli.cli,
        [
            "eval",
            "interrupted",
            *endpoint,
            "--temperature",
            "0.3",
            "--harness-kwargs",
            "epochs=2",
            "--output-dir",
            str(tmp_path),
        ],
    )
    interrupted.interrupt = False
    resumed = command.invoke(
        cli.cli,
        ["eval", *endpoint, "--resume", str(tmp_path), "--temperature", "0.7", "--harness-kwargs", "epochs=3"],
    )

    assert resumed.exit_code == 0, resumed.output
    manifest = run_store.load_manifest(tmp_path)
    assert manifest.config.temperature == 0.3
    assert manifest.config.harness_kwargs == {"epochs": 2}
    assert manifest.resume_overrides == [{"temperature": 0.7, "harness_kwargs": {"epochs": 3}}]
    assert interrupted.requests[-1].settings.temperature == 0.7
    assert interrupted.requests[-1].settings.harness_kwargs == {"epochs": 3}


def test_repeated_resumes_keep_every_override_and_the_latest_value(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each resume is recorded, and one that renames nothing keeps the value in effect."""
    first = InterruptibleRunner("first", interrupt=True)
    second = InterruptibleRunner("second", interrupt=True)
    third = InterruptibleRunner("third", interrupt=True)
    monkeypatch.setitem(registry.DATASETS, "first", registry.Dataset(runner=first))
    monkeypatch.setitem(registry.DATASETS, "second", registry.Dataset(runner=second))
    monkeypatch.setitem(registry.DATASETS, "third", registry.Dataset(runner=third))
    endpoint = ["--base-url", "http://unused/v1", "--model-name", "model"]
    command = click.testing.CliRunner()
    command.invoke(
        cli.cli,
        ["eval", "first", "second", "third", *endpoint, "--parallel-tasks", "32", "--output-dir", str(tmp_path)],
    )
    first.interrupt = False
    command.invoke(cli.cli, ["eval", *endpoint, "--resume", str(tmp_path), "--parallel-tasks", "8"])
    second.interrupt = False
    command.invoke(cli.cli, ["eval", *endpoint, "--resume", str(tmp_path), "--parallel-tasks", "4"])
    third.interrupt = False
    # This resume names nothing, so the value the run last used stays in effect.
    command.invoke(cli.cli, ["eval", *endpoint, "--resume", str(tmp_path)])

    manifest = run_store.load_manifest(tmp_path)
    assert manifest.config.parallel_tasks == 32
    assert manifest.resume_overrides == [{"parallel_tasks": 8}, {"parallel_tasks": 4}]
    assert third.requests[-1].settings.parallel_tasks == 4


def test_resume_reports_an_override_the_harness_cannot_apply() -> None:
    """InspectAI restores generation settings from its log, so it names them unapplied."""
    runner_under_test = inspect_ai_runner.InspectAIRunner("aime2026")
    settings = records.EvalSettings(
        temperature=0.7, parallel_tasks=4, harness_kwargs={"epochs": 3, "fail_on_error": False}
    )
    unsupported = runner_under_test.unsupported_resume_overrides(
        settings, ["temperature", "parallel_tasks", "harness_kwargs"]
    )
    assert unsupported == ["--temperature", "--harness-kwargs epochs"]


def test_resume_rejects_a_missing_manifest(tmp_path: pathlib.Path) -> None:
    result = click.testing.CliRunner().invoke(
        cli.cli, ["eval", "--base-url", "http://unused/v1", "--model-name", "model", "--resume", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "cannot read" in result.output
    assert not (tmp_path / "run.json").exists()


@pytest.mark.parametrize("status", ["partial", "failed"])
def test_resume_replaces_the_prior_result_instead_of_appending(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    dataset_runner = InterruptibleRunner("offline")
    monkeypatch.setitem(registry.DATASETS, "offline", registry.Dataset(runner=dataset_runner))
    run_store.RunStore(tmp_path).write_manifest(
        records.RunManifest(
            run_id="run",
            kind="eval",
            created_at="2026-09-01T00:00:00",
            version="0.1.0",
            model="model",
            base_url="http://unused/v1",
            dataset_ids=["offline"],
            dataset_settings={"offline": records.EvalSettings()},
            dataset_results=[records.DatasetResult(dataset="offline", topline_metric_name="accuracy", status=status)],
        )
    )
    result = click.testing.CliRunner().invoke(
        cli.cli, ["eval", "--base-url", "http://unused/v1", "--model-name", "model", "--resume", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    (saved,) = run_store.load_manifest(tmp_path).dataset_results
    assert saved.status == "ok"
    assert saved.metrics == {"accuracy": 0.0}


def test_resume_keeps_pinned_defaults_for_unstarted_datasets(tmp_path, monkeypatch):
    interrupted = InterruptibleRunner("interrupted", interrupt=True)
    unstarted = InterruptibleRunner("unstarted")
    monkeypatch.setitem(
        registry.DATASETS,
        "interrupted",
        registry.Dataset(runner=interrupted, config="configs/gpqa-diamond.yaml"),
    )
    dataset = registry.Dataset(runner=unstarted, config="configs/gpqa-diamond.yaml")
    monkeypatch.setitem(registry.DATASETS, "unstarted", dataset)
    command = click.testing.CliRunner()
    endpoint = ["--base-url", "http://unused/v1", "--model-name", "model"]
    first = command.invoke(cli.cli, ["eval", "interrupted", "unstarted", *endpoint, "--output-dir", str(tmp_path)])
    assert first.exit_code == 1
    assert run_store.load_manifest(tmp_path).dataset_settings["unstarted"].max_tokens is None
    dataset.config = "configs/aime2026.yaml"
    interrupted.interrupt = False
    resumed = command.invoke(cli.cli, ["eval", *endpoint, "--resume", str(tmp_path), "--temperature", "0"])
    assert resumed.exit_code == 0, resumed.output
    assert unstarted.requests[0].settings.max_tokens is None
    assert unstarted.requests[0].settings.temperature == 0
    assert unstarted.requests[0].settings.harness_kwargs == {"epochs": 16}
    factory = unstarted.requests[0].settings.prompt_factory
    assert factory is not None
    assert factory.name == "beam-gpqa-v1"
    assert factory.system_text == registry.DATASETS["gpqa-diamond"].resolve_settings().prompt_factory.system_text


def test_resume_rejects_missing_dataset_snapshots(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recording = InterruptibleRunner("old-run")
    monkeypatch.setitem(
        registry.DATASETS, "old-run", registry.Dataset(runner=recording, config="configs/gpqa-diamond.yaml")
    )
    run_store.RunStore(tmp_path).write_manifest(
        records.RunManifest(
            run_id="old-run",
            kind="eval",
            created_at="2026-09-01T00:00:00",
            version="test",
            model="model",
            base_url="http://unused/v1",
            config=records.EvalSettings(temperature=0.3),
            dataset_ids=["old-run"],
        )
    )
    original = (tmp_path / "run.json").read_bytes()
    result = click.testing.CliRunner().invoke(
        cli.cli,
        [
            "eval",
            "--base-url",
            "http://unused/v1",
            "--model-name",
            "model",
            "--resume",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1, result.output
    assert "--resume requires saved dataset settings" in result.output
    assert recording.requests == []
    assert (tmp_path / "run.json").read_bytes() == original


@pytest.mark.parametrize(
    "options",
    [
        ["--temperature", "0"],
        ["--harness-kwargs", "epochs=16"],
        ["--model-kwargs", "extra_body={custom_option: false}"],
        ["--model-kwargs", "extra_body={}"],
    ],
)
def test_resume_unchanged_dataset_options_preserve_saved_settings(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, options: list[str]
) -> None:
    recording = InterruptibleRunner("offline")
    monkeypatch.setitem(registry.DATASETS, "offline", registry.Dataset(runner=recording))
    saved = records.EvalSettings(
        temperature=0,
        harness_kwargs={"epochs": 16, "fail_on_error": False},
        model_kwargs={"extra_body": {"custom_option": False, "logprobs": True}},
    )
    run_store.RunStore(tmp_path).write_manifest(
        records.RunManifest(
            run_id="run",
            kind="eval",
            created_at="2026-09-01T00:00:00",
            version="test",
            model="model",
            base_url="http://unused/v1",
            dataset_ids=["offline"],
            dataset_settings={"offline": saved},
        )
    )
    result = click.testing.CliRunner().invoke(
        cli.cli,
        ["eval", "--base-url", "http://unused/v1", "--model-name", "model", "--resume", str(tmp_path), *options],
    )
    assert result.exit_code == 0, result.output
    assert run_store.load_manifest(tmp_path).resume_overrides == []
    assert recording.requests[0].settings == saved
    assert run_store.load_manifest(tmp_path).dataset_settings == {"offline": saved}


@pytest.mark.parametrize(
    ("saved", "options", "expected", "override"),
    [
        (
            records.EvalSettings(model_kwargs={"extra_body": {"custom_option": True, "logprobs": False}}),
            ["--model-kwargs", "extra_body={custom_option: null}"],
            records.EvalSettings(model_kwargs={"extra_body": {"custom_option": None, "logprobs": False}}),
            {"model_kwargs": {"extra_body": {"custom_option": None}}},
        ),
        (
            records.EvalSettings(harness_kwargs={"sample_shuffle": True}),
            ["--harness-kwargs", "sample_shuffle=1"],
            records.EvalSettings(harness_kwargs={"sample_shuffle": 1}),
            {"harness_kwargs": {"sample_shuffle": 1}},
        ),
    ],
)
def test_resume_records_native_option_changes(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    saved: records.EvalSettings,
    options: list[str],
    expected: records.EvalSettings,
    override: dict[str, object],
) -> None:
    recording = InterruptibleRunner("offline")
    monkeypatch.setitem(registry.DATASETS, "offline", registry.Dataset(runner=recording))
    run_store.RunStore(tmp_path).write_manifest(
        records.RunManifest(
            run_id="run",
            kind="eval",
            created_at="2026-09-01T00:00:00",
            version="test",
            model="model",
            base_url="http://unused/v1",
            dataset_ids=["offline"],
            dataset_settings={"offline": saved},
        )
    )
    result = click.testing.CliRunner().invoke(
        cli.cli,
        ["eval", "--base-url", "http://unused/v1", "--model-name", "model", "--resume", str(tmp_path), *options],
    )
    assert result.exit_code == 0, result.output
    assert run_store.load_manifest(tmp_path).resume_overrides == [override]
    assert recording.requests[0].settings.model_dump_json() == expected.model_dump_json()


@pytest.mark.parametrize("options", [["--temperature", "0"], ["--harness-kwargs", "n_attempts=1"]])
def test_harbor_resume_rejects_changed_settings_without_modifying_manifest(
    tmp_path: pathlib.Path, options: list[str]
) -> None:
    settings = registry.DATASETS["swebench-pro"].resolve_settings()
    run_store.RunStore(tmp_path).write_manifest(
        records.RunManifest(
            run_id="run",
            kind="eval",
            created_at="2026-09-01T00:00:00",
            version="0.1.0",
            model="model",
            base_url="http://unused/v1",
            config=settings,
            dataset_ids=["swebench-pro"],
            dataset_settings={"swebench-pro": settings},
        )
    )
    original = (tmp_path / "run.json").read_bytes()
    result = click.testing.CliRunner().invoke(
        cli.cli,
        ["eval", "--base-url", "http://unused/v1", "--model-name", "model", "--resume", str(tmp_path), *options],
    )
    assert result.exit_code == 1, result.output
    assert "cannot apply changed settings" in result.output
    assert (tmp_path / "run.json").read_bytes() == original


class InterruptedInspectAIRunner(inspect_ai_runner.InspectAIRunner):
    def __call__(self, request: base.RunRequest) -> base.RunOutcome:
        del request  # Stop after the real CLI and InspectAI capability check.
        raise KeyboardInterrupt


def test_resume_accepts_native_inspect_ai_change_beside_unchanged_epochs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(registry.DATASETS, "offline", registry.Dataset(runner=InterruptedInspectAIRunner("offline")))
    saved = records.EvalSettings(harness_kwargs={"epochs": 16})
    run_store.RunStore(tmp_path).write_manifest(
        records.RunManifest(
            run_id="run",
            kind="eval",
            created_at="2026-09-01T00:00:00",
            version="0.1.0",
            model="model",
            base_url="http://unused/v1",
            dataset_ids=["offline"],
            dataset_settings={"offline": saved},
        )
    )
    result = click.testing.CliRunner().invoke(
        cli.cli,
        [
            "eval",
            "--base-url",
            "http://unused/v1",
            "--model-name",
            "model",
            "--resume",
            str(tmp_path),
            "--harness-kwargs",
            "epochs=16",
            "--harness-kwargs",
            "fail_on_error=false",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "Aborted" in result.output
    assert "cannot apply" not in result.output
    assert run_store.load_manifest(tmp_path).dataset_settings["offline"].harness_kwargs == {
        "epochs": 16,
        "fail_on_error": False,
    }
