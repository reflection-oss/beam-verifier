"""Check Harbor configuration and error reporting without creating sandboxes."""

import contextlib
import datetime
import itertools
import json
import os
import pathlib
import signal
import subprocess
import sys
from collections.abc import Coroutine
from typing import Literal

import pytest
from harbor import job
from harbor.agents import factory
from harbor.environments import factory as environment_factory
from harbor.models import environment_type
from harbor.models.task import id as task_id
from harbor.models.trial import config as trial_config
from harbor.models.trial import result as trial_result
from harbor.models.verifier import result as verifier_result
from harbor.trial import hooks, queue

from beam_verifier import records
from beam_verifier.benchmarks import prompts
from beam_verifier.benchmarks.scaffolds import base
from beam_verifier.benchmarks.scaffolds.harbor import runner, worker
from beam_verifier.benchmarks.scaffolds.harbor.mirror import agent


@pytest.mark.parametrize(
    "sandbox, environment_options, provider",
    [
        ("daytona", {}, environment_type.EnvironmentType.DAYTONA),
        ("docker", {}, environment_type.EnvironmentType.DOCKER),
        ("daytona", {"type": "docker", "force_build": False}, environment_type.EnvironmentType.DOCKER),
    ],
)
def test_build_job_config_forwards_native_options(
    tmp_path: pathlib.Path,
    sandbox: Literal["docker", "daytona"],
    environment_options: dict[str, object],
    provider: environment_type.EnvironmentType,
) -> None:
    scaffold = runner.HarborRunner("coding", {"name": "example/tasks", "ref": "sha256:pinned"}, {"tools": "bash"})
    request = base.RunRequest(
        "https://endpoint.invalid/v1",
        "served-model",
        tmp_path,
        settings=records.EvalSettings(
            sandbox=sandbox,
            num_examples=2,
            parallel_tasks=3,
            temperature=0,
            max_tokens=512,
            model_kwargs={"top_p": 0.9, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            harness_kwargs={
                "n_attempts": 2,
                "environment": {"kwargs": {"region": "us"}, **environment_options},
                "n_concurrent_trials": 4,
                "agents": [{"kwargs": {"mirror_kwargs": {"disable_compaction": True}}}],
            },
            prompt_factory=prompts.BeamPromptFactory("task", "Do: {prompt}", "Be concise."),
        ),
    )
    config = scaffold.build_job_config(request)
    assert config.environment.type == provider
    assert config.environment.kwargs == {"region": "us"}
    assert config.environment.force_build == environment_options.get(
        "force_build", provider == environment_type.EnvironmentType.DOCKER
    )
    assert config.n_attempts == 2
    assert config.n_concurrent_trials == 4
    assert config.datasets[0].ref == "sha256:pinned"
    assert config.datasets[0].n_tasks == 2
    assert config.agents[0].env == {}
    instance = factory.AgentFactory.create_agent_from_config(config.agents[0], logs_dir=tmp_path)
    assert isinstance(instance, agent.MirrorAgent)
    native = instance.config.settings()
    assert native["temperature"] == 0
    assert native["max_tokens"] == 512
    assert native["disable_compaction"] is True
    assert native["extra_body"] == {"top_p": 0.9, "chat_template_kwargs": {"enable_thinking": False}}
    assert native["system_prompt"] == "Be concise."


@pytest.mark.parametrize(
    "harness_kwargs",
    [
        {"environment": {"type": "modal"}},
        {"environment": {"import_path": "example:Provider"}},
        {"agents": [{"name": "oracle"}]},
        {"agents": [{"import_path": "example:Agent"}]},
        {"agents": [{"env": {"OPENAI_API_KEY": "synthetic-key"}}]},
        {"agents": []},
    ],
)
def test_build_job_config_rejects_foreign_provider_or_agent(
    tmp_path: pathlib.Path, harness_kwargs: dict[str, object]
) -> None:
    request = base.RunRequest(
        "https://endpoint.invalid/v1", "model", tmp_path, settings=records.EvalSettings(harness_kwargs=harness_kwargs)
    )
    with pytest.raises(ValueError):
        runner.HarborRunner("coding", {"name": "example/tasks", "ref": "sha256:pinned"}).build_job_config(request)


def test_build_job_config_rejects_simulated_user_agent(tmp_path: pathlib.Path) -> None:
    request = base.RunRequest(
        "https://endpoint.invalid/v1",
        "model",
        tmp_path,
        settings=records.EvalSettings(harness_kwargs={"user_agent": {"name": "oracle", "bridge": {"kind": "acp"}}}),
    )
    with pytest.raises(ValueError, match="do not support a separate user agent"):
        runner.HarborRunner("coding", {"name": "example/tasks"}).build_job_config(request)


@pytest.mark.parametrize(
    "provider, proxy, api_key, error",
    [
        ("daytona", False, "synthetic-host-secret", "host API keys cannot enter sandboxes"),
        ("docker", False, "synthetic-host-secret", "host API keys cannot enter sandboxes"),
        ("docker", True, "synthetic-host-secret", "--host-model-proxy requires --sandbox daytona"),
        ("docker", True, "EMPTY", "--host-model-proxy requires --sandbox daytona"),
    ],
)
def test_runner_rejects_unsupported_endpoint_wiring_before_setup(
    tmp_path: pathlib.Path, provider: str, proxy: bool, api_key: str, error: str
) -> None:
    request = base.RunRequest(
        "https://endpoint.invalid/v1",
        "model",
        tmp_path,
        api_key=api_key,
        settings=records.EvalSettings(host_model_proxy=proxy, harness_kwargs={"environment": {"type": provider}}),
    )
    with pytest.raises(base.RunnerError, match=error) as raised:
        runner.HarborRunner("coding", {"name": "example/tasks"})(request)
    assert "synthetic-host-secret" not in str(raised.value)
    assert list(tmp_path.iterdir()) == []


def test_runner_refuses_existing_job_before_provider_setup(tmp_path: pathlib.Path) -> None:
    job_dir = tmp_path / "harbor_jobs" / "coding"
    job_dir.mkdir(parents=True)
    with pytest.raises(base.RunnerError, match="Refusing to reuse"):
        runner.HarborRunner("coding", {"name": "example/tasks"})(
            base.RunRequest("http://localhost/v1", "model", tmp_path)
        )


def test_build_job_config_retains_unset_native_generation_defaults(tmp_path: pathlib.Path) -> None:
    config = runner.HarborRunner("coding", {"name": "example/tasks"}).build_job_config(
        base.RunRequest("http://localhost/v1", "model", tmp_path)
    )
    native = config.agents[0].kwargs["mirror_kwargs"]
    assert config.environment.type == environment_type.EnvironmentType.DOCKER
    assert "temperature" not in native
    assert "tools" not in native
    assert "disable_compaction" not in native
    assert "max_tokens" not in native
    assert native["turn_timeout_seconds"] == 10800
    assert "OPENAI_API_KEY" not in json.dumps(config.model_dump(mode="json"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "compose_path, environment_options",
    [
        ("environment/docker-compose.yaml", {}),
        ("tests/docker-compose.yaml", {}),
        ("steps/verify/tests/docker-compose.yaml", {}),
        ("overlay.yaml", {"extra_docker_compose": ["overlay.yaml"]}),
        ("overlay.yaml", {"kwargs": {"extra_docker_compose": ["overlay.yaml"]}}),
        ("overlay.yaml", {"mounts": [{"type": "bind", "source": "/synthetic-host-home", "target": "/host"}]}),
    ],
)
async def test_validate_local_task_rejects_host_configuration(
    tmp_path: pathlib.Path, compose_path: str, environment_options: dict[str, object]
) -> None:
    event = hooks.TrialHookEvent.model_validate_json(
        (pathlib.Path(__file__).parent / "fixtures/harbor_trial_event.json").read_text()
    )
    event.result.task_id = task_id.LocalTaskId(path=tmp_path)
    (tmp_path / "task.toml").write_text('version = "1.0"\n[[steps]]\nname = "verify"\n')
    compose = tmp_path / compose_path
    compose.parent.mkdir(parents=True, exist_ok=True)
    compose.write_text('services: {main: {volumes: ["/synthetic-host-home:/host:ro"]}}\n')
    event.config.environment = trial_config.EnvironmentConfig.model_validate(environment_options)
    with pytest.raises(ValueError, match="cannot supply Compose or host mounts"):
        await worker.validate_local_task(event)


@pytest.mark.parametrize("section", ["environment", "verifier"])
def test_worker_blocks_host_credentials_in_task_environment_templates(section: str) -> None:
    code = f"""
import os
from harbor.models.task import config
from harbor.utils import env
from {worker.__package__} import worker
task = config.TaskConfig.model_validate({{
    {section!r}: {{'env': {{'PUBLIC': '${{DAYTONA_API_KEY}}'}}}}
}})
worker.isolate_task_environment(local_docker=True)
assert 'DAYTONA_API_KEY' not in os.environ
assert 'UNRELATED_HOST_CREDENTIAL' not in os.environ
assert os.environ['LANG'] == 'C.UTF-8'
assert os.environ['DOCKER_HOST'] == 'unix:///run/user/123/docker.sock'
assert os.environ['DOCKER_CONTEXT'] == 'rootless'
try:
    env.resolve_env_vars(task.{section}.env)
except ValueError as error:
    assert 'not found in host environment' in str(error)
else:
    raise AssertionError('Task received the host credential')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "DAYTONA_API_KEY": "synthetic-provider-secret",
            "UNRELATED_HOST_CREDENTIAL": "synthetic-unrelated-secret",
            "LANG": "C.UTF-8",
            "DOCKER_HOST": "unix:///run/user/123/docker.sock",
            "DOCKER_CONTEXT": "rootless",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "synthetic-provider-secret" not in result.stdout + result.stderr
    assert "synthetic-unrelated-secret" not in result.stdout + result.stderr


def test_run_in_subprocess_keeps_startup_errors_on_stderr(tmp_path: pathlib.Path) -> None:
    code = f"""
import asyncio
import pathlib
from harbor.models.job import config
from {worker.__package__} import worker
asyncio.run(worker.HarborWorker(config.JobConfig.model_validate({{
    "job_name": "missing-auth", "jobs_dir": str(pathlib.Path.cwd()), "environment": {{"type": "daytona"}}
}})).run_in_subprocess())
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={"HOME": str(tmp_path)},  # Container UIDs need not have a passwd entry.
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "Authentication credentials not found" in result.stderr
    assert "Harbor worker exited with status" in result.stderr


@pytest.mark.parametrize("mode", ["complete", "cancel", "sigint", "sigterm"])
def test_run_in_subprocess_waits_for_cleanup_and_child_exit(tmp_path: pathlib.Path, mode: str) -> None:
    script = pathlib.Path(__file__).parent / "fixtures" / "harbor_worker_lifecycle.py"
    try:
        result = subprocess.run(
            [sys.executable, str(script), mode, str(tmp_path)], capture_output=True, text=True, timeout=45
        )
        assert result.returncode == 0, result.stderr
    finally:
        # A broken supervisor must not leave its fixture child behind on timeout.
        with contextlib.suppress(FileNotFoundError, ProcessLookupError):
            os.kill(int((tmp_path / "child.pid").read_text()), signal.SIGKILL)


@pytest.mark.asyncio
async def test_log_trial_event_preserves_graded_timeout(caplog: pytest.LogCaptureFixture) -> None:
    event = hooks.TrialHookEvent.model_validate_json(
        (pathlib.Path(__file__).parent / "fixtures/harbor_trial_event.json").read_text()
    )
    event.result.exception_info = trial_result.ExceptionInfo(
        exception_type="AgentTimeoutError",
        exception_message="synthetic-sensitive-diagnostic",
        exception_traceback="",
        occurred_at=datetime.datetime(2026, 1, 1),
    )
    await worker.log_trial_event(event)
    assert "Trial example-attempt end; rewards={'reward': 1.0}; error=AgentTimeoutError" in caplog.text
    assert "synthetic-sensitive-diagnostic" not in caplog.text


@pytest.mark.asyncio
async def test_log_trial_event_reports_start_without_reward(caplog: pytest.LogCaptureFixture) -> None:
    event = hooks.TrialHookEvent.model_validate_json(
        (pathlib.Path(__file__).parent / "fixtures/harbor_trial_event.json").read_text()
    )
    event.event = hooks.TrialEvent.START
    event.result.verifier_result = None
    await worker.log_trial_event(event)
    assert "Trial example-attempt start; rewards=None; error=None" in caplog.text


def test_build_job_config_keeps_proxy_credentials_out_of_agent(tmp_path: pathlib.Path) -> None:
    request = base.RunRequest(
        "https://endpoint.invalid/v1",
        "model",
        tmp_path,
        api_key="synthetic-host-secret",
        settings=records.EvalSettings(host_model_proxy=True, sandbox="daytona"),
    )
    config = runner.HarborRunner("coding", {"name": "example/tasks"}).build_job_config(request)
    assert "synthetic-host-secret" not in config.model_dump_json()
    assert config.agents[0].env == {}


def test_resume_rejects_changed_provider_before_setup(tmp_path: pathlib.Path) -> None:
    scaffold = runner.HarborRunner("coding", {"name": "example/tasks"})
    request = base.RunRequest("http://localhost/v1", "model", tmp_path, resume=True)
    config = scaffold.build_job_config(request)
    config.environment.type = environment_type.EnvironmentType.DAYTONA
    job_dir = config.jobs_dir / config.job_name
    job_dir.mkdir(parents=True)
    (job_dir / "config.json").write_text(config.model_dump_json())
    with pytest.raises(base.RunnerError, match="unchanged job settings"):
        scaffold(request)


def test_resume_rejects_changed_proxy_mode(tmp_path: pathlib.Path) -> None:
    scaffold = runner.HarborRunner("coding", {"name": "example/tasks"})
    request = base.RunRequest(
        "http://localhost/v1", "model", tmp_path, resume=True, settings=records.EvalSettings(sandbox="daytona")
    )
    config = scaffold.build_job_config(request)
    config.agents[0].kwargs["base_url"] = "http://127.0.0.1:8900/v1"
    job_dir = config.jobs_dir / config.job_name
    job_dir.mkdir(parents=True)
    (job_dir / "config.json").write_text(config.model_dump_json())
    with pytest.raises(base.RunnerError, match="cannot change host model routing"):
        scaffold(request)


def test_resume_rejects_changed_native_settings_before_setup(tmp_path: pathlib.Path) -> None:
    scaffold = runner.HarborRunner("coding", {"name": "example/tasks"})
    request = base.RunRequest("http://localhost/v1", "model", tmp_path, resume=True)
    config = scaffold.build_job_config(request)
    job_dir = config.jobs_dir / config.job_name
    job_dir.mkdir(parents=True)
    (job_dir / "config.json").write_text(config.model_dump_json())
    request.settings.temperature = 0
    with pytest.raises(base.RunnerError, match="unchanged job settings"):
        scaffold(request)


class InProcessWorker(worker.HarborWorker):
    """Run native jobs without spawning; subprocess lifecycle is tested separately."""

    async def run_in_subprocess(self) -> None:
        native_job = await job.Job.create(self.config)
        await native_job.run()


@pytest.fixture
def completed_trials(monkeypatch: pytest.MonkeyPatch) -> list[pathlib.Path]:
    """Keep native job reconciliation and scoring; replace sandbox execution."""
    completed: list[pathlib.Path] = []
    rewards = iter([1.0, 0.0, None, 1.0])

    async def complete(config: trial_config.TrialConfig) -> trial_result.TrialResult:
        reward = next(rewards)
        directory = config.trials_dir / config.trial_name
        directory.mkdir()
        result = trial_result.TrialResult(
            task_name=config.task.path.name,
            trial_name=config.trial_name,
            trial_uri=directory.as_uri(),
            task_id=config.task.get_task_id(),
            task_checksum="fixture",
            config=config,
            agent_info=trial_result.AgentInfo(name="mirror", version="0.1.8"),
            verifier_result=verifier_result.VerifierResult(rewards={"reward": reward} if reward is not None else None),
        )
        (directory / "config.json").write_text(config.model_dump_json())
        (directory / "result.json").write_text(result.model_dump_json())
        (directory / "agent.log").write_text(f"Attempt {len(completed)}")
        completed.append(directory)
        return result

    def submit_batch(
        self: queue.TrialQueue, configs: list[trial_config.TrialConfig]
    ) -> list[Coroutine[object, object, trial_result.TrialResult]]:
        del self  # Unused.
        return [complete(config) for config in configs]

    monkeypatch.setenv("HARBOR_TELEMETRY", "0")
    monkeypatch.setattr(environment_factory.EnvironmentFactory, "run_preflight", lambda _: None)
    monkeypatch.setattr(queue.TrialQueue, "submit_batch", submit_batch)
    monkeypatch.setattr(worker, "HarborWorker", InProcessWorker)
    return completed


def test_runner_resume_retries_only_ungraded_attempts(
    tmp_path: pathlib.Path, completed_trials: list[pathlib.Path]
) -> None:
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("Return the fixture result.")
    (task / "environment" / "Dockerfile").write_text("FROM scratch\n")
    (task / "tests" / "test.sh").write_text("exit 0\n")
    request = base.RunRequest(
        "https://endpoint.invalid/v1",
        "model",
        tmp_path / "run",
        settings=records.EvalSettings(
            harness_kwargs={"datasets": [], "tasks": [{"path": task}], "n_attempts": 3, "quiet": True}
        ),
    )
    scaffold = runner.HarborRunner("coding", {"name": "example/tasks"})
    initial = scaffold(request)
    assert (initial.result.num_ok, initial.result.num_error) == (2, 1)
    assert len(completed_trials) == 3
    passed, failed, ungraded = completed_trials
    graded_files = {path: path.read_bytes() for path in itertools.chain(passed.iterdir(), failed.iterdir())}
    ungraded_files = {path.name: path.read_bytes() for path in ungraded.iterdir()}
    interrupted = ungraded.parent / "interrupted"
    interrupted.mkdir()
    (interrupted / "agent.log").write_text("interrupted evidence")

    request.resume = True
    resumed = scaffold(request)
    assert len(completed_trials) == 4
    assert resumed.result.status == "ok"
    assert (resumed.result.num_ok, resumed.result.num_error) == (3, 0)
    assert sorted(trace.score for trace in resumed.traces) == [0.0, 1.0, 1.0]
    assert resumed.result.metrics["mean"] == pytest.approx(2 / 3)
    assert {path: path.read_bytes() for path in graded_files} == graded_files
    assert not ungraded.exists()
    assert not interrupted.exists()
    archive = next(ungraded.parent.parent.glob("coding.interrupted-*"))
    assert {path.name: path.read_bytes() for path in (archive / ungraded.name).iterdir()} == ungraded_files
    assert (archive / "interrupted" / "agent.log").read_text() == "interrupted evidence"

    completed = scaffold(request)
    assert len(completed_trials) == 4
    assert completed.result == resumed.result
    assert {path: path.read_bytes() for path in graded_files} == graded_files
