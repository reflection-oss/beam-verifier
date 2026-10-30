"""Test the Mirror adapter's commands, cleanup, and log parsing without sandboxes."""

import asyncio
import logging
import pathlib
import shlex

import pytest
from harbor.agents.installed import base as installed
from harbor.environments import base as environment_base
from harbor.environments.docker import docker
from harbor.models.agent import context as agent_context
from harbor.models.task import config as task_config
from harbor.models.trial import paths

from beam_verifier.benchmarks import prompts
from beam_verifier.benchmarks.scaffolds.harbor.mirror import agent as mirror_agent
from beam_verifier.benchmarks.scaffolds.harbor.mirror import config as mirror_config


@pytest.fixture
def mirror(tmp_path: pathlib.Path) -> mirror_agent.MirrorAgent:
    return mirror_agent.MirrorAgent(
        tmp_path,
        "model-under-test",
        base_url="http://127.0.0.1:1/v1",
        extra_env={},
        logger=logging.getLogger("harbor"),
    )


class RecordingEnvironment(docker.DockerEnvironment):
    """Records sandbox commands without executing them or creating a container."""

    def __init__(self, directory: pathlib.Path) -> None:
        (directory / "Dockerfile").write_text("FROM scratch\n")
        super().__init__(
            environment_dir=directory,
            environment_name="mirror-test",
            session_id="mirror-test__env",
            trial_paths=paths.TrialPaths(trial_dir=directory / "trial"),
            task_env_config=task_config.EnvironmentConfig(),
        )
        self.commands: list[list[str]] = []
        self.environments: list[dict[str, str]] = []
        self.payloads: dict[str, bytes] = {}
        self.timeouts: list[int | None] = []
        self.uploads: list[tuple[pathlib.Path, str]] = []
        self.uploaded_dirs: list[tuple[pathlib.Path, str]] = []
        self.results: list[environment_base.ExecResult] = []
        self.block_on_turn = False
        self.blocked = asyncio.Event()

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> environment_base.ExecResult:
        del cwd, user  # Unused.
        self.environments.append(dict(env or {}))
        arguments = shlex.split(command.removeprefix("set -o pipefail; "))
        self.commands.append(arguments)
        self.timeouts.append(timeout_sec)
        if self.block_on_turn and "turn.sh" in arguments:
            self.blocked.set()
            await asyncio.Future[None]()
        return self.results.pop(0) if self.results else environment_base.ExecResult(return_code=0, stdout="1 0\n")

    async def upload_file(self, source_path: pathlib.Path | str, target_path: str) -> None:
        self.uploads.append((pathlib.Path(source_path), target_path))
        self.payloads[target_path] = pathlib.Path(source_path).read_bytes()

    async def upload_dir(self, source_dir: pathlib.Path | str, target_dir: str) -> None:
        self.uploaded_dirs.append((pathlib.Path(source_dir), target_dir))


@pytest.fixture
def environment(tmp_path: pathlib.Path) -> RecordingEnvironment:
    return RecordingEnvironment(tmp_path)


def test_an_unreadable_journal_leaves_the_trial_its_own_outcome(
    tmp_path: pathlib.Path, mirror: mirror_agent.MirrorAgent
) -> None:
    mirror.populate_context_post_run(agent_context.AgentContext())
    assert not (tmp_path / "trajectory.json").exists()


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan"), float("inf")])
def test_mirror_options_reject_invalid_deadlines(timeout: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        mirror_config.MirrorConfig(
            model="model", base_url="http://unused/v1", mirror_kwargs={"turn_timeout_seconds": timeout}
        )


def test_mirror_options_preserve_explicit_deadline() -> None:
    config = mirror_config.MirrorConfig(
        model="model", base_url="http://unused/v1", mirror_kwargs={"turn_timeout_seconds": 14400}
    )
    assert config.turn_timeout_seconds == 14400
    assert config.settings()["turn_timeout_seconds"] == 14400


@pytest.mark.parametrize("name", ["model", "base_url", "bind", "output"])
def test_mirror_options_cannot_replace_managed_endpoint_or_artifacts(name: str) -> None:
    with pytest.raises(ValueError, match="managed by beam-verifier"):
        mirror_config.MirrorConfig(model="model", base_url="http://unused/v1", mirror_kwargs={name: "override"})


@pytest.mark.parametrize("name", ["", "--output", "tools=value"])
def test_mirror_options_reject_invalid_flag_names(name: str) -> None:
    with pytest.raises(ValueError, match="invalid Mirror option name"):
        mirror_config.MirrorConfig(model="model", base_url="http://unused/v1", mirror_kwargs={name: "value"})


@pytest.mark.asyncio
async def test_run_nonzero_stops_processes_before_clearing_verifier_logs(
    mirror: mirror_agent.MirrorAgent, environment: RecordingEnvironment
) -> None:
    environment.results = [
        environment_base.ExecResult(return_code=0, stdout="1 0\n"),
        environment_base.ExecResult(return_code=1, stderr="Mirror turn failed"),
    ]

    with pytest.raises(installed.NonZeroAgentExitCodeError):
        await mirror.run("Run the task.", environment, agent_context.AgentContext())

    assert len(environment.commands) == 4
    assert environment.commands[-2][3:] == ["processes.sh", "stop", "/proc", "1 0"]
    assert environment.commands[-1] == [
        "find",
        "/logs/verifier",
        "-mindepth",
        "1",
        "-delete",
    ]


@pytest.mark.asyncio
async def test_run_cancellation_stops_processes_before_clearing_verifier_logs(
    mirror: mirror_agent.MirrorAgent, environment: RecordingEnvironment
) -> None:
    environment.block_on_turn = True
    task = asyncio.create_task(mirror.run("Run the task.", environment, agent_context.AgentContext()))
    try:
        await asyncio.wait_for(environment.blocked.wait(), timeout=5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert len(environment.commands) == 4
    assert environment.commands[-2][3:] == ["processes.sh", "stop", "/proc", "1 0"]
    assert environment.commands[-1] == [
        "find",
        "/logs/verifier",
        "-mindepth",
        "1",
        "-delete",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("turn_return_code", [0, 1], ids=["verifier-log-cleanup", "process-cleanup"])
async def test_run_cleanup_failure_prevents_harbor_from_grading(
    mirror: mirror_agent.MirrorAgent, environment: RecordingEnvironment, turn_return_code: int
) -> None:
    environment.results = [
        environment_base.ExecResult(return_code=0, stdout="1 0\n"),
        environment_base.ExecResult(return_code=turn_return_code, stderr="turn outcome"),
        environment_base.ExecResult(return_code=1, stderr="cleanup failed"),
    ]

    with pytest.raises(RuntimeError, match="could not clean up the Mirror turn before grading") as raised:
        await mirror.run("Run the task.", environment, agent_context.AgentContext())

    # Harbor grades NonZeroAgentExitCodeError and AgentTimeoutError, but stops
    # verification for other exceptions. Cleanup failures must stay outside them.
    assert type(raised.value) is RuntimeError
    assert len(environment.commands) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("architecture", ["x86_64", "aarch64"])
async def test_install_selects_sandbox_architecture(
    mirror: mirror_agent.MirrorAgent, environment: RecordingEnvironment, architecture: str
) -> None:
    environment.results = [environment_base.ExecResult(return_code=0, stdout=architecture + "\n")]
    await mirror.install(environment)
    binary, destination = environment.uploads[0]
    assert binary.parts[-3:] == ("mirror-agent", f"linux-{architecture}", "mirror-agent")
    assert destination == "/installed-agent/mirror-agent"
    assert environment.uploaded_dirs == [(binary.parent / "runtime", "/installed-agent/runtime")]
    assert [
        "/installed-agent/runtime/ld.so",
        "--library-path",
        "/installed-agent/runtime",
        "/installed-agent/mirror-agent",
        "--version",
    ] in environment.commands
    assert b"BEGIN CERTIFICATE" in environment.payloads[mirror_agent.STAGED_CA_BUNDLE_PATH]


def test_bundled_binary_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="Unsupported Mirror sandbox architecture"):
        mirror_agent.bundled_binary("unsupported")


@pytest.mark.asyncio
async def test_run_uploads_instruction_as_data_and_passes_no_host_key(
    mirror: mirror_agent.MirrorAgent, environment: RecordingEnvironment
) -> None:
    instruction = "Text: $(touch forbidden) `echo code` ' \" {prompt}\n" + "long instruction " * 10000
    await mirror.run(instruction, environment, agent_context.AgentContext())
    assert environment.payloads["/logs/agent/instruction.txt"] == instruction.encode()
    assert "touch forbidden" not in repr(environment.commands)
    assert environment.environments[1] == {
        "OPENAI_API_KEY": "EMPTY",
        "SSL_CERT_FILE": mirror_agent.STAGED_CA_BUNDLE_PATH,
    }
    assert len(environment.commands) == 3  # Successful tasks can leave intended services running.


def test_agent_rejects_environment_credentials(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="must not receive host credentials"):
        mirror_agent.MirrorAgent(
            tmp_path, "model", base_url="http://endpoint/v1", extra_env={"OPENAI_API_KEY": "synthetic-secret"}
        )


@pytest.mark.asyncio
async def test_run_passes_script_paths_and_server_options_as_arguments(
    tmp_path: pathlib.Path, environment: RecordingEnvironment
) -> None:
    logs = pathlib.PurePosixPath("/logs/agent ' $(literal)")
    agent = mirror_agent.MirrorAgent(
        tmp_path,
        "model",
        base_url="http://endpoint/v1",
        mirror_kwargs={
            "system_prompt": "Keep $variables 'literal'.",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "disable_compaction": True,
            "temperature": 0,
        },
        environment_logs_dir=logs,
    )
    await agent.run("Fix the task.", environment, agent_context.AgentContext())
    command = next(command for command in environment.commands if "turn.sh" in command)
    assert command[:2] == ["bash", "-c"]
    assert command[3:9] == [
        "turn.sh",
        str(logs / "instruction.txt"),
        str(logs / "server.log"),
        mirror_agent.SERVER_ADDRESS,
        "300",
        str(agent.config.turn_timeout_seconds + 10),
    ]
    assert command[9:22] == [
        "/installed-agent/runtime/ld.so",
        "--library-path",
        "/installed-agent/runtime",
        "/installed-agent/mirror-agent",
        "serve",
        "--bind",
        "127.0.0.1:4097",
        "--model",
        "model",
        "--base-url",
        "http://endpoint/v1",
        "--output",
        str(logs / "mirror.jsonl"),
    ]
    assert "--system-prompt=Keep $variables 'literal'." in command[22:]
    assert '--extra-body={"chat_template_kwargs": {"enable_thinking": false}}' in command[22:]
    assert "--disable-compaction" in command[22:]
    assert "--temperature=0" in command[22:]
    assert str(logs) not in command[2]


@pytest.mark.asyncio
async def test_run_uses_task_workdir_in_instruction(
    mirror: mirror_agent.MirrorAgent, environment: RecordingEnvironment
) -> None:
    environment.task_env_config.workdir = "/app"
    mirror.prompt_factory = prompts.BeamPromptFactory("swe", "Fix in {workdir}: {prompt}")
    await mirror.run("Fix {name}", environment, agent_context.AgentContext())
    assert environment.payloads["/logs/agent/instruction.txt"] == b"Fix in /app: Fix {name}"


@pytest.mark.asyncio
async def test_run_resolves_sandbox_workdir_in_instruction(
    mirror: mirror_agent.MirrorAgent, environment: RecordingEnvironment
) -> None:
    environment.results = [environment_base.ExecResult(stdout="/app\n", return_code=0)]
    mirror.prompt_factory = prompts.BeamPromptFactory("swe", "Fix in {workdir}: {prompt}")
    await mirror.run("Issue", environment, agent_context.AgentContext())
    assert environment.payloads["/logs/agent/instruction.txt"] == b"Fix in /app: Issue"
