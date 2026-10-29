"""Run the bundled Mirror executable as a Harbor agent."""

import asyncio
import json
import math
import os
import pathlib
import shlex
import ssl
import tempfile
from collections.abc import Sequence
from importlib import metadata, resources
from typing import Any, final

import httpx
from harbor.agents.installed import base as installed
from harbor.environments import base as environments
from harbor.models.agent import context as agent_context
from harbor.models.trajectories import agent as atif_agent
from harbor.models.trajectories import final_metrics as atif_final_metrics
from harbor.models.trajectories import metrics as atif_metrics
from harbor.models.trajectories import observation as atif_observation
from harbor.models.trajectories import observation_result as atif_observation_result
from harbor.models.trajectories import step as atif_step
from harbor.models.trajectories import tool_call as atif_tool_call
from harbor.models.trajectories import trajectory as atif_trajectory
from harbor.utils import trajectory_utils

import beam_verifier
from beam_verifier.benchmarks import prompts
from beam_verifier.benchmarks.scaffolds.harbor.mirror import config as mirror_config
from beam_verifier.benchmarks.scaffolds.harbor.mirror import trajectory as mirror_trajectory

STAGED_BINARY_PATH = "/installed-agent/mirror-agent"
STAGED_RUNTIME_PATH = "/installed-agent/runtime"
MIRROR_COMMAND = (f"{STAGED_RUNTIME_PATH}/ld.so", "--library-path", STAGED_RUNTIME_PATH, STAGED_BINARY_PATH)
STAGED_CA_BUNDLE_PATH = "/installed-agent/ca-certificates.crt"
SERVER_ADDRESS = "127.0.0.1:4097"
JOURNAL_FILENAME = "mirror.jsonl"
INSTRUCTION_FILENAME = "instruction.txt"
READY_TIMEOUT_SECONDS = 300
RESPONSE_GRACE_SECONDS = 10


@final
class MirrorAgent(installed.BaseInstalledAgent):
    """Harbor adapter that installs and runs Mirror inside the task sandbox."""

    SUPPORTS_ATIF: bool = True

    def __init__(
        self,
        logs_dir: pathlib.Path,
        model_name: str | None = None,
        *,
        base_url: str | None = None,
        mirror_kwargs: dict[str, object] | None = None,
        prompt_factory: dict[str, str | None] | None = None,
        extra_env: dict[str, str] | None = None,
        **harbor_kwargs: Any,
    ) -> None:
        if extra_env:
            raise ValueError("Mirror must not receive host credentials through agent environment overrides")
        super().__init__(logs_dir, model_name=model_name, **harbor_kwargs)
        if model_name is None:
            raise ValueError("Mirror needs a model name")
        if not base_url:
            raise ValueError("Mirror needs a model endpoint: pass base_url")
        self.config = mirror_config.MirrorConfig(model_name, base_url, dict(mirror_kwargs or {}))
        self.prompt_factory: prompts.BeamPromptFactory | None = None
        if prompt_factory is not None:
            name = prompt_factory["name"]
            if name is None:
                raise ValueError("The saved Beam prompt factory needs a name")
            self.prompt_factory = prompts.BeamPromptFactory(
                name=name,
                user_template=prompt_factory.get("user_template"),
                system_text=prompt_factory.get("system_text"),
            )
        if self.prompt_factory is not None and self.prompt_factory.system_text is not None:
            self.config.mirror_kwargs.setdefault("system_prompt", self.prompt_factory.system_text)

    @staticmethod
    def name() -> str:
        return "mirror"

    def version(self) -> str:
        """Return the bundled Mirror agent's version."""
        return "0.1.8"

    async def install(self, environment: environments.BaseEnvironment) -> None:
        """Install and check Mirror, its runtime, and public CA certificates.

        Mirror uses the bundled runtime. Task commands keep using the sandbox
        image's system libraries, including in musl-based images.

        Args:
            environment: The task sandbox.

        Raises:
            RuntimeError: If the host has no trusted CA certificates to copy.
        """
        machine = await self.exec_as_agent(environment, command="uname -m", timeout_sec=30)
        binary = bundled_binary((machine.stdout or "").strip())
        await environment.upload_file(binary, STAGED_BINARY_PATH)
        await environment.upload_dir(binary.parent / "runtime", STAGED_RUNTIME_PATH)
        # Some task images have no trust store. Copy public trust anchors, never
        # the host's authentication material, and support non-Linux build hosts.
        certificates = httpx.create_ssl_context().get_ca_certs(binary_form=True)
        if not certificates:
            raise RuntimeError("No CA certificates available to stage into the sandbox")
        with tempfile.TemporaryDirectory(prefix="mirror-ca-") as temporary:
            bundle = pathlib.Path(temporary) / "ca-certificates.crt"
            bundle.write_text("".join(ssl.DER_cert_to_PEM_cert(cert) for cert in certificates))
            await environment.upload_file(bundle, STAGED_CA_BUNDLE_PATH)
        await self.exec_as_root(
            environment,
            command=(
                f"chmod 755 {STAGED_BINARY_PATH} {STAGED_RUNTIME_PATH}/ld.so && chmod 644 {STAGED_CA_BUNDLE_PATH}"
            ),
        )
        await self.exec_as_agent(environment, command=shlex.join([*MIRROR_COMMAND, "--version"]), timeout_sec=30)
        await self.exec_as_agent(environment, command=dev_tcp_check_command())

    @installed.with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: environments.BaseEnvironment,
        context: agent_context.AgentContext,
    ) -> None:
        """Run Mirror on the task instruction and clean up before grading.

        Args:
            instruction: The task instruction.
            environment: The task sandbox.
            context: Harbor's agent result, filled later by `populate_context_post_run`.

        Raises:
            RuntimeError: If the workspace lookup, process snapshot, or cleanup fails.
        """
        del context  # Unused.
        if self.prompt_factory is not None:
            workdir = environment.task_env_config.workdir
            if self.prompt_factory.requires_workdir and not workdir:
                result = await self.exec_as_agent(
                    environment,
                    command=shlex.join(["bash", "-c", 'cd -- "$1" && pwd', "bash", self.config.workspace]),
                    timeout_sec=30,
                )
                workdir = (result.stdout or "").strip()
                if result.return_code != 0 or not workdir.startswith("/") or "\n" in workdir:
                    raise RuntimeError("Could not resolve Mirror's workspace for the Beam prompt")
            instruction = self.prompt_factory.render_task_instruction(instruction, workdir=workdir)
        # Allow the stop script's SIGTERM grace and a scan of the sandbox's processes.
        cleanup_timeout_seconds = 180
        with tempfile.TemporaryDirectory(prefix="mirror-instruction-") as staging:
            source = pathlib.Path(staging) / INSTRUCTION_FILENAME
            source.write_bytes(instruction.encode("utf-8"))
            source.chmod(0o644)
            await environment.upload_file(source, str(self.environment_logs_dir / INSTRUCTION_FILENAME))
        # Keep script contents and the baseline on the host: the agent can
        # rewrite sandbox files, including files it would later execute as root.
        process_script = resources.files(__package__).joinpath("processes.sh").read_text(encoding="utf-8")
        snapshot = await self.exec_as_root(
            environment,
            command=shlex.join(["bash", "-c", process_script, "processes.sh", "snapshot", "/proc"]),
            timeout_sec=cleanup_timeout_seconds,
        )
        baseline = (snapshot.stdout or "").splitlines()
        if not baseline:
            # No real snapshot is empty (PID 1 always runs). Without one, nothing
            # the turn starts could be told from the task's own processes.
            raise RuntimeError("the process snapshot before the Mirror turn was empty")
        succeeded = False
        try:
            await self.exec_as_agent(
                environment,
                command=shlex.join(
                    [
                        "bash",
                        "-c",
                        resources.files(__package__).joinpath("turn.sh").read_text(encoding="utf-8"),
                        "turn.sh",
                        str(self.environment_logs_dir / INSTRUCTION_FILENAME),
                        str(self.environment_logs_dir / "server.log"),
                        SERVER_ADDRESS,
                        str(READY_TIMEOUT_SECONDS),
                        str(self.config.turn_timeout_seconds + RESPONSE_GRACE_SECONDS),
                        *server_command(self.config, self.environment_logs_dir),
                    ]
                ),
                cwd=self.config.workspace if self.config.workspace.startswith("/") else None,
                env={
                    "OPENAI_API_KEY": "EMPTY",
                    "SSL_CERT_FILE": STAGED_CA_BUNDLE_PATH,
                },
                # Let Mirror report its own timeout before the sandbox exec is cut off.
                timeout_sec=math.ceil(READY_TIMEOUT_SECONDS + self.config.turn_timeout_seconds + RESPONSE_GRACE_SECONDS)
                + 120,
            )
            succeeded = True
        finally:
            cleanup = asyncio.create_task(self.cleanup_turn(environment, baseline, succeeded=succeeded))
            cancelled = False
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cancelled = True
            cleanup.result()
            if cancelled:
                raise asyncio.CancelledError

    async def cleanup_turn(
        self, environment: environments.BaseEnvironment, baseline: Sequence[str], *, succeeded: bool
    ) -> None:
        """Stop processes from failed runs and clear verifier files before grading.

        Runs that complete without error may leave services needed by the task;
        this does not mean the task passed. On failure, stop processes started
        during the run. Harbor skips grading if cleanup fails.
        """
        try:
            if not succeeded:
                process_script = resources.files(__package__).joinpath("processes.sh").read_text(encoding="utf-8")
                await self.exec_as_root(
                    environment,
                    command=shlex.join(["bash", "-c", process_script, "processes.sh", "stop", "/proc", *baseline]),
                    timeout_sec=180,
                )
            await self.exec_as_root(
                environment,
                command=shlex.join(
                    ["find", str(self.environment_logs_dir.parent / "verifier"), "-mindepth", "1", "-delete"]
                ),
                timeout_sec=180,
            )
        except (RuntimeError, OSError) as error:
            # Harbor grades NonZeroAgentExitCodeError, but skips grading for this
            # ordinary RuntimeError: an unsafe workspace cannot be scored.
            raise RuntimeError("could not clean up the Mirror turn before grading") from error

    def populate_context_post_run(self, context: agent_context.AgentContext) -> None:
        """Record Mirror's conversation and token counts after Harbor copies logs.

        Log file and parsing errors are logged without replacing the trial result.
        """
        try:
            self._record_journal(context)
        except (OSError, ValueError):
            self.logger.exception("Could not convert the Mirror journal to ATIF")

    def _record_journal(self, context: agent_context.AgentContext) -> None:
        """Save Mirror's conversation in Harbor's ATIF format and record token totals.

        Logs come from the sandbox, where the agent can modify files. Remove
        any existing trajectory file before writing, so a link cannot redirect
        the write.
        """
        trajectory_path = self.logs_dir / "trajectory.json"
        trajectory_path.unlink(missing_ok=True)
        journal_path = self.logs_dir / JOURNAL_FILENAME
        if not journal_path.is_file() or journal_path.is_symlink():
            self.logger.warning("Mirror wrote no readable journal at %s", journal_path)
            return
        with journal_path.open(encoding="utf-8", errors="replace") as journal_file:
            parsed = mirror_trajectory.parse(journal_file)
        if parsed.unreadable_lines:
            self.logger.warning("%d unreadable line(s) in %s", parsed.unreadable_lines, journal_path)
        usage = parsed.usage
        context.n_input_tokens = usage.prompt_tokens
        context.n_cache_tokens = usage.cached_tokens
        context.n_output_tokens = usage.completion_tokens
        steps = atif_steps(parsed)
        if not steps:
            # ATIF requires at least one step.
            self.logger.warning("Mirror journal at %s held no turns", journal_path)
            return
        trajectory = atif_trajectory.Trajectory(
            session_id=self.session_id,
            agent=atif_agent.Agent(
                name=self.name(),
                version=self.version(),
                model_name=self.model_name,
                extra=self.config.settings(),
            ),
            steps=steps,
            final_metrics=atif_final_metrics.FinalMetrics(
                total_prompt_tokens=usage.prompt_tokens,
                total_completion_tokens=usage.completion_tokens,
                total_cached_tokens=usage.cached_tokens,
                total_steps=len(steps),
            ),
        )
        descriptor = os.open(trajectory_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as trajectory_file:
            trajectory_file.write(trajectory_utils.format_trajectory_json(trajectory.to_json_dict()))


def server_command(config: mirror_config.MirrorConfig, logs_dir: pathlib.PurePosixPath) -> list[str]:
    """Build the command that starts Mirror's HTTP server inside the sandbox.

    Args:
        config: Model endpoint and Mirror settings.
        logs_dir: Sandbox directory for Mirror's journal.

    Returns:
        The bundled runtime command, Mirror binary, and `serve` arguments.
    """
    flags: list[str] = []
    for name, value in config.settings().items():
        flag = "--" + name.replace("_", "-")
        if value is True:
            flags.append(flag)
        elif value is not None and value is not False:
            values = value if isinstance(value, list) else [value]
            for item in values:
                encoded = json.dumps(item) if isinstance(item, dict) else str(item)
                flags.append(f"{flag}={encoded}")
    return [
        *MIRROR_COMMAND,
        "serve",
        "--bind",
        SERVER_ADDRESS,
        "--model",
        config.model,
        "--base-url",
        config.base_url,
        "--output",
        f"{logs_dir}/{JOURNAL_FILENAME}",
        *flags,
    ]


def dev_tcp_check_command() -> str:
    """Build a command that checks Bash's TCP connection support.

    A refused connection passes this check. An error saying `/dev/tcp` is
    missing or unsupported fails it.
    """
    probe = shlex.join(["bash", "-c", "exec 3<>/dev/tcp/127.0.0.1/1"])
    # Bash without network redirections treats `/dev/tcp/...` as an ordinary path.
    patterns = "|".join(f"*{shlex.quote(error)}*" for error in ("No such file or directory", "not supported"))
    return (
        f"out=$(LC_ALL=C {probe} 2>&1 || true); "
        f'case "$out" in {patterns}) '
        "echo 'bash has no /dev/tcp support; Mirror cannot be reached' >&2; "
        "exit 1;; esac"
    )


def atif_steps(journal: mirror_trajectory.Journal) -> list[atif_step.Step]:
    """Convert journal messages into Harbor's ATIF conversation steps.

    If Mirror summarizes its context, include the rebuilt prompt at that point.
    Mark those messages as copied context so they do not count as new model
    calls. Record the reset in the first copied step's `context_reset` field.

    Args:
        journal: The parsed journal from one Mirror run.

    Returns:
        Conversation steps in journal order.
    """
    steps: list[atif_step.Step] = []
    compactions = iter(journal.compactions)
    pending = next(compactions, None)
    for index, turn in enumerate(journal.turns):
        while pending is not None and pending.turn_index == index:
            steps.extend(_compaction_steps(pending, journal.turns, len(steps)))
            pending = next(compactions, None)
        steps.append(_step(turn, step_id=len(steps) + 1))
    while pending is not None:
        steps.extend(_compaction_steps(pending, journal.turns, len(steps)))
        pending = next(compactions, None)
    return steps


def _step(
    turn: mirror_trajectory.Turn,
    *,
    step_id: int,
    copied: bool = False,
    extra: dict[str, object] | None = None,
) -> atif_step.Step:
    observation = None
    if turn.tool_results:
        call_ids = {call.call_id for call in turn.tool_calls}
        observation = atif_observation.Observation(
            results=[_observation_result(result, call_ids) for result in turn.tool_results]
        )
    metrics = None
    if turn.source == "agent" and turn.usage is not None and not copied:
        metrics = atif_metrics.Metrics(
            prompt_tokens=turn.usage.prompt_tokens,
            completion_tokens=turn.usage.completion_tokens,
            cached_tokens=turn.usage.cached_tokens,
        )
    return atif_step.Step(
        step_id=step_id,
        source=turn.source,
        message=turn.message,
        reasoning_content=turn.reasoning_content,
        tool_calls=[
            atif_tool_call.ToolCall(
                tool_call_id=call.call_id,
                function_name=call.name,
                arguments=dict(call.arguments),
            )
            for call in turn.tool_calls
        ]
        or None,
        observation=observation,
        metrics=metrics,
        is_copied_context=True if copied else None,
        extra=extra,
    )


def _compaction_steps(
    compaction: mirror_trajectory.Compaction,
    turns: Sequence[mirror_trajectory.Turn],
    steps_so_far: int,
) -> list[atif_step.Step]:
    """Convert the rebuilt prompt to copied steps, marking the context reset."""
    reset: dict[str, object] = {
        "reason": "compaction_overflow" if compaction.trigger else "compaction_threshold",
        "summary": compaction.summary,
    }
    if compaction.trigger:
        reset["detail"] = compaction.trigger
    if (tokens := _context_tokens(turns[: compaction.turn_index])) is not None:
        reset["tokens_before"] = tokens
    # A compaction that kept nothing still closed an epoch; an empty system step
    # marks where.
    prompt = compaction.prompt or (mirror_trajectory.Turn(source="system", message=""),)
    return [
        _step(
            turn,
            step_id=steps_so_far + offset + 1,
            copied=True,
            extra={"context_reset": reset} if offset == 0 else None,
        )
        for offset, turn in enumerate(prompt)
    ]


def _context_tokens(turns: Sequence[mirror_trajectory.Turn]) -> int | None:
    """Return input plus output tokens from the last call with an input count."""
    for turn in reversed(turns):
        if turn.usage is not None and turn.usage.prompt_tokens:
            return turn.usage.prompt_tokens + turn.usage.completion_tokens
    return None


def _observation_result(
    result: mirror_trajectory.ToolResult, call_ids: set[str]
) -> atif_observation_result.ObservationResult:
    """Convert a tool result, linking its call only if it belongs to this step.

    Tool results attach to the latest assistant message, which may not contain
    their call. Preserve unmatched IDs as metadata because ATIF rejects links
    to calls outside the step.
    """
    if result.call_id in call_ids:
        return atif_observation_result.ObservationResult(source_call_id=result.call_id, content=result.content)
    return atif_observation_result.ObservationResult(
        content=result.content,
        extra={"unmatched_call_id": result.call_id} if result.call_id else None,
    )


def bundled_binary(architecture: str) -> pathlib.Path:
    """Find the installed Mirror binary for the sandbox's architecture.

    Args:
        architecture: Sandbox architecture from `uname -m`: `x86_64` or `aarch64`.

    Returns:
        Binary path on the host, including with an editable installation.

    Raises:
        ValueError: If the sandbox architecture has no bundled executable.
        FileNotFoundError: If the installed package is missing the binary.
    """
    if architecture not in {"x86_64", "aarch64"}:
        raise ValueError(f"Unsupported Mirror sandbox architecture: {architecture!r}")
    package = beam_verifier.__name__.replace(".", "/")
    binary = pathlib.Path(
        str(
            metadata.distribution("beam-verifier").locate_file(
                f"{package}/bin/mirror-agent/linux-{architecture}/mirror-agent"
            )
        )
    )
    if not binary.is_file():
        raise FileNotFoundError(f"Missing bundled Mirror executable: {binary}")
    return binary
