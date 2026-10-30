"""Connect agents in Daytona sandboxes to the host-side proxy."""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import secrets
import shlex
import sys
from collections.abc import AsyncIterator, Awaitable, Coroutine, Iterator
from typing import final

import asyncssh
import daytona
import websockets.exceptions
from harbor import job
from harbor.environments.daytona import environment
from harbor.models.task import config as task_config
from harbor.models.task import id as task_id
from harbor.trial import hooks
from harbor.utils import logger

from beam_verifier.benchmarks.proxies import endpoint_proxy, tunnel

_WEBSOCKET_PORT = 8901
_CONNECTOR_LOG_PATH = "/logs/agent/beam-verifier-tunnel.log"

_RECONNECT_BACKOFF_S = 2.0
_MAX_RECONNECT_ATTEMPTS = 25
_SANDBOX_LOOKUP_ATTEMPTS = 15


@contextlib.contextmanager
def _tunnel_stage(sandbox: daytona.AsyncSandbox, trial_name: str | None, stage: str) -> Iterator[None]:
    """Records tunnel failures with their time, trial, sandbox and operation.

    Harbor saves this context with the error. The job log keeps the original
    failure even if cleanup raises another error.
    """
    try:
        yield
    except Exception as error:
        context = (
            f"Model tunnel failure: trial={trial_name!r}; sandbox={sandbox.id!r}; "
            f"stage={stage}; occurred_at={datetime.datetime.now(datetime.UTC).isoformat()}"
        )
        error.add_note(context)
        logger.logger.exception("%s", context)
        raise


async def _wait_first(*awaitables: Awaitable[object]) -> None:
    """Waits for one operation to finish, then cancels the rest."""
    tasks = [asyncio.ensure_future(a) for a in awaitables]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@contextlib.asynccontextmanager
async def supervise_tunnel(
    operation: Coroutine[object, object, None], ready: asyncio.Event, stopped: asyncio.Event
) -> AsyncIterator[None]:
    """Stops the owning trial if its tunnel fails after becoming ready.

    Enter and exit in the trial's task so TaskGroup can interrupt the agent.
    The supervisor closes the connector before this context finishes.

    Args:
        operation: Tunnel supervisor, which sets `ready` and observes `stopped`.
        ready: Set when model forwarding is available.
        stopped: Signals that the agent has finished.

    Yields:
        Control once the first tunnel is ready.

    Raises:
        RuntimeError: If the tunnel fails without an existing agent error.
    """
    agent_error: BaseException | None = None
    owner = asyncio.current_task()
    cancellations = owner.cancelling() if owner is not None else 0
    tunnel_cancelled_agent = False
    try:
        async with asyncio.TaskGroup() as tasks:
            supervisor = tasks.create_task(operation)
            try:
                await ready.wait()
                yield
            except BaseException as error:
                agent_error = error
                # TaskGroup cancels its owner on supervisor failure. A cancellation
                # already in flight before cleanup must keep its original outcome.
                tunnel_cancelled_agent = (
                    isinstance(error, asyncio.CancelledError)
                    and supervisor.done()
                    and not supervisor.cancelled()
                    and supervisor.exception() is not None
                    and owner is not None
                    and owner.cancelling() == cancellations + 1
                )
                if not ready.is_set():
                    supervisor.cancel()
            finally:
                stopped.set()
    except ExceptionGroup as error:
        if agent_error is None or tunnel_cancelled_agent:
            raise RuntimeError("Model tunnel failed") from error
        agent_error.add_note("The model tunnel also failed during cleanup; see the job log.")
    if agent_error is not None:
        raise agent_error


_READY_PROBE = (
    f"for ((i = 0; i < 600; i++)); do (exec 3<>/dev/tcp/127.0.0.1/{_WEBSOCKET_PORT}) 2>/dev/null && exit 0; "
    "sleep 0.2; done; exit 1"
)


@final
class DaytonaTunnels:
    """Manages tunnels from Daytona sandboxes to the host-side proxy.

    Harbor hooks validate tasks, connect after agent setup, and close tunnels
    before grading. Job cleanup closes any remaining tunnels.
    """

    async def validate_task(self, event: hooks.TrialHookEvent) -> None:
        """Checks that the task supports a tunnel before creating its sandbox.

        Raises:
            ValueError: If the task uses Docker Compose, a nonempty `steps` list,
                a non-Linux OS, or an agent user other than root.
        """
        identity = event.result.task_id
        if isinstance(identity, task_id.PackageTaskId):
            identity = identity.model_copy(update={"ref": event.lock.task.digest})
        task_dir = identity.get_local_path()
        definition = task_config.TaskConfig.model_validate_toml((task_dir / "task.toml").read_text())
        if (task_dir / "environment/docker-compose.yaml").exists() or event.config.environment.extra_docker_compose:
            raise ValueError("Host model routing does not yet support Daytona Compose tasks; omit --host-model-proxy")
        if definition.environment.os != task_config.TaskOS.LINUX or definition.steps:
            raise ValueError("Host model routing currently requires a single-step Linux task; omit --host-model-proxy")
        if definition.agent.user not in (None, "root", 0):
            raise ValueError("Host model routing does not yet support a custom agent user; omit --host-model-proxy")

    def trial_sandbox(self, matches: list[daytona.AsyncSandbox], trial_name: str) -> daytona.AsyncSandbox:
        """Selects the only running sandbox from the trial's matches.

        A creation timeout can leave duplicate sandboxes with the same trial
        label. Refuse duplicates to avoid connecting the wrong sandbox.

        Raises:
            RuntimeError: If there is not exactly one running match.
        """

        started = [sandbox for sandbox in matches if sandbox.state == daytona.SandboxState.STARTED]
        if len(started) != 1:
            raise RuntimeError(f"Expected one started Daytona sandbox for {trial_name!r}; found {len(started)}")
        return started[0]

    @contextlib.asynccontextmanager
    async def _connector_session(
        self, sandbox: daytona.AsyncSandbox, *, log_path: str, trial_name: str | None = None
    ) -> AsyncIterator[tuple[asyncssh.SSHKey, str]]:
        """Installs the sandbox connector and removes it on exit.

        The connector accepts new connections without needing to restart.

        Yields:
            The connector's SSH host key and connection password, once ready.

        Raises:
            ValueError: If the sandbox is public.
            RuntimeError: If the connector cannot be installed or started.
            daytona.DaytonaError: If a Daytona operation fails.
        """

        if sandbox.public:
            raise ValueError("Host model routing requires a private Daytona sandbox")
        host_key = asyncssh.generate_private_key("ssh-ed25519")
        password = secrets.token_urlsafe(32)
        session_name = f"beam-verifier-tunnel-{secrets.token_hex(16)}"
        connector_dir = f"/tmp/{session_name}"
        connector_path = f"{connector_dir}/tunnel-connector"
        config_path = f"{connector_dir}/config.json"
        try:
            with _tunnel_stage(sandbox, trial_name, "connector startup"):
                # Private directory before any authentication material is uploaded.
                created = await sandbox.process.exec(f"mkdir -m 700 {connector_dir}", timeout=10)
                if created.exit_code:
                    raise RuntimeError("Could not create the private model tunnel directory")
                machine = await sandbox.process.exec("uname -m", timeout=10)
                await sandbox.fs.upload_file(
                    str(tunnel.binary_path(machine.result.strip())), connector_path, timeout=30
                )
                await sandbox.fs.upload_file(
                    json.dumps(
                        {
                            "host_key": host_key.export_private_key().decode(),
                            "password": password,
                            "websocket_port": _WEBSOCKET_PORT,
                            "model_port": endpoint_proxy.DEFAULT_PORT,
                        }
                    ).encode(),
                    config_path,
                    timeout=30,
                )
                await sandbox.process.create_session(session_name, request_timeout=20)
                command = (
                    f"chmod 755 {connector_path} && exec {connector_path} {config_path} >{shlex.quote(log_path)} 2>&1"
                )
                await sandbox.process.execute_session_command(
                    session_name,
                    daytona.SessionExecuteRequest(command=f"bash -c {shlex.quote(command)}", run_async=True),
                    timeout=10,
                )
                ready = await sandbox.process.exec(f"bash -c {shlex.quote(_READY_PROBE)}", timeout=130)
                if ready.exit_code:
                    raise RuntimeError(f"Model tunnel connector failed to start; see {log_path!r} in the sandbox")
            yield host_key, password
        finally:
            # Shielded: a supervisor cancelled by job teardown mid-cleanup must
            # still remove the connector from a sandbox that outlives the job.
            removal = asyncio.ensure_future(self._remove_connector(sandbox, session_name, connector_dir))
            cancelled: asyncio.CancelledError | None = None
            with _tunnel_stage(sandbox, trial_name, "connector cleanup"):
                while not removal.done():
                    try:
                        await asyncio.shield(removal)
                    except asyncio.CancelledError as error:
                        cancelled = error
                removal.result()
            if cancelled is not None:
                raise cancelled

    async def _remove_connector(self, sandbox: daytona.AsyncSandbox, session_name: str, connector_dir: str) -> None:
        """Stops the connector session and deletes its files."""

        try:
            # A failed create request may still have created the session remotely.
            with contextlib.suppress(daytona.DaytonaNotFoundError):
                await sandbox.process.delete_session(session_name, request_timeout=20)
        finally:
            with contextlib.suppress(daytona.DaytonaNotFoundError):
                await sandbox.fs.delete_file(connector_dir, recursive=True, request_timeout=20)

    @contextlib.asynccontextmanager
    async def _forward_model(
        self, sandbox: daytona.AsyncSandbox, *, host_key: asyncssh.SSHKey, password: str, proxy_port: int
    ) -> AsyncIterator[asyncssh.SSHClientConnection]:
        """Opens a model tunnel using a new Daytona preview link.

        A new link avoids reusing an expired preview token after a disconnect.

        Raises:
            ValueError: If the preview URL is not HTTPS or its token is missing.
            daytona.DaytonaError: If a Daytona operation fails. SSH and WebSocket
                connection errors are also passed to the caller.
        """

        preview = await sandbox.get_preview_link(_WEBSOCKET_PORT, request_timeout=20)
        if not preview.url.startswith("https://") or not preview.token:
            raise ValueError("Daytona did not return an authenticated HTTPS preview for the model tunnel")
        async with tunnel.forward_model_requests(
            preview.url.replace("https://", "wss://", 1),
            {"X-Daytona-Preview-Token": preview.token},
            host_key=host_key,
            password=password,
            model_port=endpoint_proxy.DEFAULT_PORT,
            proxy_port=proxy_port,
        ) as connection:
            yield connection

    @contextlib.asynccontextmanager
    async def connect_sandbox(
        self, sandbox: daytona.AsyncSandbox, proxy_port: int, *, log_path: str
    ) -> AsyncIterator[asyncssh.SSHClientConnection]:
        """Connects a sandbox's model port to the host-side proxy.

        Start after the host-side proxy is ready; exit before closing the proxy
        or sandbox. Does not reconnect. Exit closes the connection and attempts
        to remove the connector, including after startup failures.

        Yields:
            The SSH connection. Use `is_closed()` to detect a disconnect.

        Raises:
            ValueError: If the sandbox is public or its preview lacks HTTPS or a token.
            RuntimeError: If the connector cannot be installed or started.
            daytona.DaytonaError: If a Daytona operation fails. SSH and WebSocket
                connection errors are also passed to the caller.
        """
        async with self._connector_session(sandbox, log_path=log_path) as (host_key, password):
            async with self._forward_model(
                sandbox, host_key=host_key, password=password, proxy_port=proxy_port
            ) as connection:
                yield connection

    async def _maintain_tunnel(
        self,
        sandbox: daytona.AsyncSandbox,
        proxy_port: int,
        *,
        log_path: str,
        stopped: asyncio.Event,
        ready: asyncio.Event,
        trial_name: str,
    ) -> None:
        """Keeps a Harbor trial's model tunnel connected while its agent runs.

        Starts the connector once and reconnects if the tunnel disconnects.
        Raises the last error after `_MAX_RECONNECT_ATTEMPTS` consecutive
        connection failures. This only restores the tunnel; the agent decides
        whether to retry failed model requests.

        Args:
            sandbox: The trial's running private sandbox.
            proxy_port: Host-side proxy port.
            log_path: Sandbox file for connector output.
            stopped: Set by the caller to close the tunnel.
            ready: Set after the first connection and each successful reconnect.
            trial_name: Trial name included in failure logs.
        """

        # A provider preview blip surfaces as any of these; a dead sandbox raises
        # them repeatedly. `ChannelListenError` subclasses `Exception` directly and
        # is what a forward refused by a lingering prior listener raises.
        transient = (
            asyncssh.Error,
            asyncssh.ChannelListenError,
            OSError,
            daytona.DaytonaError,
            ValueError,
            websockets.exceptions.WebSocketException,
        )
        async with self._connector_session(sandbox, log_path=log_path, trial_name=trial_name) as (host_key, password):
            failures = 0
            while not stopped.is_set():
                try:
                    async with self._forward_model(
                        sandbox, host_key=host_key, password=password, proxy_port=proxy_port
                    ) as connection:
                        failures = 0
                        ready.set()
                        await _wait_first(stopped.wait(), connection.wait_closed())
                    if stopped.is_set():
                        return
                except transient:
                    failures += 1
                    if stopped.is_set() or failures >= _MAX_RECONNECT_ATTEMPTS:
                        stage = "forwarding shutdown" if stopped.is_set() else "reconnect attempts exhausted"
                        with _tunnel_stage(sandbox, trial_name, stage):
                            raise
                    await asyncio.sleep(_RECONNECT_BACKOFF_S)
                except Exception:
                    with _tunnel_stage(sandbox, trial_name, "forwarding"):
                        raise

    @contextlib.asynccontextmanager
    async def model_tunnels(self, harbor_job: job.Job, proxy_port: int) -> AsyncIterator[None]:
        """Manages one model tunnel per Harbor trial while its agent runs.

        Enter once, before `harbor_job.run()`, and keep the host-side proxy
        running until exit. Hooks remain registered for the rest of the job.

        Raises:
            RuntimeError: Through Harbor hooks if a trial has no unique running
                sandbox or its tunnel fails. Task validation and connection
                errors are also reported through these hooks.
        """

        tunnel.binary_path("x86_64")
        supervisors: dict[str, contextlib.AbstractAsyncContextManager[None]] = {}
        async with contextlib.AsyncExitStack() as cleanup:

            async def start(event: hooks.TrialHookEvent) -> None:
                manager = await environment.DaytonaClientManager.get_instance()
                client = await manager.get_client()
                query = daytona.ListSandboxesQuery(labels={"harbor.session_id": f"{event.trial_name}__env"})
                # The provider's listing can lag a just-started sandbox's state.
                for attempt in range(_SANDBOX_LOOKUP_ATTEMPTS):
                    matches = [sandbox async for sandbox in client.list(query, request_timeout=20)]
                    try:
                        sandbox = self.trial_sandbox(matches, event.trial_name)
                        break
                    except RuntimeError:
                        if attempt == _SANDBOX_LOOKUP_ATTEMPTS - 1:
                            raise
                        await asyncio.sleep(_RECONNECT_BACKOFF_S)
                stopped, ready = asyncio.Event(), asyncio.Event()
                supervisor = supervise_tunnel(
                    self._maintain_tunnel(
                        sandbox,
                        proxy_port,
                        log_path=_CONNECTOR_LOG_PATH,
                        stopped=stopped,
                        ready=ready,
                        trial_name=event.trial_name,
                    ),
                    ready,
                    stopped,
                )
                await supervisor.__aenter__()
                supervisors[event.trial_name] = supervisor
                cleanup.push_async_callback(supervisor.__aexit__, None, None, None)

            async def stop(event: hooks.TrialHookEvent) -> None:
                supervisor = supervisors.pop(event.trial_name, None)
                if supervisor is not None:
                    # Harbor emits AGENT_END in a finally block. Pass its active
                    # error through so tunnel cleanup cannot replace the outcome.
                    await supervisor.__aexit__(*sys.exc_info())

            harbor_job.on_trial_started(self.validate_task)
            harbor_job.on_agent_started(start)
            harbor_job.on_agent_ended(stop)
            harbor_job.on_trial_ended(stop)
            yield
