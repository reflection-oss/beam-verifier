"""Run Harbor jobs in separate processes and close sandbox service clients."""

import asyncio
import contextlib
import dataclasses
import multiprocessing
import os
import signal
import sys
import threading

from harbor import job
from harbor.models import environment_type
from harbor.models.job import config as job_config
from harbor.models.task import config as task_config
from harbor.models.task import id as task_id
from harbor.models.task import paths as task_paths
from harbor.trial import hooks
from harbor.utils import logger


@dataclasses.dataclass
class HarborWorker:
    """Run a Harbor job in a separate process.

    Args:
        config: Harbor job settings, without host model credentials.
        proxy_port: Host-side proxy port for Daytona tunnels; `None` disables them.
    """

    config: job_config.JobConfig
    proxy_port: int | None = None

    async def run_in_subprocess(self) -> None:
        """Start a worker process and wait for it to finish, including cleanup.

        On cancellation, interrupt the worker and wait for it to exit before
        returning control to the caller.

        Raises:
            RuntimeError: If the worker exits unsuccessfully.
        """
        loop = asyncio.get_running_loop()
        exited = asyncio.Event()
        task = asyncio.current_task()
        handles_sigterm = threading.current_thread() is threading.main_thread()
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        # Spawn avoids inheriting provider clients, threads and event-loop state.
        with contextlib.closing(multiprocessing.get_context("spawn").Process(target=self.run_job)) as process:
            process.start()
            loop.add_reader(process.sentinel, exited.set)
            try:
                if handles_sigterm and task is not None:
                    loop.add_signal_handler(signal.SIGTERM, task.cancel)
                try:
                    await exited.wait()
                except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                    if process.is_alive() and process.pid is not None:
                        with contextlib.suppress(ProcessLookupError):
                            os.kill(process.pid, signal.SIGINT)
                    await exited.wait()
                    raise
            finally:
                loop.remove_reader(process.sentinel)
                process.join()
                if handles_sigterm and task is not None:
                    loop.remove_signal_handler(signal.SIGTERM)
                    signal.signal(signal.SIGTERM, previous_sigterm)
            if process.exitcode:
                raise RuntimeError(f"Harbor worker exited with status {process.exitcode}; see console output")

    def run_job(self) -> None:
        """Execute Harbor in the worker process and close its service clients."""
        # Only the parent receives terminal signals and forwards cancellation.
        os.setsid()
        config = self.config
        with asyncio.Runner() as runtime, contextlib.ExitStack() as cleanup:
            if config.environment.type == environment_type.EnvironmentType.DAYTONA:
                # The provider SDK is installed only with the Daytona extra.
                from harbor.environments.daytona import environment  # noqa: PLC0415

                manager = runtime.run(environment.DaytonaClientManager.get_instance())
                if "connection_pool_maxsize" in config.environment.kwargs:
                    runtime.run(
                        manager.configure(connection_pool_maxsize=config.environment.kwargs["connection_pool_maxsize"])
                    )
                client = runtime.run(manager.get_client())
                cleanup.callback(runtime.run, client.close())
            isolate_task_environment(local_docker=config.environment.type == environment_type.EnvironmentType.DOCKER)
            harbor_job = runtime.run(job.Job.create(config))
            if config.environment.type == environment_type.EnvironmentType.DOCKER:
                harbor_job.on_trial_started(validate_local_task)
            if not config.quiet:
                logger.logger.info("Harbor job %s; detailed log: %s", config.job_name, harbor_job.job_dir / "job.log")
                # Rich redraws its progress display only on a terminal. Redirected
                # runs need persistent lifecycle events instead of screen updates.
                if not sys.stdout.isatty():
                    harbor_job.on_trial_started(log_trial_event)
                    harbor_job.on_trial_ended(log_trial_event)
            if config.environment.type == environment_type.EnvironmentType.DAYTONA and self.proxy_port is not None:
                from beam_verifier.benchmarks.proxies import daytona  # noqa: PLC0415

                tunnels = contextlib.AsyncExitStack()
                cleanup.callback(runtime.run, tunnels.aclose())
                runtime.run(
                    tunnels.enter_async_context(daytona.DaytonaTunnels().model_tunnels(harbor_job, self.proxy_port))
                )
            runtime.run(harbor_job.run())


async def validate_local_task(event: hooks.TrialHookEvent) -> None:
    """Reject custom Compose files and host mounts before creating containers.

    These could expose host credentials or the Docker socket. Local tasks use
    Harbor's container configuration with a task image or Dockerfile.

    Args:
        event: Harbor's trial-start notification, before sandbox creation.

    Raises:
        ValueError: If the task or caller supplies Docker Compose or host mounts.
    """
    identity = event.result.task_id
    if isinstance(identity, task_id.PackageTaskId):
        identity = identity.model_copy(update={"ref": event.lock.task.digest})
    paths = task_paths.TaskPaths(identity.get_local_path())
    definition = task_config.TaskConfig.model_validate_toml(paths.config_path.read_text())
    contexts = [paths.environment_dir, paths.tests_dir]
    contexts.extend(paths.step_tests_dir(step.name) for step in definition.steps or [])
    environment_config = event.config.environment
    if (
        any((directory / "docker-compose.yaml").exists() for directory in contexts)
        or environment_config.extra_docker_compose
        or environment_config.mounts
        or environment_config.kwargs.get("extra_docker_compose")
        or environment_config.kwargs.get("mounts")
    ):
        raise ValueError("Local Docker tasks cannot supply Compose or host mounts; use an image or Dockerfile")


async def log_trial_event(event: hooks.TrialHookEvent) -> None:
    """Log trial starts and ends when output is not an interactive terminal.

    Args:
        event: Harbor's trial start or end notification.
    """
    result = event.result
    logger.logger.info(
        "Trial %s %s; rewards=%s; error=%s",
        event.trial_name,
        event.event.value,
        result.verifier_result.rewards if result.verifier_result else None,
        result.exception_info.exception_type if result.exception_info else None,
    )


def isolate_task_environment(*, local_docker: bool = False) -> None:
    """Remove host secrets from the worker's environment before loading tasks.

    Call only inside the worker, after creating the Daytona client if needed.
    Harbor can read host environment variables into sandbox settings. Keeping
    only required path, locale, certificate, and Docker settings prevents tasks
    from copying API keys into their sandboxes.

    Args:
        local_docker: Keep `DOCKER_HOST` and `DOCKER_CONTEXT` for the Docker CLI.

    Raises:
        ValueError: If `DOCKER_HOST` specifies anything other than a Unix socket.
    """
    runtime_keys = {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR"}
    if local_docker:
        docker_host = os.environ.get("DOCKER_HOST")
        if docker_host and not docker_host.startswith("unix://"):
            raise ValueError("Local Docker requires a Unix socket in DOCKER_HOST")
        runtime_keys.update({"DOCKER_HOST", "DOCKER_CONTEXT"})
    for key in tuple(os.environ):
        if key not in runtime_keys:
            del os.environ[key]
