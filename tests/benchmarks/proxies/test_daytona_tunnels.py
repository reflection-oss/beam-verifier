"""Check sandbox selection and failure logs without contacting Daytona."""

import asyncio
import dataclasses
import pathlib
import sys

import daytona
import pytest
from daytona._async import filesystem, process
from harbor.models.trial import result as trial_result

from beam_verifier.benchmarks.proxies import daytona as daytona_proxy


@dataclasses.dataclass
class FailingProcess(process.AsyncProcess):
    """A fake Daytona process service that fails during connector setup."""

    failure: Exception
    cleanup_failure_type: type[Exception] | None

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> daytona.ExecuteResponse:
        del command, cwd, env, timeout  # Unused.
        try:
            raise self.failure
        except Exception as error:
            raise daytona.DaytonaError("Process service failed") from error

    async def delete_session(self, session_id: str, request_timeout: float | None = None) -> None:
        del session_id, request_timeout  # Unused.
        if self.cleanup_failure_type is not None:
            raise self.cleanup_failure_type("Connector removal failed")


class EmptyFileSystem(filesystem.AsyncFileSystem):
    """A fake filesystem that accepts cleanup without creating files."""

    def __init__(self) -> None:
        pass

    async def delete_file(self, path: str, recursive: bool = False, request_timeout: float | None = None) -> None:
        del path, recursive, request_timeout  # Unused.


class FailingSandbox(daytona.AsyncSandbox):
    """A fake sandbox with failing commands and an empty filesystem."""

    process_service: FailingProcess

    def model_post_init(self, context: object) -> None:
        """Skip Daytona event subscriptions for this fake sandbox."""
        del context  # Unused.

    @property
    def process(self) -> FailingProcess:
        return self.process_service

    @property
    def fs(self) -> EmptyFileSystem:
        return EmptyFileSystem()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_type", [ConnectionError, TimeoutError, ValueError])
@pytest.mark.parametrize(
    "cleanup_failure_type, expected_type, expected_message, expected_stage",
    [
        (None, daytona.DaytonaError, "Process service failed", "connector startup"),
        (OSError, OSError, "Connector removal failed", "connector cleanup"),
    ],
)
async def test_connect_sandbox_preserves_causes_in_saved_error_and_log(
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    failure_type: type[Exception],
    cleanup_failure_type: type[Exception] | None,
    expected_type: type[Exception],
    expected_message: str,
    expected_stage: str,
) -> None:
    sandbox = FailingSandbox.model_construct(
        id="sandbox-under-test",
        public=False,
        process_service=FailingProcess(failure_type("Original failure"), cleanup_failure_type),
    )
    saved_error = tmp_path / "exception.json"
    with pytest.raises(expected_type, match=expected_message):
        try:
            async with daytona_proxy.DaytonaTunnels().connect_sandbox(sandbox, 8900, log_path="connector.log"):
                pytest.fail("A failed connector must not become ready")
        except Exception as error:
            saved_error.write_text(trial_result.ExceptionInfo.from_exception(error).model_dump_json())
            raise

    recorded = trial_result.ExceptionInfo.model_validate_json(saved_error.read_text())
    assert f"{failure_type.__name__}: Original failure" in recorded.exception_traceback
    assert "DaytonaError: Process service failed" in recorded.exception_traceback
    assert "stage=connector startup" in recorded.exception_traceback
    assert f"stage={expected_stage}" in recorded.exception_traceback
    assert recorded.exception_type == expected_type.__name__
    assert recorded.exception_message == expected_message
    assert "sandbox='sandbox-under-test'" in caplog.text
    assert f"stage={expected_stage}" in caplog.text
    assert expected_message in caplog.text
    assert "occurred_at=" in caplog.text
    assert f"{failure_type.__name__}: Original failure" in caplog.text


def test_trial_sandbox_rejects_duplicate_started_matches() -> None:
    actual = daytona.AsyncSandbox.model_construct(id="actual", state=daytona.SandboxState.STARTED)
    orphan = daytona.AsyncSandbox.model_construct(id="orphan", state=daytona.SandboxState.STARTED)
    with pytest.raises(RuntimeError, match="Expected one started Daytona sandbox.*found 2"):
        daytona_proxy.DaytonaTunnels().trial_sandbox([actual, orphan], "trial")


def test_trial_sandbox_selects_only_started_match() -> None:
    started = daytona.AsyncSandbox.model_construct(id="started", state=daytona.SandboxState.STARTED)
    stopped = daytona.AsyncSandbox.model_construct(id="stopped", state=daytona.SandboxState.STOPPED)
    assert daytona_proxy.DaytonaTunnels().trial_sandbox([stopped, started], "trial") is started


def test_trial_sandbox_rejects_missing_match() -> None:
    with pytest.raises(RuntimeError, match="Expected one started Daytona sandbox.*found 0"):
        daytona_proxy.DaytonaTunnels().trial_sandbox([], "trial")


@pytest.mark.asyncio
async def test_supervise_tunnel_interrupts_agent_after_readiness() -> None:
    ready, stopped = asyncio.Event(), asyncio.Event()
    disconnect, cleaned = asyncio.Event(), asyncio.Event()
    agent_cancelled = asyncio.Event()

    async def connection() -> None:
        ready.set()
        try:
            await disconnect.wait()
            raise ConnectionError("Reconnect attempts exhausted")
        finally:
            cleaned.set()

    async def trial() -> None:
        supervisor = daytona_proxy.supervise_tunnel(connection(), ready, stopped)
        await supervisor.__aenter__()
        try:
            disconnect.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                agent_cancelled.set()
                raise
        finally:
            await supervisor.__aexit__(*sys.exc_info())

    task = asyncio.create_task(trial())
    with pytest.raises(RuntimeError, match="Model tunnel failed") as raised:
        await asyncio.wait_for(task, timeout=5)
    assert isinstance(raised.value.__cause__, ExceptionGroup)
    assert "Reconnect attempts exhausted" in str(raised.value.__cause__.exceptions[0])
    assert agent_cancelled.is_set(), "The agent kept running after losing model access"
    assert cleaned.is_set(), "The connector outlived the failed trial"
    assert stopped.is_set()
    assert task.cancelling() == 0, "Tunnel failure left a pending job cancellation"


@pytest.mark.asyncio
async def test_supervise_tunnel_preserves_agent_error_when_cleanup_fails() -> None:
    ready, stopped = asyncio.Event(), asyncio.Event()

    async def connection() -> None:
        ready.set()
        await stopped.wait()
        raise ConnectionError("Connector cleanup failed")

    with pytest.raises(ValueError, match="Agent failed") as raised:
        async with daytona_proxy.supervise_tunnel(connection(), ready, stopped):
            raise ValueError("Agent failed")
    assert "model tunnel also failed" in raised.value.__notes__[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_fails", [False, True])
@pytest.mark.parametrize("becomes_ready", [False, True])
async def test_supervise_tunnel_preserves_trial_cancellation(cleanup_fails: bool, becomes_ready: bool) -> None:
    ready, stopped = asyncio.Event(), asyncio.Event()
    running, cleaned = asyncio.Event(), asyncio.Event()

    async def connection() -> None:
        if becomes_ready:
            ready.set()
        running.set()
        try:
            await stopped.wait()
        finally:
            cleaned.set()
            if cleanup_fails:
                raise ConnectionError("Connector cleanup failed")

    async def trial() -> None:
        async with daytona_proxy.supervise_tunnel(connection(), ready, stopped):
            await asyncio.Event().wait()

    task = asyncio.create_task(trial())
    await running.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    assert cleaned.is_set()
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_supervise_tunnel_reports_startup_failure() -> None:
    async def connection() -> None:
        raise ConnectionError("Connector startup failed")

    with pytest.raises(RuntimeError, match="Model tunnel failed"):
        async with daytona_proxy.supervise_tunnel(connection(), asyncio.Event(), asyncio.Event()):
            pytest.fail("A failed connector became ready")
