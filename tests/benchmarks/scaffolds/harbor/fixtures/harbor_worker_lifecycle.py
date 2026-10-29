"""Test worker cancellation in a separate process so pytest receives no signals."""

import asyncio
import dataclasses
import multiprocessing
import os
import pathlib
import signal
import sys
from multiprocessing import synchronize

import pytest
from harbor.models.job import config as job_config

from beam_verifier.benchmarks.scaffolds.harbor import worker


@dataclasses.dataclass(kw_only=True)
class LifecycleWorker(worker.HarborWorker):
    """A test worker that waits for permission to finish cleanup."""

    directory: pathlib.Path
    ready: synchronize.Event
    finish: synchronize.Event
    cleaning: synchronize.Event
    release: synchronize.Event

    def run_job(self) -> None:
        os.setsid()
        with asyncio.Runner() as runtime:
            runtime.run(self.run_task())

    async def run_task(self) -> None:
        (self.directory / "child.pid").write_text(str(os.getpid()))
        try:
            self.ready.set()
            await asyncio.to_thread(self.finish.wait)
        finally:
            # Cancellation must also unblock the executor thread before exit.
            self.finish.set()
            self.cleaning.set()
            await asyncio.to_thread(self.release.wait)
            (self.directory / "cleaned").touch()


async def control(probe: LifecycleWorker, supervisor: asyncio.Task[None], mode: str) -> None:
    """Finish or interrupt the worker, then check its parent waits for cleanup."""
    await asyncio.to_thread(probe.ready.wait)
    if mode == "complete":
        probe.finish.set()
    elif mode == "cancel":
        supervisor.cancel()
    else:
        os.kill(os.getpid(), signal.SIGINT if mode == "sigint" else signal.SIGTERM)
    await asyncio.to_thread(probe.cleaning.wait)
    try:
        assert not supervisor.done(), "Supervisor returned while child cleanup was blocked"
    finally:
        probe.release.set()


async def exercise(mode: str, directory: pathlib.Path) -> None:
    context = multiprocessing.get_context("spawn")
    probe = LifecycleWorker(
        job_config.JobConfig(),
        directory=directory,
        ready=context.Event(),
        finish=context.Event(),
        cleaning=context.Event(),
        release=context.Event(),
    )
    supervisor = asyncio.create_task(probe.run_in_subprocess())
    controller = asyncio.create_task(control(probe, supervisor, mode))
    try:
        await supervisor
    except asyncio.CancelledError:
        assert mode != "complete", "Normal completion was cancelled"
    else:
        assert mode == "complete", "Interruption was swallowed"
    finally:
        await controller
    assert (directory / "cleaned").exists(), "Supervisor returned before cleanup completed"
    with pytest.raises(ProcessLookupError):
        os.kill(int((directory / "child.pid").read_text()), 0)


if __name__ == "__main__":
    asyncio.run(exercise(sys.argv[1], pathlib.Path(sys.argv[2])))
