"""An offline InspectAI task that stops once after its first graded sample."""

import os
import pathlib
import signal

import inspect_ai
from inspect_ai import dataset, model, scorer, solver

# InspectAI's file discovery requires the bare `@task` name, so this alias must stay public.
task = inspect_ai.task  # pylint: disable=import-alias-should-be-private


@solver.solver
def interrupted_sample(calls_path: str, *, kill: bool) -> solver.Solver:
    async def solve(state: solver.TaskState, generate: solver.Generate) -> solver.TaskState:
        calls = pathlib.Path(calls_path)
        previous = calls.read_text() if calls.exists() else ""
        calls.write_text(previous + f"{state.sample_id}\n")
        if state.sample_id == 2 and "2\n" not in previous:
            if kill:
                os.kill(os.getpid(), signal.SIGKILL)
            raise RuntimeError("interrupted before the second sample finished")
        state.output = model.ModelOutput.from_content("mockllm", "C")
        return state

    return solve


@task
def resume_task(calls_path: str, *, kill: bool = False) -> inspect_ai.Task:
    return inspect_ai.Task(
        dataset=[
            dataset.Sample(id=1, input="first", target="wrong"),
            dataset.Sample(id=2, input="second", target="C"),
        ],
        solver=interrupted_sample(calls_path, kill=kill),
        scorer=scorer.match(),
    )
