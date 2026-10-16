"""Apply the configured Beam prompt before a task's first model request."""

import json
from typing import Literal, Unpack

import inspect_ai
import yaml
from inspect_ai import model, solver, util

from beam_verifier.benchmarks import prompts
from beam_verifier.benchmarks.scaffolds.inspect_ai import scoring


def task_arguments(arguments: dict[str, object] | str) -> dict[str, object]:
    """Resolve native file arguments before nesting them in the wrapper task.

    Args:
      arguments: Native task arguments or the path to their JSON/YAML file.

    Returns:
      Resolved arguments, so saved logs do not depend on the original file.

    Raises:
      ValueError: If the file does not contain an argument mapping.
    """
    if not isinstance(arguments, str):
        return arguments
    content = util.resource(arguments, type="file")
    resolved = json.loads(content) if content.lstrip().startswith("{") else yaml.safe_load(content)
    if not isinstance(resolved, dict):
        raise ValueError(f"task_args file {arguments!r} must contain a mapping")
    return resolved


def _render(state: solver.TaskState, factory: prompts.BeamPromptFactory) -> None:
    """Render after the native solver has formatted its question and choices."""
    if factory.user_template is not None:
        if not isinstance(state.input, str):
            raise ValueError("a user prompt template requires a text sample")
        state.user_prompt.text = factory.render_user(state.input)
    system_parts = [message.text for message in state.messages if isinstance(message, model.ChatMessageSystem)]
    if factory.system_text is not None:
        system_parts.append(factory.system_text)
    messages: list[model.ChatMessage] = [
        message for message in state.messages if not isinstance(message, model.ChatMessageSystem)
    ]
    if system_parts:
        messages.insert(0, model.ChatMessageSystem(content="\n\n".join(system_parts)))
    state.messages = messages


@solver.solver
def beam_prompt(native: solver.Solver, factory: prompts.BeamPromptFactory) -> solver.Solver:
    """Keep native execution while adapting each sample attempt exactly once.

    Args:
      native: The benchmark's solver, including any formatting and tool loop.
      factory: Shared Beam templates applied to the native conversation.

    Returns:
      A solver with prompt state isolated to each sample attempt.
    """

    async def solve(state: solver.TaskState, generate: solver.Generate) -> solver.TaskState:
        rendered = False

        async def generate_with_prompt(
            state: solver.TaskState,
            tool_calls: Literal["loop", "single", "none"] = "loop",
            **kwargs: Unpack[model.GenerateConfigArgs],
        ) -> solver.TaskState:
            nonlocal rendered
            if not rendered:
                _render(state, factory)
                rendered = True
            return await generate(state, tool_calls=tool_calls, **kwargs)

        state = await native(state, generate_with_prompt)
        if not rendered:
            _render(state, factory)
        return state

    return solve


@inspect_ai.task
def beam_prompt_task(
    task_name: str,
    task_args: dict[str, object],
    prompt_factory: str,
    user_template: str | None = None,
    system_text: str | None = None,
) -> inspect_ai.Task:
    """Load a registered task and apply its Beam prompting protocol.

    Args:
      task_name: Native registered task name.
      task_args: Arguments forwarded to that task unchanged.
      prompt_factory: Versioned prompt identity retained in the evaluation log.
      user_template: Optional template over the original text sample.
      system_text: Optional benchmark instructions to add before generation.

    Returns:
      The native task with its solver wrapped at generation time.
    """
    task = util.registry_create("task", task_name, **task_args)
    factory = prompts.BeamPromptFactory(name=prompt_factory, user_template=user_template, system_text=system_text)
    task.solver = beam_prompt(task.solver, factory)
    if prompt_factory == "beam-aime-v1":
        # The upstream AIME scorer expects an answer on the last line, while
        # this protocol accepts a box anywhere in the final response.
        task.scorer = [scoring.beam_boxed_math()]
    return task
