"""Exercise prompts through native solvers and offline model generation."""

import pathlib
import subprocess
import sys

import inspect_ai
import pytest
from inspect_ai import dataset, log, model, scorer, solver, tool
from inspect_evals.utils import aime_common

from beam_verifier import records
from beam_verifier.benchmarks import prompts, registry
from beam_verifier.benchmarks.scaffolds import base
from beam_verifier.benchmarks.scaffolds.inspect_ai import runner


@inspect_ai.task(name="beam_verifier_prompt_aime")
def aime_task(question: str = "What is 6 times 7?") -> inspect_ai.Task:
    """Use the real AIME solver and scorer with a local sample."""
    return inspect_ai.Task(
        dataset=[dataset.Sample(input=question, target="42")],
        solver=aime_common.aime_solver(),
        scorer=aime_common.aime_scorer(),
    )


@inspect_ai.task(name="beam_verifier_prompt_gpqa")
def gpqa_task() -> inspect_ai.Task:
    """Use GPQA's native choice formatting and scoring with local samples."""
    return inspect_ai.Task(
        dataset=[
            dataset.Sample(input="Which is even?", choices=["2", "3"], target="A"),
            dataset.Sample(input="Which is odd?", choices=["3", "2"], target="A"),
        ],
        solver=solver.multiple_choice(cot=True),
        scorer=scorer.choice(),
    )


@tool.tool
def echo() -> tool.Tool:
    """A local tool whose schema and response must survive prompting."""

    async def execute(text: str) -> str:
        """Echo text.

        Args:
          text: Text to return.
        """
        return text

    return execute


@inspect_ai.task(name="beam_verifier_prompt_conversation")
def conversation_task() -> inspect_ai.Task:
    """Format instructions during solving, then continue after a tool call."""
    return inspect_ai.Task(
        dataset=[
            dataset.Sample(
                input=[
                    model.ChatMessageSystem(content="Sample instructions."),
                    model.ChatMessageUser(content="Echo hi."),
                ],
                target="hi",
            )
        ],
        solver=[solver.system_message("Solver instructions."), solver.use_tools(echo()), solver.generate()],
        scorer=scorer.exact(),
    )


def _run(
    tmp_path: pathlib.Path,
    task: str,
    prompt_factory: prompts.BeamPromptFactory | None,
    outputs: list[model.ModelOutput] | None,
    task_args: dict[str, object] | str | None = None,
) -> log.EvalLog:
    dataset_runner = runner.InspectAIRunner(dataset_id="local", task=task)
    outcome = dataset_runner(
        base.RunRequest(
            base_url="http://unused/v1",
            model_name="unused",
            run_dir=tmp_path,
            settings=records.EvalSettings(
                prompt_factory=prompt_factory,
                harness_kwargs={
                    "model": "mockllm/model",
                    "model_args": {"custom_outputs": outputs} if outputs is not None else {},
                    "task_args": task_args or {},
                    "display": "none",
                },
            ),
        )
    )
    assert outcome.result.status == "ok", outcome.warning
    logs = log.list_eval_logs(str(tmp_path / "inspect_logs" / "local"))
    result = log.read_eval_log(logs[0].name)
    assert result.status == "success", result.error
    return result


def test_aime_prompt_accepts_display_math_answer_without_changing_output(tmp_path: pathlib.Path) -> None:
    result = _run(
        tmp_path,
        "beam_verifier_prompt_aime",
        registry.DATASETS["aime2026"].resolve_settings().prompt_factory,
        [model.ModelOutput.from_content("mockllm", "\\[\n\\boxed{42}\n\\]")],
        task_args={"question": "Compute {6 * 7}."},
    )
    assert result.samples is not None
    sample = result.samples[0]
    assert sample.messages[0].text == (
        "Solve the following problem and output the final answer within \\boxed{}.\nCompute {6 * 7}."
    )
    assert sample.scores is not None
    assert sample.scores["beam_boxed_math"].value == "C"
    assert sample.scores["beam_boxed_math"].answer == "42"
    assert sample.output.completion == "\\[\n\\boxed{42}\n\\]"
    assert result.eval.task_args["prompt_factory"] == "beam-aime-v1"
    assert result.eval.task_args["task_args"] == {"question": "Compute {6 * 7}."}


def test_gpqa_prompt_preserves_choices_and_is_scoped_per_sample(tmp_path: pathlib.Path) -> None:
    result = _run(
        tmp_path,
        "beam_verifier_prompt_gpqa",
        registry.DATASETS["gpqa-diamond"].resolve_settings().prompt_factory,
        [
            model.ModelOutput.from_content("mockllm", "ANSWER: A"),
            model.ModelOutput.from_content("mockllm", "ANSWER: A"),
        ],
    )
    assert result.samples is not None
    first, second = result.samples
    assert (
        first.messages[0].text
        == second.messages[0].text
        == (
            "You are an expert in answering questions about STEM subjects.\n"
            "You are given some examples as well as a question and a list of choices.\n"
            "You need to answer the question based on the choices and the explanation provided.\n"
            "Always respond with the letter corresponding to the best answer."
        )
    )
    assert "Which is even?" in first.messages[1].text
    assert "A) 2\nB) 3" in first.messages[1].text
    assert "Which is odd?" in second.messages[1].text
    assert first.scores is not None and second.scores is not None
    assert first.scores["choice"].value == second.scores["choice"].value == "C"


def test_conversation_prompt_preserves_tool_calls_and_later_turns(tmp_path: pathlib.Path) -> None:
    result = _run(
        tmp_path,
        "beam_verifier_prompt_conversation",
        prompts.BeamPromptFactory("conversation-v1"),
        [
            model.ModelOutput(
                model="mockllm",
                choices=[
                    model.ChatCompletionChoice(
                        message=model.ChatMessageAssistant(
                            content="",
                            tool_calls=[tool.ToolCall(id="echo-call", function="echo", arguments={"text": "hi"})],
                        )
                    )
                ],
            ),
            model.ModelOutput.from_content("mockllm", "hi"),
        ],
    )
    assert result.samples is not None
    sample = result.samples[0]
    assert [message.role for message in sample.messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert sample.messages[0].text == "Sample instructions.\n\nSolver instructions."
    assistant = sample.messages[2]
    response = sample.messages[3]
    assert isinstance(assistant, model.ChatMessageAssistant)
    assert assistant.tool_calls == [tool.ToolCall(id="echo-call", function="echo", arguments={"text": "hi"})]
    assert isinstance(response, model.ChatMessageTool)
    assert (response.tool_call_id, response.text) == ("echo-call", "hi")
    assert sample.scores is not None and sample.scores["exact"].value == "C"


@pytest.mark.parametrize("arguments", ['question: "Compute {6 * 7}."', '{"question": "Compute {6 * 7}."}'])
def test_retry_restores_prompt_and_native_file_arguments(tmp_path: pathlib.Path, arguments: str) -> None:
    config_file = tmp_path / "task-args.yaml"
    config_file.write_text(arguments)
    result = _run(
        tmp_path,
        "beam_verifier_prompt_aime",
        registry.DATASETS["aime2026"].resolve_settings().prompt_factory,
        None,
        task_args=str(config_file),
    )
    config_file.unlink()
    # Model an interrupted log with no completed samples to reuse.
    interrupted = result.model_copy(update={"samples": []})
    assert result.location is not None
    log.write_eval_log(interrupted, result.location)
    script = f"""
import pathlib
import runpy
import sys
from {records.__package__} import records
from {prompts.__package__} import prompts
from {base.__package__} import base
from {runner.__package__} import runner
runpy.run_path(sys.argv[1])
dataset_runner = runner.InspectAIRunner(
    "local", task="beam_verifier_prompt_aime",
)
outcome = dataset_runner(base.RunRequest(
    base_url="http://unused/v1", model_name="unused", run_dir=pathlib.Path(sys.argv[2]), resume=True,
    settings=records.EvalSettings(
        prompt_factory=prompts.BeamPromptFactory("replacement-v1", user_template="Wrong prompt"),
        harness_kwargs={{"task_args": sys.argv[3], "display": "none"}},
    ),
))
assert outcome.result.status == "ok", outcome.warning
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, __file__, str(tmp_path), str(config_file)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    logs = log.list_eval_logs(str(tmp_path / "inspect_logs" / "local"))
    retried = log.read_eval_log(logs[0].name)
    assert retried.status == "success", retried.error
    assert retried.samples is not None and result.samples is not None
    assert retried.samples[0].messages[0].text == result.samples[0].messages[0].text
    assert retried.eval.task_args["prompt_factory"] == "beam-aime-v1"
    assert retried.eval.task_args["task_args"] == {"question": "Compute {6 * 7}."}
