"""Exercise the InspectAI runner with native in-memory tasks and offline models."""

import os
import pathlib
import signal
import subprocess
import sys

import inspect_ai
import pytest
from inspect_ai import dataset as inspect_ai_dataset
from inspect_ai import log as inspect_ai_log
from inspect_ai import model as inspect_ai_model
from inspect_ai import scorer as inspect_ai_scorer
from inspect_ai import solver as inspect_ai_solver
from inspect_ai import tool as inspect_ai_tool

from beam_verifier import records
from beam_verifier.benchmarks.scaffolds import base
from beam_verifier.benchmarks.scaffolds.inspect_ai import runner


@inspect_ai.task(name="beam_verifier_option_forwarding")
def option_forwarding_task(answer: str = "default") -> inspect_ai.Task:
    """An in-memory task executed with InspectAI's own mock model."""
    return inspect_ai.Task(
        dataset=[inspect_ai_dataset.Sample(input="Answer the question.", target=answer)],
        solver=inspect_ai_solver.generate(),
        scorer=inspect_ai_scorer.match(),
    )


@pytest.mark.parametrize("parallel_tasks", [None, 2])
def test_runner_forwards_native_options_and_preserves_library_defaults(tmp_path: pathlib.Path, parallel_tasks):
    dataset_runner = runner.InspectAIRunner("offline", task_args={"answer": "C"})
    output_dir = tmp_path / "custom_logs"
    outcome = dataset_runner(
        base.RunRequest(
            base_url="http://127.0.0.1:1/v1",
            model_name="unused",
            run_dir=tmp_path,
            settings=records.EvalSettings(
                parallel_tasks=parallel_tasks,
                temperature=0.8,
                model_kwargs={"top_p": 0.9, "extra_body": {"custom_option": False}},
                harness_kwargs={
                    "tasks": option_forwarding_task,
                    "model": "mockllm/model",
                    "model_args": {"custom_outputs": [inspect_ai_model.ModelOutput.from_content("mockllm", "C")]},
                    "log_dir": str(output_dir),
                    "display": "none",
                    "temperature": 0.2,
                },
            ),
        )
    )
    assert outcome.result.status == "ok"
    assert outcome.traces[0].score == 1.0
    (saved_log,) = inspect_ai_log.list_eval_logs(str(output_dir))
    config = inspect_ai_log.read_eval_log(saved_log.name).eval.model_generate_config
    assert config.temperature == 0.2
    assert config.max_connections == parallel_tasks
    assert config.top_p == 0.9
    assert config.extra_body == {"custom_option": False}


@inspect_ai.task(name="beam_verifier_ten_samples")
def ten_samples_task() -> inspect_ai.Task:
    return inspect_ai.Task(
        dataset=[inspect_ai_dataset.Sample(input=f"q{i}", id=i) for i in range(10)],
        solver=inspect_ai_solver.generate(),
        scorer=inspect_ai_scorer.match(),
    )


def test_a_fixed_subset_is_the_same_random_sample_on_every_run(tmp_path: pathlib.Path):
    """`limit` after a seeded shuffle picks the same random subset on every run."""

    def run(directory):
        dataset_runner = runner.InspectAIRunner("offline", limit=4, shuffle_seed=7)
        harness_kwargs = {"tasks": ten_samples_task, "model": "mockllm/model", "display": "none"}
        request = base.RunRequest(
            base_url="http://127.0.0.1:1/v1",
            model_name="unused",
            run_dir=directory,
            settings=records.EvalSettings(harness_kwargs=harness_kwargs),
        )
        return dataset_runner(request).traces

    first, second = run(tmp_path / "a"), run(tmp_path / "b")
    ids = [trace.sample_id for trace in first]
    assert len(set(ids)) == 4
    assert ids == [trace.sample_id for trace in second]
    assert ids != ["0", "1", "2", "3"]


@inspect_ai_tool.tool
def add():
    async def execute(a: int, b: int) -> int:
        """Adds two numbers, the way a calculator would, and returns their sum as an integer value to the caller.

        Args:
          a: First addend.
          b: Second addend.
        """
        return a + b

    return execute


@inspect_ai.task(name="beam_verifier_offers_a_tool")
def tool_task() -> inspect_ai.Task:
    return inspect_ai.Task(
        dataset=[inspect_ai_dataset.Sample(input="What is 1 + 2?", target="3")],
        solver=[inspect_ai_solver.use_tools(add()), inspect_ai_solver.generate(tool_calls="none")],
        scorer=inspect_ai_scorer.match(),
    )


def test_traces_record_the_tools_the_request_offered(tmp_path: pathlib.Path):
    """Traces preserve the tool schemas the evaluation sent."""
    harness_kwargs = {"tasks": tool_task, "model": "mockllm/model", "display": "none"}
    request = base.RunRequest(
        base_url="http://127.0.0.1:1/v1",
        model_name="unused",
        run_dir=tmp_path,
        settings=records.EvalSettings(harness_kwargs=harness_kwargs),
    )
    (trace,) = runner.InspectAIRunner("offline")(request).traces
    (tool,) = trace.tools
    assert tool["type"] == "function" and tool["function"]["name"] == "add"
    assert tool["function"]["parameters"]["required"] == ["a", "b"]
    assert tool["function"]["parameters"]["properties"]["a"]["type"] == "integer"
    # Over 100 characters, so the log stores it as an attachment reference.
    assert tool["function"]["description"].startswith("Adds two numbers, the way a calculator would")


def test_partial_model_args_preserve_caller_credentials_and_transport_settings(tmp_path: pathlib.Path):
    request = base.RunRequest(
        base_url="http://endpoint/v1",
        model_name="beam",
        api_key="synthetic-test-key",
        run_dir=tmp_path,
        settings=records.EvalSettings(
            client_timeout=120, harness_kwargs={"model_args": {"strict_tools": False, "stream": False}}
        ),
    )
    kwargs = runner.InspectAIRunner("offline").build_eval_kwargs(request)
    assert kwargs["model_base_url"] == "http://endpoint/v1"
    assert kwargs["model_args"] == {
        "api_key": "synthetic-test-key",
        "client_timeout": 120,
        "strict_tools": False,
        "stream": False,
    }
    assert request.settings.harness_kwargs["model_args"] == {"strict_tools": False, "stream": False}


def test_native_model_options_override_transport_defaults(tmp_path: pathlib.Path):
    settings = records.EvalSettings(
        harness_kwargs={"model_args": {"api_key": "synthetic-override-key", "strict_tools": True, "stream": True}}
    )
    kwargs = runner.InspectAIRunner("offline").build_eval_kwargs(
        base.RunRequest(base_url="http://endpoint/v1", model_name="beam", run_dir=tmp_path, settings=settings)
    )
    assert kwargs["model_args"] == {
        "api_key": "synthetic-override-key",
        "client_timeout": 600,
        "strict_tools": True,
        "stream": True,
    }


def test_runner_surfaces_downstream_configuration_errors(tmp_path: pathlib.Path):
    with pytest.raises(base.RunnerError, match="custom_outputs.*must be an Iterable"):
        runner.InspectAIRunner("offline")(
            base.RunRequest(
                base_url="http://127.0.0.1:1/v1",
                model_name="unused",
                run_dir=tmp_path,
                settings=records.EvalSettings(
                    harness_kwargs={
                        "tasks": option_forwarding_task,
                        "model": "mockllm/model",
                        "model_args": {"custom_outputs": 42},
                        "display": "none",
                    }
                ),
            )
        )


def test_resume_retries_only_unfinished_samples_and_keeps_zero_scores(tmp_path: pathlib.Path) -> None:
    calls = tmp_path / "calls"
    dataset_runner = runner.InspectAIRunner("offline", task_args={"calls_path": str(calls)})
    request = base.RunRequest(
        base_url="http://unused/v1",
        model_name="unused",
        run_dir=tmp_path,
        settings=records.EvalSettings(
            harness_kwargs={
                "tasks": str(pathlib.Path(__file__).parent / "fixtures" / "inspect_resume_task.py") + "@resume_task",
                "model": "mockllm/model",
                "display": "none",
                "max_samples": 1,
            }
        ),
    )
    first = dataset_runner(request)
    assert first.result.status == "partial"
    assert calls.read_text() == "1\n2\n"
    request.resume = True
    resumed = dataset_runner(request)
    assert resumed.result.status == "ok"
    assert calls.read_text() == "1\n2\n2\n"
    assert [trace.score for trace in resumed.traces] == [0.0, 1.0]
    assert resumed.result.n == 2
    assert resumed.result.metrics["accuracy"] == 0.5


def test_resume_recovers_a_sigkilled_inspect_process(tmp_path: pathlib.Path) -> None:
    calls = tmp_path / "calls"
    settings = records.EvalSettings(
        harness_kwargs={
            "tasks": str(pathlib.Path(__file__).parent / "fixtures" / "inspect_resume_task.py") + "@resume_task",
            "task_args": {"calls_path": str(calls), "kill": True},
            "model": "mockllm/model",
            "display": "none",
            "max_samples": 1,
            "log_buffer": 1,
        }
    )
    script = f"""
import pathlib
import sys
from {records.__package__} import records
from {base.__package__} import base
from {runner.__package__} import runner
runner.InspectAIRunner("offline")(base.RunRequest(
    base_url="http://unused/v1", model_name="unused", run_dir=pathlib.Path(sys.argv[1]),
    settings=records.EvalSettings.model_validate_json(sys.argv[2]),
))
"""
    killed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), settings.model_dump_json()],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert killed.returncode == -signal.SIGKILL, killed.stderr
    assert calls.read_text() == "1\n2\n"

    resumed = runner.InspectAIRunner("offline")(
        base.RunRequest(
            base_url="http://unused/v1", model_name="unused", run_dir=tmp_path, settings=settings, resume=True
        )
    )
    assert calls.read_text() == "1\n2\n2\n"
    assert [trace.score for trace in resumed.traces] == [0.0, 1.0]
    assert resumed.result.status == "ok"


def test_resume_restores_named_provider_credentials(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry constructs the saved provider using only the explicitly supplied key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("BEAM_VERIFIER_API_KEY", raising=False)
    dataset_runner = runner.InspectAIRunner("offline", task_args={"calls_path": str(tmp_path / "calls")})
    request = base.RunRequest(
        base_url="http://unused/v1",
        model_name="unused",
        api_key="initial-test-key",
        run_dir=tmp_path,
        settings=records.EvalSettings(
            harness_kwargs={
                "tasks": str(pathlib.Path(__file__).parent / "fixtures" / "inspect_resume_task.py") + "@resume_task",
                "model": "openai-api/beam-verifier/unused",
                "model_args": {"stream": False, "strict_tools": False},
                "display": "none",
                "max_samples": 1,
            }
        ),
    )
    assert dataset_runner(request).result.status == "partial"
    (saved_log,) = inspect_ai_log.list_eval_logs(str(tmp_path / "inspect_logs" / "offline"))
    saved = inspect_ai_log.read_eval_log(saved_log.name)
    assert saved.eval.model == "openai-api/beam-verifier/unused"
    assert "api_key" not in saved.eval.model_args
    assert "initial-test-key" not in saved.model_dump_json()

    request.resume = True
    request.api_key = "retry-test-key"
    assert dataset_runner(request).result.status == "ok"
    assert "BEAM_VERIFIER_API_KEY" not in os.environ
