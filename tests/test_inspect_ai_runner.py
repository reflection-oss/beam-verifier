"""Offline tests for the InspectAI runner and the parser for the logs it reads.

No network: the log records here are built with InspectAI's own types rather than
read from a run, so they stay faithful to what `parse_log` is handed in practice.
"""

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
from beam_verifier.benchmarks.scaffolds.inspect_ai import log, runner


def _eval_spec() -> inspect_ai_log.EvalSpec:
    """The run identity every log record in this module shares."""
    return inspect_ai_log.EvalSpec(
        created="2026-01-01T00:00:00+00:00",
        task="offline",
        dataset=inspect_ai_log.EvalDataset(),
        model="openai-api/beam-verifier/m",
        config=inspect_ai_log.EvalConfig(),
    )


@pytest.fixture
def scored_log():
    """Builds the log of a run whose scorer reported an aggregate metric.

    Returns:
        A callable taking the run `status` and the `accuracy` its scorer recorded,
        and returning the `EvalLog`.
    """

    def build(status: str, accuracy: float) -> inspect_ai_log.EvalLog:
        return inspect_ai_log.EvalLog(
            eval=_eval_spec(),
            status=status,
            results=inspect_ai_log.EvalResults(
                scores=[
                    inspect_ai_log.EvalScore(
                        name="s",
                        scorer="s",
                        metrics={"accuracy": inspect_ai_log.EvalMetric(name="accuracy", value=accuracy)},
                    )
                ]
            ),
        )

    return build


@pytest.fixture
def sampled_log():
    """Builds the log of a run that recorded per-sample scores but no aggregate.

    Returns:
        A callable taking the run `status` and the per-sample scores, and returning
        the `EvalLog`.
    """

    def build(status: str, scores: tuple[float | str, ...]) -> inspect_ai_log.EvalLog:
        samples = [
            inspect_ai_log.EvalSample(
                id=index,
                epoch=1,
                input="q",
                target="t",
                scores={"scorer": inspect_ai_scorer.Score(value=value)},
            )
            for index, value in enumerate(scores)
        ]
        return inspect_ai_log.EvalLog(eval=_eval_spec(), status=status, samples=samples)

    return build


@pytest.fixture
def grouped_log():
    """Builds the log of a run whose scorer reported InspectAI `grouped()` metrics.

    The metric table mirrors a real grouped run: one entry per group for each
    grouped metric, plus that metric's aggregate over every group. InspectAI names
    every aggregate `all` and deduplicates the colliding key, so the second lands
    under `all2`.

    Returns:
        A callable taking the metrics the scorer also reported ungrouped, and
        returning the `EvalLog`.
    """

    def build(ungrouped: dict[str, float]) -> inspect_ai_log.EvalLog:
        accuracy = {
            "metric": {"type": "metric", "name": "accuracy", "params": {}},
            "group_key": "category_name",
            "name_template": "{group_name}_acc",
        }
        stderr = {
            "metric": {"type": "metric", "name": "stderr", "params": {}},
            "group_key": "category_name",
            "name_template": "{group_name}_sterr",
        }
        metrics = {
            "live_simple_acc": inspect_ai_log.EvalMetric(name="live_simple_acc", value=0.8, params=accuracy),
            "sql_acc": inspect_ai_log.EvalMetric(name="sql_acc", value=0.2, params=accuracy),
            "all": inspect_ai_log.EvalMetric(name="all", value=0.5, params=accuracy),
            "live_simple_sterr": inspect_ai_log.EvalMetric(name="live_simple_sterr", value=0.04, params=stderr),
            "sql_sterr": inspect_ai_log.EvalMetric(name="sql_sterr", value=0.03, params=stderr),
            "all2": inspect_ai_log.EvalMetric(name="all", value=0.05, params=stderr),
        }
        metrics.update({name: inspect_ai_log.EvalMetric(name=name, value=value) for name, value in ungrouped.items()})
        samples = [
            inspect_ai_log.EvalSample(
                id=f"{group}_0",
                epoch=1,
                input="q",
                target="t",
                metadata={"category_name": group},
                scores={"scorer": inspect_ai_scorer.Score(value=1.0)},
            )
            for group in ("live_simple", "sql")
        ]
        return inspect_ai_log.EvalLog(
            eval=_eval_spec(),
            status="success",
            results=inspect_ai_log.EvalResults(
                scores=[inspect_ai_log.EvalScore(name="s", scorer="s", metrics=metrics)]
            ),
            samples=samples,
        )

    return build


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param("C", 1.0, id="letter-correct"),
        pytest.param("I", 0.0, id="letter-incorrect"),
        pytest.param("P", 0.5, id="letter-partial"),
        pytest.param(0.25, 0.25, id="float"),
        pytest.param(True, 1.0, id="bool"),
        pytest.param("unrecognized", None, id="unreadable-value-scores-nothing"),
    ],
)
def test_score_to_float_reads_letter_grades_and_numbers(value: object, expected: float | None):
    """InspectAI scorers report a sample's score as a letter grade as often as a
    number, and both have to reach the trace as one comparable float."""
    assert log.score_to_float(value) == expected


def test_parse_log_degrades_for_a_foreign_log_object():
    """InspectAI's schema is a moving target, so a log missing every expected field
    degrades to an empty partial result rather than raising."""
    result, traces = log.parse_log(object(), "offline")
    assert result.status == "partial"
    assert result.metrics == {}
    assert traces == []


@pytest.mark.parametrize(
    ("run_status", "expected_status"),
    [
        pytest.param("success", "ok", id="finished-run-is-complete"),
        pytest.param("error", "partial", id="errored-run-stays-partial"),
        pytest.param("cancelled", "partial", id="cancelled-run-stays-partial"),
    ],
)
def test_parse_log_status_says_whether_the_run_finished(run_status: str, expected_status: str, scored_log):
    """A run that stopped early still carries the scorer's metrics, so only the status
    separates that score from a complete one."""
    result, _ = log.parse_log(scored_log(status=run_status, accuracy=0.9), "offline")
    assert result.status == expected_status
    assert result.metrics["accuracy"] == 0.9


def test_grouped_metrics_are_reported_per_subset(grouped_log):
    """InspectAI flattens one metric per group into the scorer's metric table, which
    would otherwise bury the headline metrics under every subset of the dataset."""
    result, _ = log.parse_log(grouped_log(ungrouped={"accuracy": 0.5, "stderr": 0.05}), "offline")
    assert result.metrics == {"accuracy": 0.5, "stderr": 0.05}
    assert result.subset_metrics == {
        "live_simple": {"accuracy": 0.8, "stderr": 0.04},
        "sql": {"accuracy": 0.2, "stderr": 0.03},
    }


def test_a_grouped_aggregate_is_the_topline_when_nothing_reports_it_ungrouped(grouped_log):
    """A scorer can report its metrics grouped and nothing else, and the run still
    has to have a headline score rather than reading as unscored."""
    result, _ = log.parse_log(grouped_log(ungrouped={}), "offline")
    assert result.topline_metric_name == "accuracy"
    assert result.metrics == {"accuracy": 0.5, "stderr": 0.05}
    assert result.status == "ok"


def test_a_group_template_without_fixed_text_still_names_each_subset():
    """InspectAI's default `name_template` is the group name alone, so a group has to
    survive a template carrying no text around it."""
    params = {
        "metric": {"type": "metric", "name": "accuracy", "params": {}},
        "group_key": "category_name",
        "name_template": "{group_name}",
    }
    metrics = {
        "live_simple": inspect_ai_log.EvalMetric(name="live_simple", value=0.8, params=params),
        "sql": inspect_ai_log.EvalMetric(name="sql", value=0.2, params=params),
        "all": inspect_ai_log.EvalMetric(name="all", value=0.5, params=params),
    }
    record = inspect_ai_log.EvalLog(
        eval=_eval_spec(),
        status="success",
        results=inspect_ai_log.EvalResults(scores=[inspect_ai_log.EvalScore(name="s", scorer="s", metrics=metrics)]),
    )
    result, _ = log.parse_log(record, "offline")
    assert result.subset_metrics == {"live_simple": {"accuracy": 0.8}, "sql": {"accuracy": 0.2}}
    assert result.metrics == {"accuracy": 0.5}


def test_a_trace_names_the_group_it_was_scored_in(grouped_log):
    """A row's subset has to name the group whose per-subset metric covers it, not the
    sample id, or the rows cannot be read against the subset table."""
    _, traces = log.parse_log(grouped_log(ungrouped={}), "offline")
    assert [trace.subset for trace in traces] == ["live_simple", "sql"]


def test_a_trace_has_no_subset_when_the_scorer_grouped_by_several_keys():
    """Grouped by two keys, the subset table spans two dimensions and one name cannot
    place a row in both, so naming it by either would be arbitrary."""
    metrics = {
        "math_acc": inspect_ai_log.EvalMetric(
            name="math_acc",
            value=0.8,
            params={
                "metric": {"type": "metric", "name": "accuracy", "params": {}},
                "group_key": "category",
                "name_template": "{group_name}_acc",
            },
        ),
        "easy_rate": inspect_ai_log.EvalMetric(
            name="easy_rate",
            value=0.6,
            params={
                "metric": {"type": "metric", "name": "accuracy", "params": {}},
                "group_key": "difficulty",
                "name_template": "{group_name}_rate",
            },
        ),
    }
    record = inspect_ai_log.EvalLog(
        eval=_eval_spec(),
        status="success",
        results=inspect_ai_log.EvalResults(scores=[inspect_ai_log.EvalScore(name="s", scorer="s", metrics=metrics)]),
        samples=[
            inspect_ai_log.EvalSample(
                id="s0",
                epoch=1,
                input="q",
                target="t",
                metadata={"category": "math", "difficulty": "easy"},
                scores={"scorer": inspect_ai_scorer.Score(value=1.0)},
            )
        ],
    )
    result, traces = log.parse_log(record, "mixed")
    assert result.subset_metrics == {"math": {"accuracy": 0.8}, "easy": {"accuracy": 0.6}}
    assert traces[0].subset is None


def test_a_trace_has_no_subset_when_the_scorer_grouped_nothing(sampled_log):
    """A sample id is not a subset: a run whose scorer grouped nothing has to leave
    the field unset rather than label every row with its own id."""
    _, traces = log.parse_log(sampled_log(status="success", scores=(1.0, 0.0)), "gpqa")
    assert [trace.subset for trace in traces] == [None, None]


def test_a_trace_keeps_its_sample_id_and_epoch_whatever_its_subset_names(grouped_log, sampled_log):
    """Every row records its sample id and epoch, grouped or not."""
    _, grouped = log.parse_log(grouped_log(ungrouped={}), "offline")
    _, ungrouped = log.parse_log(sampled_log(status="success", scores=(1.0, 0.0)), "gpqa")
    assert [(trace.sample_id, trace.epoch, trace.subset) for trace in grouped] == [
        ("live_simple_0", 1, "live_simple"),
        ("sql_0", 1, "sql"),
    ]
    assert [(trace.sample_id, trace.epoch, trace.subset) for trace in ungrouped] == [("0", 1, None), ("1", 1, None)]


def test_metrics_recomputed_from_samples_when_a_run_stopped_early(sampled_log):
    """A run cut short never reaches InspectAI's scoring step, so the topline comes
    from the samples that did finish and carry their own scores."""
    result, _ = log.parse_log(sampled_log(status="error", scores=(1.0, 0.0, 1.0, 1.0)), "offline")
    assert result.status == "partial"
    assert result.metrics["accuracy"] == 0.75
    assert result.metrics["stderr"] > 0


def test_metrics_recompute_credits_an_unscored_sample_zero(sampled_log):
    """The recomputed topline covers every attempted sample, so an unscorable one
    lowers the score instead of shrinking the denominator; its own trace stays
    ungraded rather than scored zero."""
    result, traces = log.parse_log(sampled_log(status="error", scores=(1.0, 1.0, "unrecognized", 1.0)), "offline")
    assert result.metrics["accuracy"] == 0.75
    assert (traces[2].score, traces[2].error_type) == (None, "ungraded")
    assert result.num_error == 1


def test_metrics_stay_empty_when_no_sample_scored(sampled_log):
    """A run where nothing was scored is a harness failure, not a model scoring
    zero, so the recompute reports no metrics at all."""
    result, _ = log.parse_log(sampled_log(status="error", scores=("unrecognized",)), "offline")
    assert result.metrics == {}
    assert result.status == "partial"


@pytest.mark.parametrize(
    ("messages", "expected_roles"),
    [
        pytest.param(
            [
                inspect_ai_model.ChatMessageSystem(content="sys"),
                inspect_ai_model.ChatMessageUser(content="q"),
                inspect_ai_model.ChatMessageAssistant(content="a"),
            ],
            ["system", "user"],
            id="graded-reply-dropped",
        ),
        pytest.param(
            [
                inspect_ai_model.ChatMessageUser(content="q1"),
                inspect_ai_model.ChatMessageAssistant(content="a1"),
                inspect_ai_model.ChatMessageUser(content="q2"),
                inspect_ai_model.ChatMessageAssistant(content="a2"),
            ],
            ["user", "assistant", "user"],
            id="interior-turns-kept",
        ),
    ],
)
def test_messages_are_the_request_without_the_graded_reply(messages: list[object], expected_roles: list[str]):
    """`Trace.messages` is the request a report renders as the prompt: the reply being
    graded belongs in `Trace.content`, while earlier turns are part of the request."""
    assert [message["role"] for message in log.messages_to_dicts(messages)] == expected_roles


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        pytest.param(
            [
                inspect_ai_model.ChatMessageAssistant(
                    content=[
                        inspect_ai_model.ContentReasoning(reasoning="thinking hard"),
                        inspect_ai_model.ContentText(text="ANSWER: 277"),
                    ]
                )
            ],
            ("ANSWER: 277", "thinking hard", None),
            id="reasoning-split-from-answer",
        ),
        pytest.param(
            [inspect_ai_model.ChatMessageAssistant(content="plain reply")],
            ("plain reply", None, None),
            id="plain-string-content",
        ),
        pytest.param(
            [
                inspect_ai_model.ChatMessageAssistant(
                    content=[],
                    tool_calls=[inspect_ai_tool.ToolCall(id="c1", function="get_weather", arguments={"city": "Paris"})],
                )
            ],
            ("", None, [{"id": "c1", "function": "get_weather", "arguments": {"city": "Paris"}}]),
            id="tool-calls-are-the-answer",
        ),
        pytest.param(
            [inspect_ai_model.ChatMessageUser(content="q")],
            ("", None, None),
            id="no-assistant-turn",
        ),
    ],
)
def test_assistant_output_reads_the_graded_reply(messages: list[object], expected: tuple[object, ...]):
    """A reasoning model returns its thinking as a separate content part and a
    function-calling model answers with tool calls alone, so the graded reply is never
    simply the message text."""
    assert log.assistant_output(messages) == expected


def test_dataset_result_defaults_to_complete():
    """A result is `partial` only because a run said so, so the default has to be the
    complete one: a mislabelled partial score would read as a real number."""
    assert records.DatasetResult(dataset="x", topline_metric_name="accuracy").status == "ok"


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
                model_kwargs={"extra_body": {"custom_option": True}},
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
    assert config.extra_body == {"custom_option": True}


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
    ids = [trace.subset for trace in first]
    assert len(ids) == 4 and ids == [trace.subset for trace in second] and ids != ["0", "1", "2", "3"]


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


def test_model_kwargs_merge_into_the_generate_options(tmp_path: pathlib.Path):
    request = base.RunRequest(
        base_url="http://127.0.0.1:1/v1",
        model_name="model",
        run_dir=tmp_path,
        settings=records.EvalSettings(
            temperature=0.8,
            model_kwargs={"top_p": 0.9, "extra_body": {"custom_option": False}},
            harness_kwargs={"temperature": 0.2},
        ),
    )
    kwargs = runner.InspectAIRunner("offline").build_eval_kwargs(request)
    # The run's model kwargs merge per key over the generate-config defaults;
    # explicit `--harness-kwargs` still wins verbatim on a name collision.
    assert kwargs["top_p"] == 0.9
    assert kwargs["extra_body"] == {"custom_option": False}
    assert kwargs["temperature"] == 0.2


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
