"""Translate offline fixtures validated with InspectAI's native log types."""

import pathlib
from typing import Any

import pytest
import yaml
from inspect_ai import log as inspect_ai_log
from inspect_ai import scorer as inspect_ai_scorer

from beam_verifier.benchmarks.scaffolds.inspect_ai import log

_EXAMPLES = yaml.safe_load((pathlib.Path(__file__).parent / "fixtures/logs.yaml").read_text())


@pytest.fixture
def native_logs() -> dict[str, inspect_ai_log.EvalLog]:
    examples = yaml.safe_load((pathlib.Path(__file__).parent / "fixtures/logs.yaml").read_text())
    return {
        name: inspect_ai_log.EvalLog.model_validate({"eval": examples["eval"], **value})
        for name, value in examples["logs"].items()
    }


@pytest.fixture
def sampled_log(native_logs: dict[str, inspect_ai_log.EvalLog]):
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
        return inspect_ai_log.EvalLog(eval=native_logs["empty"].eval, status=status, samples=samples)

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
def test_parse_log_preserves_letter_grades_and_numeric_scores(
    value: object, expected: float | None, sampled_log
) -> None:
    """InspectAI scorers report a sample's score as a letter grade as often as a
    number, and both have to reach the trace as one comparable float."""
    _, traces = log.parse_log(sampled_log(status="success", scores=(value,)), "offline")
    assert traces[0].score == expected


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
def test_parse_log_status_says_whether_the_run_finished(
    run_status: str, expected_status: str, native_logs: dict[str, inspect_ai_log.EvalLog]
):
    """A run that stopped early still carries the scorer's metrics, so only the status
    separates that score from a complete one."""
    record = native_logs["scored"]
    record.status = run_status
    result, _ = log.parse_log(record, "offline")
    assert result.status == expected_status
    assert result.metrics["accuracy"] == 0.9


def test_grouped_metrics_are_reported_per_subset(native_logs: dict[str, inspect_ai_log.EvalLog]) -> None:
    """InspectAI flattens one metric per group into the scorer's metric table, which
    would otherwise bury the headline metrics under every subset of the dataset."""
    record = native_logs["grouped"]
    assert record.results is not None
    record.results.scores[0].metrics.update(
        accuracy=inspect_ai_log.EvalMetric(name="accuracy", value=0.5),
        stderr=inspect_ai_log.EvalMetric(name="stderr", value=0.05),
    )
    result, _ = log.parse_log(record, "offline")
    assert result.metrics == {"accuracy": 0.5, "stderr": 0.05}
    assert result.subset_metrics == {
        "live_simple": {"accuracy": 0.8, "stderr": 0.04},
        "sql": {"accuracy": 0.2, "stderr": 0.03},
    }


def test_a_grouped_aggregate_is_the_topline_when_nothing_reports_it_ungrouped(
    native_logs: dict[str, inspect_ai_log.EvalLog],
) -> None:
    """A scorer can report its metrics grouped and nothing else, and the run still
    has to have a headline score rather than reading as unscored."""
    result, _ = log.parse_log(native_logs["grouped"], "offline")
    assert result.topline_metric_name == "accuracy"
    assert result.metrics == {"accuracy": 0.5, "stderr": 0.05}
    assert result.status == "ok"


def test_a_group_template_without_fixed_text_still_names_each_subset(
    native_logs: dict[str, inspect_ai_log.EvalLog],
) -> None:
    """InspectAI's default `name_template` is the group name alone, so a group has to
    survive a template carrying no text around it."""
    record = native_logs["bare_group_names"]
    result, _ = log.parse_log(record, "offline")
    assert result.subset_metrics == {"live_simple": {"accuracy": 0.8}, "sql": {"accuracy": 0.2}}
    assert result.metrics == {"accuracy": 0.5}


def test_a_trace_has_no_subset_when_the_scorer_grouped_by_several_keys(
    native_logs: dict[str, inspect_ai_log.EvalLog],
) -> None:
    """Grouped by two keys, the subset table spans two dimensions and one name cannot
    place a row in both, so naming it by either would be arbitrary."""
    record = native_logs["multiple_group_keys"]
    result, traces = log.parse_log(record, "mixed")
    assert result.subset_metrics == {"math": {"accuracy": 0.8}, "easy": {"accuracy": 0.6}}
    assert traces[0].subset is None


def test_a_trace_keeps_its_sample_id_and_epoch_whatever_its_subset_names(
    native_logs: dict[str, inspect_ai_log.EvalLog], sampled_log
) -> None:
    """Every row records its sample id and epoch, grouped or not."""
    _, grouped = log.parse_log(native_logs["grouped"], "offline")
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


@pytest.mark.parametrize("case", _EXAMPLES["requests"], ids=[case["id"] for case in _EXAMPLES["requests"]])
def test_messages_are_the_request_without_the_graded_reply(
    case: dict[str, Any], native_logs: dict[str, inspect_ai_log.EvalLog]
) -> None:
    """`Trace.messages` is the request a report renders as the prompt: the reply being
    graded belongs in `Trace.content`, while earlier turns are part of the request."""
    record = native_logs["empty"]
    record.samples = [inspect_ai_log.EvalSample(id=1, epoch=1, input="q", target="t", messages=case["messages"])]
    _, traces = log.parse_log(record, "offline")
    assert [message["role"] for message in traces[0].messages] == case["expected"]


@pytest.mark.parametrize("case", _EXAMPLES["replies"], ids=[case["id"] for case in _EXAMPLES["replies"]])
def test_assistant_output_reads_the_graded_reply(
    case: dict[str, Any], native_logs: dict[str, inspect_ai_log.EvalLog]
) -> None:
    """A reasoning model returns its thinking as a separate content part and a
    function-calling model answers with tool calls alone, so the graded reply is never
    simply the message text."""
    record = native_logs["empty"]
    record.samples = [inspect_ai_log.EvalSample(id=1, epoch=1, input="q", target="t", messages=case["messages"])]
    _, traces = log.parse_log(record, "offline")
    trace = traces[0]
    assert [trace.content, trace.reasoning_content, trace.tool_calls] == case["expected"]
