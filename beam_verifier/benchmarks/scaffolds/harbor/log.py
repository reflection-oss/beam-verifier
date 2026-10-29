"""Read Harbor results into Beam Verifier's result and trace records."""

import collections
import pathlib
import tempfile

from harbor.models.job import result as job_result
from harbor.models.trial import result as trial_result

from beam_verifier import records
from beam_verifier.benchmarks.scaffolds import base
from beam_verifier.benchmarks.scaffolds.harbor.mirror import trajectory
from beam_verifier.reporting import run_store


def parse_job(job_dir: pathlib.Path, dataset_id: str) -> base.RunOutcome:
    """Read Harbor's job metrics, trial rewards, and errors.

    Keep both the reward and error when an agent times out but is still graded.
    Mark missing or ungraded trials as incomplete; reject malformed results.

    Args:
        job_dir: Harbor output directory for this job.
        dataset_id: Dataset name recorded in the returned result.

    Returns:
        Metrics and traces, with a warning if the job is incomplete.

    Raises:
        ValueError: If results are malformed, have no graded trials or metrics,
            or do not contain exactly one agent/dataset pair.
        OSError: If a result cannot be read.
    """
    result = job_result.JobResult.model_validate_json((job_dir / "result.json").read_text())
    if len(result.stats.evals) != 1:
        raise ValueError("A Harbor dataset runner requires results from exactly one agent/dataset pair")
    evaluation = next(iter(result.stats.evals.values()))
    metrics = {name: float(value) for metric in evaluation.metrics for name, value in metric.items()}
    metrics.update({f"pass@{k}": value for k, value in evaluation.pass_at_k.items()})
    traces = [
        _trace(trial_result.TrialResult.model_validate_json(path.read_text()), path.parent, index)
        for index, path in enumerate(sorted(job_dir.glob("*/result.json")))
    ]
    num_ok = sum(trace.score is not None for trace in traces)
    if not num_ok or not metrics:
        raise ValueError("Harbor wrote no graded result with aggregate metrics")
    complete = result.finished_at is not None and len(traces) == result.n_total_trials and num_ok == len(traces)
    return base.RunOutcome(
        result=records.DatasetResult(
            dataset=dataset_id,
            topline_metric_name="mean" if "mean" in metrics else next(iter(metrics)),
            status="ok" if complete else "partial",
            metrics=metrics,
            n=len(traces),
            num_ok=num_ok,
            num_error=len(traces) - num_ok,
            error_types=dict(collections.Counter(trace.error_type for trace in traces if trace.error_type)),
        ),
        traces=traces,
        warning=None if complete else "Harbor results are incomplete or contain ungraded trials",
    )


def _trace(trial: trial_result.TrialResult, trial_dir: pathlib.Path, index: int) -> records.Trace:
    journal_path = trial_dir / "agent" / "mirror.jsonl"
    journal = trajectory.parse(
        journal_path.read_text(encoding="utf-8", errors="replace").split("\n") if journal_path.is_file() else []
    )
    messages = _messages(journal)
    instruction = next(
        (message for message in messages if message["role"] == "user"), {"role": "user", "content": trial.task_name}
    )
    reward = (
        trial.verifier_result.rewards.get("reward") if trial.verifier_result and trial.verifier_result.rewards else None
    )
    exception = trial.exception_info
    last_agent = next((turn for turn in reversed(journal.turns) if turn.source == "agent"), None)
    duration = (
        (trial.finished_at - trial.started_at).total_seconds()
        if trial.finished_at is not None and trial.started_at is not None
        else None
    )
    return records.Trace(
        index=index,
        sample_id=trial.trial_name,
        request_hash=run_store.request_hash([instruction]),
        messages=messages,
        content=last_agent.message if last_agent else "",
        reasoning_content=last_agent.reasoning_content if last_agent else None,
        tool_calls=[_tool_call(call) for call in last_agent.tool_calls] if last_agent else None,
        latency_ms=round(duration * 1000) if duration is not None else None,
        score=reward,
        subset=trial.task_name,
        prompt_tokens=trial.agent_result.n_input_tokens if trial.agent_result else None,
        completion_tokens=trial.agent_result.n_output_tokens if trial.agent_result else None,
        error_type=exception.exception_type if exception else ("ungraded" if reward is None else None),
        error_message=exception.exception_message if exception else None,
    )


def _tool_call(call: trajectory.ToolCall) -> dict[str, object]:
    return {"id": call.call_id, "function": call.name, "arguments": dict(call.arguments)}


def _messages(journal: trajectory.Journal) -> list[dict[str, object]]:
    """Retain tool calls, observations and reasoning in the shared report schema."""
    messages: list[dict[str, object]] = []
    for turn in journal.turns:
        message: dict[str, object] = {
            "role": "assistant" if turn.source == "agent" else turn.source,
            "content": turn.message,
        }
        if turn.reasoning_content is not None:
            message["reasoning_content"] = turn.reasoning_content
        if turn.tool_calls:
            message["tool_calls"] = [_tool_call(call) for call in turn.tool_calls]
        messages.append(message)
        messages.extend(
            {"role": "tool", "tool_call_id": result.call_id, "content": result.content} for result in turn.tool_results
        )
    return messages


def archive_ungraded_trials(job_dir: pathlib.Path) -> None:
    """Preserve unfinished artifacts outside the job so native resume retries them.

    Harbor reuses every saved result, including cancellations. Move only attempts
    without a grading reward; a zero reward or a graded timeout stays completed.
    Malformed results fail visibly instead of risking a repeat of a scored task.

    Args:
        job_dir: Existing Harbor job directory.

    Raises:
        ValueError: If an existing result is malformed.
        OSError: If artifacts cannot be read or moved.
    """
    archive: pathlib.Path | None = None
    for trial_dir in job_dir.iterdir():
        if not trial_dir.is_dir():
            continue
        result_path = trial_dir / "result.json"
        if result_path.is_file():
            trial = trial_result.TrialResult.model_validate_json(result_path.read_text())
            if (
                trial.verifier_result
                and trial.verifier_result.rewards
                and trial.verifier_result.rewards.get("reward") is not None
            ):
                continue
        if archive is None:
            archive = pathlib.Path(tempfile.mkdtemp(prefix=f"{job_dir.name}.interrupted-", dir=job_dir.parent))
        trial_dir.rename(archive / trial_dir.name)
