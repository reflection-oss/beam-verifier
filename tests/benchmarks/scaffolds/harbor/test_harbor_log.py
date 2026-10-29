"""Check that reading Harbor results preserves its metrics, rewards, and errors."""

import datetime
import pathlib
import uuid

import pytest
from harbor.models.job import result as job_result
from harbor.models.task import id as task_id
from harbor.models.trial import config as trial_config
from harbor.models.trial import result as trial_result
from harbor.models.verifier import result as verifier_result

from beam_verifier.benchmarks.scaffolds.harbor import log


@pytest.fixture
def trial() -> trial_result.TrialResult:
    return trial_result.TrialResult(
        task_name="task",
        trial_name="attempt",
        trial_uri="file:///trial",
        task_id=task_id.LocalTaskId(path=pathlib.Path("task")),
        task_checksum="fixture",
        config=trial_config.TrialConfig(task=trial_config.TaskConfig(path=pathlib.Path("task"))),
        agent_info=trial_result.AgentInfo(name="mirror", version="0.1.8"),
        verifier_result=verifier_result.VerifierResult(rewards={"reward": 1.0}),
    )


def write_job(directory: pathlib.Path, trial: trial_result.TrialResult, *, total: int = 1) -> None:
    """Write Harbor result files whose job metrics differ from the trial reward."""
    (directory / "attempt").mkdir()
    (directory / "attempt" / "result.json").write_text(trial.model_dump_json())
    stats = job_result.JobStats.from_trial_results([trial], n_total_trials=total)
    evaluation = next(iter(stats.evals.values()))
    evaluation.metrics = [{"mean": 0.75}]
    evaluation.pass_at_k = {2: 0.9}
    result = job_result.JobResult(
        id=uuid.UUID(int=1),
        started_at=datetime.datetime(2026, 1, 1),
        finished_at=datetime.datetime(2026, 1, 1),
        n_total_trials=total,
        stats=stats,
    )
    (directory / "result.json").write_text(result.model_dump_json())


@pytest.mark.parametrize(
    "journal, message",
    [
        (b"", ""),
        (
            '{"chat":{"assistant":{"content":"First\u2028second\u2029third"}}}\n'.encode(),
            "First\u2028second\u2029third",
        ),
        (b'{"chat":{"assistant":{"content":"Bad byte: \xff"}}}\n', "Bad byte: \ufffd"),
    ],
)
def test_parse_job_preserves_native_metrics(
    tmp_path: pathlib.Path, trial: trial_result.TrialResult, journal: bytes, message: str
) -> None:
    write_job(tmp_path, trial)
    journal_path = tmp_path / "attempt" / "agent" / "mirror.jsonl"
    journal_path.parent.mkdir()
    journal_path.write_bytes(journal)
    outcome = log.parse_job(tmp_path, "coding")
    assert outcome.result.metrics == {"mean": 0.75, "pass@2": 0.9}
    assert outcome.result.status == "ok"
    assert outcome.result.num_ok == 1
    assert outcome.traces[0].score == 1.0
    assert outcome.traces[0].content == message


def test_parse_job_preserves_timeout_with_passing_reward(
    tmp_path: pathlib.Path, trial: trial_result.TrialResult
) -> None:
    trial.exception_info = trial_result.ExceptionInfo(
        exception_type="AgentTimeoutError",
        exception_message="deadline",
        exception_traceback="",
        occurred_at=datetime.datetime(2026, 1, 1),
    )
    write_job(tmp_path, trial)
    outcome = log.parse_job(tmp_path, "coding")
    assert outcome.traces[0].score == 1.0
    assert outcome.traces[0].error_type == "AgentTimeoutError"
    assert outcome.result.num_ok == 1
    assert outcome.result.num_error == 0


def test_parse_job_marks_missing_trials_partial(tmp_path: pathlib.Path, trial: trial_result.TrialResult) -> None:
    write_job(tmp_path, trial, total=2)
    outcome = log.parse_job(tmp_path, "coding")
    assert outcome.result.status == "partial"
    assert outcome.warning


def test_parse_job_rejects_ungraded_result(tmp_path: pathlib.Path, trial: trial_result.TrialResult) -> None:
    trial.verifier_result = None
    write_job(tmp_path, trial)
    with pytest.raises(ValueError, match="no graded result"):
        log.parse_job(tmp_path, "coding")


def test_archive_ungraded_trials_rejects_corrupt_results(tmp_path: pathlib.Path) -> None:
    (tmp_path / "attempt").mkdir()
    (tmp_path / "attempt" / "result.json").write_text("broken JSON")
    with pytest.raises(ValueError):
        log.archive_ungraded_trials(tmp_path)
    assert (tmp_path / "attempt" / "result.json").read_text() == "broken JSON"


def test_parse_job_retains_tools_reasoning_and_observations(
    tmp_path: pathlib.Path, trial: trial_result.TrialResult
) -> None:
    write_job(tmp_path, trial)
    agent_dir = tmp_path / "attempt" / "agent"
    agent_dir.mkdir()
    fixture = pathlib.Path(__file__).parent / "fixtures" / "mirror_tool_journal.jsonl"
    (agent_dir / "mirror.jsonl").write_bytes(fixture.read_bytes())
    trace = log.parse_job(tmp_path, "coding").traces[0]
    assert trace.reasoning_content == "Check the workspace."
    assert trace.tool_calls == [{"id": "call_1", "function": "bash", "arguments": {"command": "pwd"}}]
    assert trace.messages[-1] == {"role": "tool", "tool_call_id": "call_1", "content": "/app"}
    assert trace.messages[1]["tool_calls"] == trace.tool_calls
