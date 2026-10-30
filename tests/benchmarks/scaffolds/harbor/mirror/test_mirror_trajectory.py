"""Mirror journal parsing and conversion to Harbor's native trajectory format."""

import json
import pathlib

import yaml

from beam_verifier.benchmarks.scaffolds.harbor.mirror import agent as mirror_agent
from beam_verifier.benchmarks.scaffolds.harbor.mirror import trajectory as mirror_trajectory

_JOURNALS = yaml.safe_load((pathlib.Path(__file__).parent / "fixtures/journals.yaml").read_text())

JOURNAL = _JOURNALS["tool_turns"]


def test_journal_orders_turns_and_attaches_tool_results() -> None:
    parsed = mirror_trajectory.parse(json.dumps(record) for record in JOURNAL)
    assert [turn.source for turn in parsed.turns] == ["system", "user", "agent", "agent"]
    (call,) = parsed.turns[2].tool_calls
    assert (call.name, call.arguments) == ("bash", {"command": "ls"})
    assert [result.content for result in parsed.turns[2].tool_results] == ["README.md", "stray"]
    assert parsed.usage.prompt_tokens == 120
    (compaction,) = parsed.compactions
    assert compaction.summary == "Listed the files."
    assert compaction.trigger.endswith("compacting")


def test_journal_counts_a_partial_line_instead_of_failing() -> None:
    parsed = mirror_trajectory.parse(['{"chat": {"user": {"content": "hi"}}}', '{"chat": {"user'])
    assert (len(parsed.turns), parsed.unreadable_lines) == (1, 1)


def test_journal_counts_summary_usage_once_without_recounting_retained_turns() -> None:
    records = _JOURNALS["counts_summary_usage_once_without_recounting_retained_turns"]

    parsed = mirror_trajectory.parse(json.dumps(record) for record in records)

    assert parsed.usage == mirror_trajectory.Usage(prompt_tokens=130, completion_tokens=13, cached_tokens=47)
    steps = mirror_agent.atif_steps(parsed)
    assert [step.metrics for step in steps if step.is_copied_context] == [None, None]


def test_journal_preserves_background_results_and_extension_messages_in_order() -> None:
    records = _JOURNALS["preserves_background_results_and_extension_messages_in_order"]

    steps = mirror_agent.atif_steps(mirror_trajectory.parse(json.dumps(record) for record in records))

    assert [(step.source, step.message) for step in steps] == [
        ("user", "Run the background job."),
        ("agent", "The job is running."),
        ("user", '<task_result id="job-1" tool="bash" status="succeeded">\nJob completed.\n</task_result>'),
        ("user", "Check the generated file."),
        ("agent", "The file is correct."),
    ]


def test_journal_keeps_tool_results_across_background_notifications() -> None:
    parsed = mirror_trajectory.parse(
        json.dumps(record)
        for record in [
            JOURNAL[1],
            {"taskResult": {"taskId": "background", "toolName": "bash", "success": True, "content": "done"}},
            JOURNAL[2],
        ]
    )
    assert parsed.turns[0].tool_results == (mirror_trajectory.ToolResult(call_id="call_1", content="README.md"),)
    assert parsed.turns[1].source == "user"
    assert parsed.orphaned_tool_results == 0
