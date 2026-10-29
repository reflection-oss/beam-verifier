"""Records for runs and traces (the data beam-verifier saves and views)."""

import json
from typing import Any, Literal

import pydantic

from beam_verifier.benchmarks import prompts

# Pytest outcomes that do not fail a suite. Shared by the report aggregation
# and every renderer, so they can never disagree on what counts as a failure.
BENIGN_OUTCOMES = ("passed", "skipped", "xfailed", "xpassed")


class Trace(pydantic.BaseModel):
    """A full per-row record: request + response + grading. One JSONL line per row."""

    index: int
    sample_id: str | None = None
    epoch: int | None = None
    request_hash: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    expected_answer: Any = None
    # response
    content: str = ""
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    # grading
    score: float | None = None
    parsed_pred: Any = None
    subset: str | None = None


# Serialized to run.json via the parent RunManifest's model_dump_json.
class DatasetResult(pydantic.BaseModel):  # pydantic: justified
    dataset: str
    topline_metric_name: str
    status: str = pydantic.Field(
        default="ok",
        description='"ok" when the run completed and reported its own aggregate metrics; '
        '"partial" when the run was cut short and metrics cover only the '
        'samples the harness evaluated; "failed" when the dataset was attempted but produced '
        "no score at all. Neither a partial nor a failed row may be read as a full one.",
    )
    error: str | None = pydantic.Field(
        default=None,
        description='Why the dataset produced no score; set only on a "failed" row.',
    )
    metrics: dict[str, float] = {}
    subset_metrics: dict[str, dict[str, float]] | None = None
    n: int = pydantic.Field(default=0, description="Attempted samples: every trace the run produced, scored or not.")
    num_ok: int = pydantic.Field(default=0, description="Samples with a usable outcome; reported as scored.")
    num_error: int = pydantic.Field(default=0, description="Samples without a usable outcome (errored or ungraded).")
    finish_reasons: dict[str, int] = {}
    error_types: dict[str, int] = {}


# Serialized to run.json via the parent RunManifest's model_dump_json.
class SuiteResult(pydantic.BaseModel):  # pydantic: justified
    """Aggregated outcome of one functional-test suite."""

    suite: str  # directory name under functional_tests/
    display_name: str
    required: bool
    status: str  # "PASS" | "FAIL" | "SKIP"
    counts: dict[str, int] = {}  # outcome counts within the suite
    failing_ids: list[str] = []  # exact node ids of failing cases
    skip_reason: str | None = None


# Serialized to run.json via the parent RunManifest's model_dump_json.
class VerifyResult(pydantic.BaseModel):  # pydantic: justified
    suites: list[SuiteResult] = []
    evidence: list[str] = []  # suites that actually ran (status != SKIP)
    functional: dict[str, int] = {}  # passed/failed/skipped/xfailed/... counts
    functional_cases: list[dict[str, Any]] = []  # {suite, name, outcome}
    passed: bool = False


# Serialized to run.json via the parent RunManifest's model_dump_json.
class EvalSettings(pydantic.BaseModel):  # pydantic: justified
    """Eval options; unset fields inherit dataset defaults, null defers to the harness."""

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    reasoning_effort: str | None = None
    prompt_factory: prompts.BeamPromptFactory | None = None
    num_examples: int | None = None
    parallel_tasks: int | None = None
    client_timeout: int = 600
    harness_kwargs: dict[str, object] = {}
    model_kwargs: dict[str, object] = {}
    sandbox: Literal["docker", "daytona"] = "docker"
    host_model_proxy: bool = False
    parse_url: str | None = None
    format_url: str | None = None
    completions_url: str | None = None


class RunManifest(pydantic.BaseModel):
    """The saved record of one run: what ran, with which settings, and its results."""

    run_id: str
    kind: str = pydantic.Field(description='The command that produced the run: "eval" or "verify".')
    created_at: str
    finished_at: str | None = None
    duration_s: float | None = None
    version: str
    model: str
    base_url: str
    config: EvalSettings = EvalSettings()
    dataset_settings: dict[str, EvalSettings] = pydantic.Field(
        default={}, description="Resolved settings for each dataset, pinned before the first evaluation."
    )
    resume_overrides: list[dict[str, object]] = pydantic.Field(
        default=[], description="Settings each --resume named again, oldest first."
    )
    dataset_ids: list[str] = []
    dataset_results: list[DatasetResult] = []
    verify: VerifyResult | None = None


def turn_text(turn: dict[str, Any]) -> str:
    """A turn's display text: its content, then one `name(arguments)` line per tool call.

    A function-calling turn often carries no text at all, the response being the
    calls themselves, so a renderer showing `content` alone shows nothing for it.

    Args:
      turn: A message dict (`content`, optional `tool_calls`).

    Returns:
      The content and calls; empty when the turn produced neither.
    """
    calls = "\n".join(
        f"{call.get('function', '')}({json.dumps(call.get('arguments') or {}, default=str)})"
        for call in turn.get("tool_calls") or []
    )
    return "\n".join(part for part in (str(turn.get("content") or ""), calls) if part)


def answer_text(trace: Trace) -> str:
    """The answer a report shows for a row: its text content and its tool calls.

    Args:
      trace: The row.

    Returns:
      The content, then one `name(arguments)` line per tool call; empty when the
      turn produced neither.
    """
    return turn_text({"content": trace.content, "tool_calls": trace.tool_calls})


def trace_failed(trace: Trace) -> bool:
    """Reports whether a row should be rendered as a failure.

    The score decides when the row was graded; the error decides only for
    an ungraded row.

    Args:
      trace: The row.

    Returns:
      True when the row failed.
    """
    if trace.score is not None:
        return trace.score == 0.0
    return trace.error_type is not None
