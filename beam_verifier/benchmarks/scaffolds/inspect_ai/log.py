"""Map an InspectAI `EvalLog` onto beam-verifier records."""

import collections
import math
import statistics
from typing import Any

from beam_verifier import records
from beam_verifier.reporting import run_store

# InspectAI's value_to_float maps unknown values to zero. Reports must keep
# unrecognized scores ungraded instead of turning them into model failures.
_SCORES_TO_FLOAT = {
    "C": 1.0,
    "I": 0.0,
    "P": 0.5,
    "N": 0.0,
}


def score_to_float(value: Any) -> float | None:
    """Normalize a raw `Score.value` to a float, or None if uninterpretable.

    Handles numeric scores and the `C`/`I`/`P` letter grades many scorers use.
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return _SCORES_TO_FLOAT.get(value.strip().upper())
    return None


def messages_to_dicts(messages: Any) -> list[dict[str, str]]:
    """Convert an InspectAI sample's messages into the request sent to the model.

    InspectAI stores the whole conversation, whereas `Trace.messages` is the request
    alone, so trailing assistant turns are dropped.

    Args:
      messages: Sample messages from the InspectAI log.

    Returns:
      JSON-serializable request messages.
    """
    out: list[dict[str, str]] = []
    for message in messages or []:
        role = getattr(message, "role", None) or "unknown"
        content = getattr(message, "text", None)
        if content is None:
            content = getattr(message, "content", "")
        out.append({"role": role, "content": content if isinstance(content, str) else str(content)})
    while out and out[-1]["role"] == "assistant":
        out.pop()
    return out


def assistant_output(messages: Any) -> tuple[str, str | None, list[dict[str, Any]] | None]:
    """Split the graded assistant turn into its answer, reasoning and tool calls.

    InspectAI stores an assistant message as typed content parts: a reasoning model's
    thinking arrives as a `reasoning` part and the worked answer as `text`. A
    function-calling answer may carry no text at all, the response being the tool
    calls themselves.

    Args:
      messages: Sample messages from the InspectAI log.

    Returns:
      The answer text, the reasoning text and the tool calls, the latter two None
      when absent.
    """
    last_assistant = None
    for message in messages or []:
        if getattr(message, "role", None) == "assistant":
            last_assistant = message
    if last_assistant is None:
        return "", None, None

    tool_calls = [
        {
            "id": getattr(call, "id", ""),
            "function": getattr(call, "function", ""),
            "arguments": getattr(call, "arguments", None) or {},
        }
        for call in getattr(last_assistant, "tool_calls", None) or []
    ] or None

    content = getattr(last_assistant, "content", None)
    if isinstance(content, str):
        return content, None, tool_calls

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    for part in content or []:
        reasoning = getattr(part, "reasoning", None)
        if reasoning:
            reasoning_parts.append(str(reasoning))
            continue
        text = getattr(part, "text", None)
        if text:
            text_parts.append(str(text))
    return "\n".join(text_parts), "\n".join(reasoning_parts) or None, tool_calls


def _usage_for_sample(sample: Any) -> tuple[int | None, int | None]:
    """Prompt and completion token counts for one `EvalSample`, each may be None."""
    usage = getattr(sample, "model_usage", None) or {}
    for entry in usage.values():
        return getattr(entry, "input_tokens", None), getattr(entry, "output_tokens", None)
    return None, None


_ATTACHMENT_PREFIX = "attachment://"


def _resolve_attachments(value: Any, attachments: dict[str, str]) -> Any:
    """Replaces InspectAI's `attachment://<id>` references (strings over 100 characters in a log)."""
    if isinstance(value, str) and value.startswith(_ATTACHMENT_PREFIX):
        return attachments.get(value[len(_ATTACHMENT_PREFIX) :], value)
    if isinstance(value, dict):
        return {k: _resolve_attachments(v, attachments) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_attachments(v, attachments) for v in value]
    return value


def request_tools(sample: Any) -> list[dict[str, Any]] | None:
    """The tool schemas of the sample's first model request, in OpenAI's `tools` format."""
    attachments = getattr(sample, "attachments", None) or {}
    for event in getattr(sample, "events", None) or []:
        if getattr(event, "event", None) == "model" and getattr(event, "tools", None):
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters.model_dump(exclude_none=True),
                    },
                }
                for tool in event.tools
            ]
            return _resolve_attachments(tools, attachments)
    return None


def parse_log(log: Any, dataset_id: str) -> tuple[records.DatasetResult, list[records.Trace]]:
    """Parse an InspectAI log into a DatasetResult plus per-sample Traces.

    The headline metric is `accuracy` when the log reports it, else the first
    metric the log reports. InspectAI's `grouped()` flattens one metric per group
    into the scorer's metric table, so those land in `subset_metrics` — keyed by
    group name and the grouped metric's own name — rather than beside the headline
    metrics, and each sample's group names its trace's subset. A scorer that groups
    nothing — or groups by more than one key — leaves the subset unset, rather than
    labelling a sample id or one arbitrary dimension as a subset.

    Args:
      log: The run's `EvalLog`.
      dataset_id: Dataset id, used as the result and trace-file name.

    Returns:
      The result and its traces.
    """
    metrics: dict[str, float] = {}
    subset_metrics: dict[str, dict[str, float]] = {}
    group_keys: set[str] = set()
    for scorer in getattr(getattr(log, "results", None), "scores", None) or []:
        for metric_name, metric in (getattr(scorer, "metrics", None) or {}).items():
            value = float(getattr(metric, "value", 0.0) or 0.0)
            params = getattr(metric, "params", None) or {}
            template = params.get("name_template")
            if not template:
                metrics[str(metric_name)] = value
                continue
            group_keys.add(str(params.get("group_key")))
            grouped_name = str((params.get("metric") or {}).get("name") or metric_name)
            # `metric.name` drops the suffix the log appends to a colliding metric key.
            name = str(getattr(metric, "name", metric_name))
            prefix, _, suffix = str(template).partition("{group_name}")
            if name == params.get("all_label", "all"):
                # The aggregate over every group, kept only when nothing reports it ungrouped.
                metrics.setdefault(grouped_name, value)
            elif name.startswith(prefix) and name.endswith(suffix):
                group = name[len(prefix) : len(name) - len(suffix)]
                subset_metrics.setdefault(group, {})[grouped_name] = value

    # A row names one group, so it can only be named when every grouped metric groups
    # by the same key. Grouped by several, the subset table spans dimensions a single
    # name cannot place the row in, and naming it by one of them would be arbitrary.
    group_key = next(iter(group_keys)) if len(group_keys) == 1 else None

    resolved_topline = "accuracy" if "accuracy" in metrics else next(iter(metrics), "accuracy")

    traces: list[records.Trace] = []
    for index, sample in enumerate(getattr(log, "samples", None) or []):
        messages = messages_to_dicts(getattr(sample, "messages", None))
        prompt_tokens, completion_tokens = _usage_for_sample(sample)
        score_value: float | None = None
        parsed_pred: Any = None
        for score in (getattr(sample, "scores", None) or {}).values():
            score_value = score_to_float(getattr(score, "value", None))
            parsed_pred = getattr(score, "answer", None)
            break
        error = getattr(sample, "error", None)
        error_type = type(error).__name__ if error else None
        if score_value is None and error_type is None:
            # Without a readable score or an error the row would render as a pass.
            error_type = "ungraded"
        sample_group = (getattr(sample, "metadata", None) or {}).get(group_key) if group_key else None
        total_time = getattr(sample, "total_time", None)
        output = getattr(sample, "output", None)
        fallback = getattr(output, "completion", None) if output is not None else None
        answer_text, reasoning_text, tool_calls = assistant_output(getattr(sample, "messages", None))
        completion = answer_text or fallback
        traces.append(
            records.Trace(
                index=index,
                sample_id=str(getattr(sample, "id", index)),
                epoch=getattr(sample, "epoch", None),
                request_hash=run_store.request_hash(messages),
                messages=messages,
                tools=request_tools(sample),
                expected_answer=getattr(sample, "target", None),
                content=completion or "",
                reasoning_content=reasoning_text,
                tool_calls=tool_calls,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=int(total_time * 1000) if isinstance(total_time, (int, float)) else None,
                error_type=error_type,
                error_message=str(error)[:500] if error else None,
                score=score_value,
                parsed_pred=parsed_pred,
                subset=None if sample_group is None else str(sample_group),
            )
        )

    if not metrics:
        # A run cut short never reaches InspectAI's scoring step, so `results` is empty;
        # recompute a topline over every attempted sample: one without a usable score
        # earns zero credit, and a run with no scored sample keeps empty metrics.
        if any(t.score is not None for t in traces):
            credited = [t.score if t.score is not None else 0.0 for t in traces]
            metrics = {"accuracy": statistics.fmean(credited)}
            if len(credited) > 1:
                metrics["stderr"] = statistics.stdev(credited) / math.sqrt(len(credited))

    # A log that did not finish holds an incomplete score even when the scorer ran, so
    # the result stays `partial` whatever metrics it carries.
    status = "ok" if metrics and getattr(log, "status", None) == "success" else "partial"

    result = records.DatasetResult(
        dataset=dataset_id,
        topline_metric_name=resolved_topline,
        status=status,
        metrics=metrics,
        subset_metrics=subset_metrics or None,
        n=len(traces),
        num_ok=sum(1 for t in traces if t.error_type is None),
        num_error=sum(1 for t in traces if t.error_type is not None),
        error_types=dict(collections.Counter(t.error_type for t in traces if t.error_type)),
    )
    return result, traces
