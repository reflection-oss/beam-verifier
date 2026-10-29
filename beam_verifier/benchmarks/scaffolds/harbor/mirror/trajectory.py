"""Read Mirror's JSONL journal into messages, tool results, and token counts."""

import dataclasses
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, cast

type Source = Literal["system", "user", "agent"]


@dataclasses.dataclass
class ToolCall:
    """A tool call requested by an assistant message."""

    call_id: str
    name: str
    arguments: Mapping[str, object]


@dataclasses.dataclass
class ToolResult:
    """A tool call's output and ID as recorded in the journal."""

    call_id: str
    content: str


@dataclasses.dataclass
class Usage:
    """Input, output, and cached token counts for a call or a whole run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0


@dataclasses.dataclass
class Turn:
    """A conversation message with its tool calls, attached results, and usage."""

    source: Source
    message: str
    reasoning_content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    usage: Usage | None = None


@dataclasses.dataclass
class Compaction:
    """A recorded context compaction: a summary and the messages Mirror kept.

    The rebuilt prompt becomes the context for later model calls. Its copied
    messages do not count as new calls.
    """

    turn_index: int
    """Number of conversation messages before this context reset."""
    prompt: tuple[Turn, ...]
    """The rebuilt prompt: system message, summary, and retained messages."""
    summary: str
    """The recorded summary, or an empty string if none was recorded."""
    trigger: str
    """Warning before compaction, or empty if no warning was recorded."""
    summary_usage: Usage | None = None
    """Token counts for the model call that produced the summary."""


@dataclasses.dataclass
class Journal:
    """Parsed messages, context compactions, and read errors from a Mirror run."""

    turns: tuple[Turn, ...]
    compactions: tuple[Compaction, ...]
    unreadable_lines: int
    """Count of nonblank lines that could not be read as JSON objects."""
    orphaned_tool_results: int
    """Count of dropped tool results with no preceding assistant message."""

    @property
    def usage(self) -> Usage:
        """Token totals for recorded model calls, including context summaries."""
        usages = [turn.usage for turn in self.turns] + [compaction.summary_usage for compaction in self.compactions]
        return Usage(
            prompt_tokens=sum(usage.prompt_tokens for usage in usages if usage),
            completion_tokens=sum(usage.completion_tokens for usage in usages if usage),
            cached_tokens=sum(usage.cached_tokens for usage in usages if usage),
        )


def _mapping(value: object) -> Mapping[str, object] | None:
    # Decoded JSON objects always key by `str`; narrow once here so the readers
    # above work in known types.
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return None


def _field(record: Mapping[str, object], camel: str, snake: str) -> object:
    """Read a JSON field using either its camelCase or snake_case name."""
    if camel in record:
        return record[camel]
    return record.get(snake)


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _count(value: object) -> int:
    # Canonical protobuf JSON renders 64-bit integers as strings.
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, str):
        try:
            return max(int(value), 0)
        except ValueError:
            return 0
    return 0


def _arguments(raw: object) -> Mapping[str, object]:
    # ATIF requires an object; keep malformed arguments as text rather than lose them.
    text = _text(raw)
    if not text:
        return {}
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return {"raw_arguments": text}
    parsed = _mapping(decoded)
    return parsed if parsed is not None else {"raw_arguments": text}


def _tool_calls(raw: object) -> tuple[ToolCall, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    calls: list[ToolCall] = []
    for entry in cast(Sequence[object], raw):
        call = _mapping(entry)
        if call is None:
            continue
        function = _mapping(call.get("function"))
        if function is None:
            continue
        calls.append(
            ToolCall(
                call_id=_text(call.get("id")),
                name=_text(function.get("name")),
                arguments=_arguments(function.get("arguments")),
            )
        )
    return tuple(calls)


def _usage(record: Mapping[str, object]) -> Usage | None:
    usage = _mapping(record.get("usage"))
    if usage is None:
        return None
    return Usage(
        prompt_tokens=_count(_field(usage, "promptTokens", "prompt_tokens")),
        completion_tokens=_count(_field(usage, "completionTokens", "completion_tokens")),
        cached_tokens=_count(_field(usage, "cachedTokens", "cached_tokens")),
    )


def _turn(chat: Mapping[str, object]) -> Turn | None:
    """Read a system, user, or assistant message; return `None` for other records."""
    if (system := _mapping(chat.get("system"))) is not None:
        return Turn(source="system", message=_text(system.get("content")))
    if (user := _mapping(chat.get("user"))) is not None:
        return Turn(source="user", message=_text(user.get("content")))
    if (assistant := _mapping(chat.get("assistant"))) is not None:
        reasoning = _text(_field(assistant, "reasoningContent", "reasoning_content"))
        return Turn(
            source="agent",
            message=_text(assistant.get("content")),
            reasoning_content=reasoning or None,
            tool_calls=_tool_calls(_field(assistant, "toolCalls", "tool_calls")),
            usage=_usage(chat),
        )
    return None


def _tool_result(chat: Mapping[str, object]) -> ToolResult | None:
    tool = _mapping(chat.get("tool"))
    if tool is None:
        return None
    return ToolResult(
        call_id=_text(_field(tool, "toolCallId", "tool_call_id")),
        content=_text(tool.get("content")),
    )


def _attach_result(turns: list[Turn], result: ToolResult) -> bool:
    """Attach a tool result to the latest assistant message, if one exists."""
    for index in range(len(turns) - 1, -1, -1):
        caller = turns[index]
        if caller.source == "agent":
            turns[index] = dataclasses.replace(caller, tool_results=(*caller.tool_results, result))
            return True
    return False


def _user_turn(record: Mapping[str, object]) -> Turn | None:
    """Read Mirror background-task results or extension messages as user messages."""
    task = _mapping(_field(record, "taskResult", "task_result"))
    if task is not None:
        status = "succeeded" if task.get("success") is True else "failed"
        task_id = _text(_field(task, "taskId", "task_id"))
        tool = _text(_field(task, "toolName", "tool_name"))
        content = _text(task.get("content"))
        return Turn(
            source="user",
            message=(f'<task_result id="{task_id}" tool="{tool}" status="{status}">\n{content}\n</task_result>'),
        )
    extension = _mapping(record.get("extension"))
    if extension is None:
        return None
    model_message = _mapping(_field(extension, "modelMessage", "model_message"))
    if model_message is None:
        return None
    return Turn(source="user", message=_text(model_message.get("content")))


def _compaction(snapshot: Mapping[str, object], turn_index: int, trigger: str) -> Compaction | None:
    """Read a recorded context compaction, or return `None` if absent."""
    compaction = _mapping(snapshot.get("compaction"))
    if compaction is None:
        return None
    summary = _text(compaction.get("summary"))
    prompt: list[Turn] = []
    if system_prompt := _text(_field(snapshot, "systemPrompt", "system_prompt")):
        prompt.append(Turn(source="system", message=system_prompt))
    if summary:
        # Match the user turn Mirror inserts into the rebuilt prompt.
        message = (
            "The conversation so far was compacted into this summary. Continue from "
            f"where it leaves off instead of starting over:\n<summary>\n{summary}\n"
            "</summary>"
        )
        prompt.append(Turn(source="user", message=message))
    recent = compaction.get("recent")
    if isinstance(recent, Sequence) and not isinstance(recent, (str, bytes)):
        for kept in cast(Sequence[object], recent):
            record = _mapping(kept)
            if record is None:
                continue
            chat = _mapping(record.get("chat"))
            if chat is None:
                if (turn := _user_turn(record)) is not None:
                    prompt.append(turn)
            elif (result := _tool_result(chat)) is not None:
                _attach_result(prompt, result)
            elif (turn := _turn(chat)) is not None:
                prompt.append(turn)
    return Compaction(
        turn_index=turn_index,
        prompt=tuple(prompt),
        summary=summary,
        trigger=trigger,
        summary_usage=_usage(compaction),
    )


def parse(lines: Iterable[str]) -> Journal:
    """Read Mirror's journal into conversation messages and context compactions.

    Attach tool results to the latest assistant message. Skip records with no
    conversation content. If the system prompt appears only in saved context,
    add it once at the beginning.

    Args:
        lines: Journal lines in the order Mirror wrote them.

    Returns:
        Parsed messages and compactions, plus counts of unreadable lines and
        tool results with no preceding assistant message.
    """
    turns: list[Turn] = []
    compactions: list[Compaction] = []
    # The latest warning since the last conversation record, which names what
    # forced a compaction that follows it.
    trigger = ""
    unreadable = 0
    orphaned = 0
    snapshot_prompt: str | None = None
    for line in lines:
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError:
            unreadable += 1
            continue
        record = _mapping(decoded)
        if record is None:
            unreadable += 1
            continue
        log = _mapping(record.get("log"))
        if log is not None:
            text = _text(log.get("text"))
            # Context overflow logs this warning; proactive compaction logs nothing.
            if text.endswith("; compacting"):
                trigger = text
            continue
        snapshot = _mapping(_field(record, "contextWindow", "context_window"))
        if snapshot is not None:
            if snapshot_prompt is None:
                snapshot_prompt = _text(_field(snapshot, "systemPrompt", "system_prompt"))
            if (compaction := _compaction(snapshot, len(turns), trigger)) is not None:
                compactions.append(compaction)
                trigger = ""
        chat = _mapping(record.get("chat"))
        if chat is None:
            if (turn := _user_turn(record)) is not None:
                turns.append(turn)
            continue
        trigger = ""
        if (result := _tool_result(chat)) is not None:
            if not _attach_result(turns, result):
                orphaned += 1
            continue
        if (turn := _turn(chat)) is not None:
            turns.append(turn)
    if snapshot_prompt and not any(turn.source == "system" for turn in turns):
        turns.insert(0, Turn(source="system", message=snapshot_prompt))
        compactions = [
            dataclasses.replace(compaction, turn_index=compaction.turn_index + 1) for compaction in compactions
        ]
    return Journal(
        turns=tuple(turns),
        compactions=tuple(compactions),
        unreadable_lines=unreadable,
        orphaned_tool_results=orphaned,
    )
