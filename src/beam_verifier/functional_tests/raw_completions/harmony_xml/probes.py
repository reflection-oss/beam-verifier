"""Structural and semantic probe ids for Harmony XML conformance."""

from __future__ import annotations

import dataclasses
import enum


class ProbeId(enum.StrEnum):
    """Structural and semantic conformance categories."""

    CHANNEL_PRESENCE_VALIDITY = "channel_presence_validity"
    MANDATORY_FIRST_ANALYSIS = "mandatory_first_analysis"
    NO_TRAILING_REASONING = "no_trailing_reasoning"
    REASONING_MESSAGE_FORM = "reasoning_message_form"
    MESSAGE_START_AFTER_END = "message_start_after_end"
    FINAL_TERMINATOR = "final_terminator"
    TOOL_CALL_HEADER_FORM = "tool_call_header_form"
    TOOL_CALL_NAME_EXISTS = "tool_call_name_exists"
    CODEC_CORRECTNESS = "codec_correctness"
    GROUPING_PLACEMENT = "grouping_placement"
    PARALLEL_TERMINATION = "parallel_termination"
    TOOL_CALL_XML_WELL_FORMED = "tool_call_xml_well_formed"
    TOOL_CALL_PARAMETER_EXISTS = "tool_call_parameter_exists"
    TOOL_CALL_PARAMETER_REQUIRED = "tool_call_parameter_required"
    TOOL_CALL_PARAMETER_TYPES = "tool_call_parameter_types"
    UNCLASSIFIED = "unclassified"


PROBE_MESSAGES: dict[ProbeId, str] = {
    ProbeId.CHANNEL_PRESENCE_VALIDITY: ("Assistant messages must use the analysis or final channel."),
    ProbeId.MANDATORY_FIRST_ANALYSIS: ("The first assistant message must be analysis."),
    ProbeId.NO_TRAILING_REASONING: ("A completion must end on final or a tool call, not analysis."),
    ProbeId.REASONING_MESSAGE_FORM: ("Analysis must be non-empty, have no recipient, and terminate with <|end|>."),
    ProbeId.MESSAGE_START_AFTER_END: ("After <|end|>, the next assistant message must start with <|start|>assistant."),
    ProbeId.FINAL_TERMINATOR: ("Final must terminate with <|return|>, never <|end|>."),
    ProbeId.TOOL_CALL_HEADER_FORM: ("A tool call must put to=NAME before <|channel|>analysis."),
    ProbeId.TOOL_CALL_NAME_EXISTS: ("to=NAME must name a declared tool."),
    ProbeId.CODEC_CORRECTNESS: ("Tool calls must use analysis <|constrain|>xml before <|message|>."),
    ProbeId.GROUPING_PLACEMENT: ("Parallel tool calls must be grouped without analysis or final between them."),
    ProbeId.PARALLEL_TERMINATION: ("Non-last tool calls end with <|end|>; the last ends with <|call|>."),
    ProbeId.TOOL_CALL_XML_WELL_FORMED: ("Tool-call XML must parse as rfl:parameter elements."),
    ProbeId.TOOL_CALL_PARAMETER_EXISTS: ("Each tool parameter must exist in the tool schema."),
    ProbeId.TOOL_CALL_PARAMETER_REQUIRED: ("Each required tool parameter must be present."),
    ProbeId.TOOL_CALL_PARAMETER_TYPES: ("Tool parameter values must coerce to their declared JSON Schema types."),
    ProbeId.UNCLASSIFIED: ("The completion contains content after a structurally complete generation."),
}


class CompletionShape(enum.StrEnum):
    """Structural form of a generation, independent of pass/fail."""

    UNKNOWN = "unknown"
    FINAL_ONLY = "final_only"
    NO_CALL = "no_call"
    SINGLE_TOOL = "single_tool"
    PARALLEL_TOOL = "parallel_tool"


@dataclasses.dataclass(frozen=True, slots=True)
class CompletionFacets:
    """Orthogonal attributes of a generation, independent of `CompletionShape`."""

    builtin_tool: bool = False
    schema_tool: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class ProbeFailure:
    """A conformance failure attributed to one probe."""

    probe_id: ProbeId
    message: str


@dataclasses.dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of validating one model generation."""

    ok: bool
    failure: ProbeFailure | None = None
    truncated: bool = False
    shape: CompletionShape = CompletionShape.UNKNOWN
    facets: CompletionFacets = dataclasses.field(default_factory=CompletionFacets)

    @classmethod
    def success(cls) -> ValidationResult:
        """Return a successful result."""
        return cls(ok=True)

    @classmethod
    def incomplete(cls) -> ValidationResult:
        """Return a truncation result without attributing a probe."""
        return cls(ok=False, truncated=True)

    @classmethod
    def fail(cls, probe_id: ProbeId) -> ValidationResult:
        """Return a failure for `probe_id`."""
        return cls(
            ok=False,
            failure=ProbeFailure(probe_id, PROBE_MESSAGES[probe_id]),
        )

    def classified(self, shape: CompletionShape, facets: CompletionFacets) -> ValidationResult:
        """Return a copy carrying the structural classification of the walk."""
        return dataclasses.replace(self, shape=shape, facets=facets)
