"""Functional tests for OpenAI-compatible endpoints."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class SuiteSpec:
    """One functional-test suite: a top-level directory under functional_tests/.

    Attributes:
        directory: The suite's directory name, as it appears in pytest node ids.
        display_name: Human-readable name used in reports.
        required: Whether verify fails when the suite does not run.
        gate: The endpoint option(s) whose absence explains a skipped optional
            suite in reports, e.g. "--parse-url".
    """

    directory: str
    display_name: str
    required: bool
    gate: str | None = None


# Human-readable category per e2e test module, keyed by module stem; report
# rendering shows it next to each failing test.
TEST_MODULE_CATEGORIES = {
    "test_request_contract": "Request contract conformance",
    "test_response_contract": "Response contract conformance",
    "test_tool_protocol": "Tool-call protocol conformance",
    "test_parser": "Deterministic parser extraction",
    "test_formatter": "Deterministic prompt formatting",
    "test_raw_completions": "Raw Harmony completion conformance",
}

SUITES = (
    SuiteSpec(directory="e2e", display_name="API", required=True),
    SuiteSpec(
        directory="parser",
        display_name="Parser (non-streaming)",
        required=False,
        gate="--parse-url",
    ),
    SuiteSpec(directory="formatter", display_name="Formatter", required=False, gate="--format-url"),
    SuiteSpec(
        directory="raw_completions",
        display_name="Raw completions",
        required=False,
        gate="--completions-url",
    ),
)
