"""User-visible HTML and terminal reports from saved runs."""

import pathlib
import re

from beam_verifier import records
from beam_verifier.reporting import html, run_store, view


def test_html_report_shows_a_tool_call_answer(
    tmp_path: pathlib.Path, run_dir: pathlib.Path, trace_examples: dict[str, records.Trace]
) -> None:
    """A function-calling row answers with its tool calls and no text at all, so a
    report showing `content` alone would render the graded answer as empty."""
    store = run_store.RunStore(run_dir)
    store.write_traces("tool-eval", [trace_examples["tool_answer"]])
    out_path = tmp_path / "report.html"
    html.write_report(run_dir, out_path)
    report = out_path.read_text()
    assert "calc_binomial_probability({&quot;n&quot;: 20, &quot;k&quot;: 5})" in report
    assert "(empty)" not in report


def test_html_report_renders_a_conversation_turn_by_turn(
    tmp_path: pathlib.Path, run_dir: pathlib.Path, trace_examples: dict[str, records.Trace]
) -> None:
    """A transcript trace shows every turn — text, tool calls, tool results —
    while a plain request keeps its single prompt block."""
    store = run_store.RunStore(run_dir)
    manifest = run_store.load_manifest(run_dir)
    manifest.dataset_results[0].n = 2
    store.write_manifest(manifest)
    store.write_traces("tool-eval", [trace_examples["tool_conversation"], trace_examples["plain_answer"]])
    out_path = tmp_path / "report.html"
    html.write_report(run_dir, out_path)
    report = out_path.read_text()
    assert "multiply({&quot;a&quot;: 6, &quot;b&quot;: 7})" in report
    assert "42" in report
    assert "reasoning_content (8 chars)" in report
    assert report.count("<div class='lbl'>tool</div>") == 1
    assert report.count("<div class='lbl'>prompt</div>") == 1


def test_html_report_shows_the_reasoning_of_a_text_only_transcript(
    tmp_path: pathlib.Path, run_dir: pathlib.Path, trace_examples: dict[str, records.Trace]
) -> None:
    """A transcript whose only structure is per-turn reasoning must render turn by
    turn: the single prompt block shows role and content alone."""
    store = run_store.RunStore(run_dir)
    store.write_traces("tool-eval", [trace_examples["reasoning_answer"]])
    out_path = tmp_path / "report.html"
    html.write_report(run_dir, out_path)
    report = out_path.read_text()
    assert "Compute 6 * 7." in report
    assert "<div class='lbl'>prompt</div>" not in report


def test_terminal_row_shows_markup_shaped_values_literally(
    tmp_path: pathlib.Path, run_dir: pathlib.Path, trace_examples: dict[str, records.Trace]
) -> None:
    """A dataset's and a model's own text routinely holds `[...]`, which rich reads as
    console markup: it drops what looks like a tag and raises on a stray closing one.
    Every such value in the row has to survive to the screen as written."""
    store = run_store.RunStore(run_dir)
    store.write_traces("tool-eval", [trace_examples["markup_values"]])
    with view.console.capture() as captured:
        view.render(run_dir, row=0)
    # rich styles this row, so the colour codes come out interleaved with the text.
    shown = re.sub(r"\x1b\[[0-9;]*m", "", captured.get())
    assert "compare [a, b]" in shown
    assert 'cosine({"flags": [true, false]})' in shown
    assert "cosine(v=[1, 2])" in shown
    assert "cosine(v=[/])" in shown
    assert "failed on [bold] input" in shown


def test_terminal_row_shows_each_turns_reasoning_and_role_literally(
    tmp_path: pathlib.Path, run_dir: pathlib.Path, trace_examples: dict[str, records.Trace]
) -> None:
    """A transcript stores reasoning per message. The row must show each turn's
    reasoning and print a markup-shaped role literally."""
    store = run_store.RunStore(run_dir)
    store.write_traces("tool-eval", [trace_examples["markup_role"]])
    with view.console.capture() as captured:
        view.render(run_dir, row=0)
    shown = re.sub(r"\x1b\[[0-9;]*m", "", captured.get())
    assert "[/]: Please calculate six times seven." in shown
    assert "assistant reasoning: Multiply the numbers" in shown


def test_answer_text_keeps_the_text_of_a_turn_that_also_called_a_tool(trace_examples: dict[str, records.Trace]):
    """A turn can answer in text and call a tool, and the report has to show both."""
    trace = trace_examples["text_and_tool"]
    assert records.answer_text(trace) == 'Let me look that up.\nsearch({"q": "x"})'


def test_answer_text_is_empty_when_a_turn_produced_nothing():
    """An empty answer still has to read as empty, not as a call with no name."""
    assert records.answer_text(records.Trace(index=0, request_hash="h", messages=[])) == ""


def test_html_report_shows_a_failed_dataset_without_a_score(
    tmp_path: pathlib.Path, run_manifest: records.RunManifest, failed_dataset: records.DatasetResult
) -> None:
    run_dir = tmp_path / "run"
    store = run_store.RunStore(run_dir)
    run_manifest.dataset_results = [failed_dataset]
    store.write_manifest(run_manifest)
    out_path = tmp_path / "report.html"
    html.write_report(run_dir, out_path)
    report = out_path.read_text()
    assert "failed — task loading failed" in report
    assert "aime2026" in report


def test_report_rows_name_the_subset_that_scored_them(tmp_path: pathlib.Path, run_dir: pathlib.Path) -> None:
    """A subset table the rows cannot be tied back to is half the breakdown, so each
    row has to carry the group its per-subset metric covers."""
    run_store.RunStore(run_dir).write_traces(
        "tool-eval", [records.Trace(index=0, request_hash="h", messages=[], content="a", score=1.0, subset="sql")]
    )
    out_path = tmp_path / "report.html"
    html.write_report(run_dir, out_path)
    assert "tool-eval · sql" in out_path.read_text()
    with view.console.capture() as captured:
        view.render(run_dir)
    assert "subset" in captured.get()
    assert "sql" in captured.get()


def test_report_rows_of_an_ungrouped_run_claim_no_subset(tmp_path: pathlib.Path, run_dir: pathlib.Path) -> None:
    """A run whose scorer grouped nothing has no subsets, and the rows must not
    invent one."""
    run_store.RunStore(run_dir).write_traces(
        "tool-eval", [records.Trace(index=0, request_hash="h", messages=[], content="a", score=1.0)]
    )
    out_path = tmp_path / "report.html"
    html.write_report(run_dir, out_path)
    assert "tool-eval ·" not in out_path.read_text()
