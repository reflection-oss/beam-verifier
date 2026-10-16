"""Offline tests for records + run store (no network)."""

import json
import pathlib
import re

import pytest

from beam_verifier import records
from beam_verifier.reporting import html, run_store, view


def test_request_hash_deterministic():
    m = [{"role": "user", "content": "hi"}]
    assert run_store.request_hash(m) == run_store.request_hash([{"role": "user", "content": "hi"}])
    assert run_store.request_hash(m) != run_store.request_hash([{"role": "user", "content": "bye"}])


def test_run_store_roundtrip(tmp_path):
    store = run_store.RunStore(tmp_path / "run")
    traces = [
        records.Trace(
            index=0,
            request_hash="h",
            messages=[{"role": "user", "content": "q"}],
            expected_answer="B",
            content="The answer is B",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            score=1.0,
            parsed_pred="B",
        )
    ]
    store.write_traces("gpqa", traces)
    man = records.RunManifest(
        run_id="r",
        kind="eval",
        created_at="t",
        version="0.1.0",
        model="m",
        base_url="e",
        dataset_results=[
            records.DatasetResult(dataset="gpqa", topline_metric_name="grade", metrics={"grade": 100.0}, n=1, num_ok=1)
        ],
    )
    store.write_manifest(man)

    m2, t2 = run_store.load_run(tmp_path / "run")
    assert m2.model == "m" and m2.kind == "eval"
    assert t2["gpqa"][0].score == 1.0 and t2["gpqa"][0].content == "The answer is B"
    assert t2["gpqa"][0].finish_reason == "stop"


def test_verify_manifest_preserves_endpoint_urls(tmp_path):
    store = run_store.RunStore(tmp_path / "run")
    store.write_manifest(
        records.RunManifest(
            run_id="r",
            kind="verify",
            created_at="t",
            version="0.1.0",
            model="m",
            base_url="https://example.test/parse",
            config=records.EvalSettings(
                parse_url="https://example.test/parse",
                format_url="https://example.test/format",
            ),
        )
    )

    manifest, _ = run_store.load_run(tmp_path / "run")
    assert manifest.config.parse_url == "https://example.test/parse"
    assert manifest.config.format_url == "https://example.test/format"


def test_jsonl_unicode_line_separators_roundtrip(tmp_path):
    # Content with Unicode line separators that JSON leaves unescaped but str.splitlines() breaks on.
    store = run_store.RunStore(tmp_path / "run")
    nasty = "answer with\x85weird separators \\sin(\\theta)"
    store.write_traces("x", [records.Trace(index=0, request_hash="h", messages=[], content=nasty, score=1.0)])
    store.write_manifest(
        records.RunManifest(run_id="r", kind="eval", created_at="t", version="0.1.0", model="m", base_url="e")
    )
    _, t2 = run_store.load_run(tmp_path / "run")
    assert len(t2["x"]) == 1
    assert t2["x"][0].content == nasty


def test_load_manifest_renames_keys_written_under_an_earlier_schema(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "r",
                "kind": "eval",
                "created_at": "t",
                "reflection_eval_version": "0.1.0",
                "model": "m",
                "endpoint": "http://endpoint/v1",
                "evalsets": [{"evalset": "gpqa-diamond", "topline_metric_name": "accuracy"}],
            }
        )
    )
    manifest = run_store.load_manifest(run_dir)
    assert manifest.version == "0.1.0"
    assert manifest.base_url == "http://endpoint/v1"
    assert [result.dataset for result in manifest.dataset_results] == ["gpqa-diamond"]


@pytest.mark.source_checkout
def test_package_root_is_the_directory_holding_pyproject():
    root = run_store.package_root()
    assert root is not None, "Expected to locate the package root from a source checkout"
    assert (root / "pyproject.toml").is_file(), f"No pyproject.toml at resolved package root {root}"


@pytest.mark.source_checkout
def test_default_runs_dir_is_dot_runs_at_the_package_root():
    root = run_store.package_root()
    assert root is not None
    assert run_store.default_runs_dir() == root / run_store.RUNS_DIR_NAME


def test_new_run_dir_honors_an_explicit_base(tmp_path):
    run_dir = run_store.new_run_dir("zai-org/GLM-5.2-FP8", "eval", base=tmp_path)
    assert run_dir.parent == tmp_path
    assert run_dir.name.endswith("-zai-org-GLM-5.2-FP8-eval")
    assert (run_dir / "traces").is_dir()


def test_failed_dataset_is_saved_with_its_error(tmp_path):
    """A dataset whose runner raised stays in the saved run, with the reason why."""
    store = run_store.RunStore(tmp_path / "run")
    store.write_manifest(
        records.RunManifest(
            run_id="r",
            kind="eval",
            created_at="t",
            version="0.1.0",
            model="m",
            base_url="e",
            dataset_results=[
                records.DatasetResult(
                    dataset="aime2026", topline_metric_name="", status="failed", error="task loading failed"
                ),
                records.DatasetResult(
                    dataset="math500", topline_metric_name="grade", metrics={"grade": 42.0}, n=1, num_ok=1
                ),
            ],
        )
    )
    manifest, _ = run_store.load_run(tmp_path / "run")
    failed = manifest.dataset_results[0]
    assert failed.dataset == "aime2026" and failed.status == "failed"
    assert failed.error == "task loading failed"
    assert not failed.metrics


def test_html_report_shows_a_tool_call_answer(tmp_path):
    """A function-calling row answers with its tool calls and no text at all, so a
    report showing `content` alone would render the graded answer as empty."""
    run_dir = tmp_path / "run"
    store = run_store.RunStore(run_dir)
    store.write_traces(
        "tool-eval",
        [
            records.Trace(
                index=0,
                request_hash="h",
                messages=[{"role": "user", "content": "q"}],
                content="",
                tool_calls=[{"id": "c1", "function": "calc_binomial_probability", "arguments": {"n": 20, "k": 5}}],
                score=0.0,
            )
        ],
    )
    store.write_manifest(
        records.RunManifest(
            run_id="r",
            kind="eval",
            created_at="2026-09-10T00:00:00",
            version="0.1.0",
            model="m",
            base_url="e",
            dataset_results=[records.DatasetResult(dataset="tool-eval", topline_metric_name="accuracy", n=1)],
        )
    )
    out_path = tmp_path / "report.html"
    html.write_report(run_dir, out_path)
    report = out_path.read_text()
    assert "calc_binomial_probability({&quot;n&quot;: 20, &quot;k&quot;: 5})" in report
    assert "(empty)" not in report


def test_html_report_renders_a_conversation_turn_by_turn(tmp_path):
    """A transcript trace shows every turn — text, tool calls, tool results —
    while a plain request keeps its single prompt block."""
    run_dir = tmp_path / "run"
    store = run_store.RunStore(run_dir)
    store.write_traces(
        "conversation",
        [
            records.Trace(
                index=0,
                request_hash="h",
                messages=[
                    {"role": "user", "content": "Please calculate six times seven."},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"function": "multiply", "arguments": {"a": 6, "b": 7}}],
                        "reasoning_content": "thinking",
                    },
                    {"role": "tool", "content": "42"},
                ],
                content="",
                score=1.0,
            ),
            records.Trace(
                index=1,
                request_hash="h2",
                messages=[{"role": "user", "content": "plain question"}],
                content="plain answer",
                score=1.0,
            ),
        ],
    )
    store.write_manifest(
        records.RunManifest(
            run_id="r",
            kind="eval",
            created_at="2026-09-10T00:00:00",
            version="0.1.0",
            model="m",
            base_url="e",
            dataset_results=[records.DatasetResult(dataset="conversation", topline_metric_name="accuracy", n=2)],
        )
    )
    out_path = tmp_path / "report.html"
    html.write_report(run_dir, out_path)
    report = out_path.read_text()
    assert "multiply({&quot;a&quot;: 6, &quot;b&quot;: 7})" in report
    assert "42" in report
    assert "reasoning_content (8 chars)" in report
    assert report.count("<div class='lbl'>tool</div>") == 1
    assert report.count("<div class='lbl'>prompt</div>") == 1


def test_html_report_shows_the_reasoning_of_a_text_only_transcript(tmp_path):
    """A transcript whose only structure is per-turn reasoning must render turn by
    turn: the single prompt block shows role and content alone."""
    run_dir = tmp_path / "run"
    store = run_store.RunStore(run_dir)
    store.write_traces(
        "conversation",
        [
            records.Trace(
                index=0,
                request_hash="h",
                messages=[
                    {"role": "user", "content": "What is 6 times 7?"},
                    {"role": "assistant", "content": "42", "reasoning_content": "Compute 6 * 7."},
                ],
                content="42",
                score=1.0,
            )
        ],
    )
    store.write_manifest(
        records.RunManifest(
            run_id="r",
            kind="eval",
            created_at="2026-09-10T00:00:00",
            version="0.1.0",
            model="m",
            base_url="e",
            dataset_results=[records.DatasetResult(dataset="conversation", topline_metric_name="accuracy", n=1)],
        )
    )
    out_path = tmp_path / "report.html"
    html.write_report(run_dir, out_path)
    report = out_path.read_text()
    assert "Compute 6 * 7." in report
    assert "<div class='lbl'>prompt</div>" not in report


def test_terminal_row_shows_a_tool_call_answer_literally(tmp_path):
    """Tool-call arguments carry the brackets rich reads as console markup, so the
    terminal row has to print the call as written rather than consuming them."""
    run_dir = tmp_path / "run"
    store = run_store.RunStore(run_dir)
    store.write_traces(
        "tool-eval",
        [
            records.Trace(
                index=0,
                request_hash="h",
                messages=[{"role": "user", "content": "q"}],
                content="",
                tool_calls=[{"id": "c1", "function": "cosine", "arguments": {"flags": [True, False]}}],
                score=1.0,
            )
        ],
    )
    store.write_manifest(
        records.RunManifest(
            run_id="r",
            kind="eval",
            created_at="2026-09-10T00:00:00",
            version="0.1.0",
            model="m",
            base_url="e",
            dataset_results=[records.DatasetResult(dataset="tool-eval", topline_metric_name="accuracy", n=1)],
        )
    )
    with view.console.capture() as captured:
        view.render(run_dir, row=0)
    assert 'cosine({"flags": [true, false]})' in captured.get()


def test_terminal_row_shows_markup_shaped_values_literally(tmp_path):
    """A dataset's and a model's own text routinely holds `[...]`, which rich reads as
    console markup: it drops what looks like a tag and raises on a stray closing one.
    Every such value in the row has to survive to the screen as written."""
    run_dir = tmp_path / "run"
    store = run_store.RunStore(run_dir)
    store.write_traces(
        "tool-eval",
        [
            records.Trace(
                index=0,
                request_hash="h",
                messages=[{"role": "user", "content": "compare [a, b]"}],
                content="",
                tool_calls=[{"id": "c1", "function": "cosine", "arguments": {"flags": [True, False]}}],
                expected_answer="cosine(v=[1, 2])",
                parsed_pred="cosine(v=[/])",
                error_type="EvalError",
                error_message="failed on [bold] input",
                score=0.0,
            )
        ],
    )
    store.write_manifest(
        records.RunManifest(
            run_id="r",
            kind="eval",
            created_at="2026-09-10T00:00:00",
            version="0.1.0",
            model="m",
            base_url="e",
            dataset_results=[records.DatasetResult(dataset="tool-eval", topline_metric_name="accuracy", n=1)],
        )
    )
    with view.console.capture() as captured:
        view.render(run_dir, row=0)
    # rich styles this row, so the colour codes come out interleaved with the text.
    shown = re.sub(r"\x1b\[[0-9;]*m", "", captured.get())
    assert "compare [a, b]" in shown
    assert 'cosine({"flags": [true, false]})' in shown
    assert "cosine(v=[1, 2])" in shown
    assert "cosine(v=[/])" in shown
    assert "failed on [bold] input" in shown


def test_terminal_row_shows_each_turns_reasoning_and_role_literally(tmp_path):
    """A transcript stores reasoning per message. The row must show each turn's
    reasoning and print a markup-shaped role literally."""
    run_dir = tmp_path / "run"
    store = run_store.RunStore(run_dir)
    store.write_traces(
        "conversation",
        [
            records.Trace(
                index=0,
                request_hash="h",
                messages=[
                    {"role": "[/]", "content": "Please calculate six times seven."},
                    {"role": "assistant", "content": "", "reasoning_content": "Multiply the numbers"},
                ],
                content="",
                score=1.0,
            )
        ],
    )
    store.write_manifest(
        records.RunManifest(
            run_id="r",
            kind="eval",
            created_at="2026-09-10T00:00:00",
            version="0.1.0",
            model="m",
            base_url="e",
            dataset_results=[records.DatasetResult(dataset="conversation", topline_metric_name="accuracy", n=1)],
        )
    )
    with view.console.capture() as captured:
        view.render(run_dir, row=0)
    shown = re.sub(r"\x1b\[[0-9;]*m", "", captured.get())
    assert "[/]: Please calculate six times seven." in shown
    assert "assistant reasoning: Multiply the numbers" in shown


def test_answer_text_keeps_the_text_of_a_turn_that_also_called_a_tool():
    """A turn can answer in text and call a tool, and the report has to show both."""
    trace = records.Trace(
        index=0,
        request_hash="h",
        messages=[],
        content="Let me look that up.",
        tool_calls=[{"id": "c1", "function": "search", "arguments": {"q": "x"}}],
    )
    assert records.answer_text(trace) == 'Let me look that up.\nsearch({"q": "x"})'


def test_answer_text_is_empty_when_a_turn_produced_nothing():
    """An empty answer still has to read as empty, not as a call with no name."""
    assert records.answer_text(records.Trace(index=0, request_hash="h", messages=[])) == ""


def test_html_report_shows_a_failed_dataset_without_a_score(tmp_path):
    run_dir = tmp_path / "run"
    store = run_store.RunStore(run_dir)
    store.write_manifest(
        records.RunManifest(
            run_id="r",
            kind="eval",
            created_at="2026-09-10T00:00:00",
            version="0.1.0",
            model="m",
            base_url="e",
            dataset_results=[
                records.DatasetResult(
                    dataset="aime2026", topline_metric_name="", status="failed", error="task loading failed"
                )
            ],
        )
    )
    out_path = tmp_path / "report.html"
    html.write_report(run_dir, out_path)
    report = out_path.read_text()
    assert "failed — task loading failed" in report
    assert "aime2026" in report


def _run_with_trace(tmp_path, trace: records.Trace) -> pathlib.Path:
    """Saves a one-row eval run and returns its directory."""
    run_dir = tmp_path / "run"
    store = run_store.RunStore(run_dir)
    store.write_traces("tool-eval", [trace])
    store.write_manifest(
        records.RunManifest(
            run_id="r",
            kind="eval",
            created_at="2026-09-10T00:00:00",
            version="0.1.0",
            model="m",
            base_url="e",
            dataset_results=[records.DatasetResult(dataset="tool-eval", topline_metric_name="accuracy", n=1)],
        )
    )
    return run_dir


def test_report_rows_name_the_subset_that_scored_them(tmp_path):
    """A subset table the rows cannot be tied back to is half the breakdown, so each
    row has to carry the group its per-subset metric covers."""
    run_dir = _run_with_trace(
        tmp_path, records.Trace(index=0, request_hash="h", messages=[], content="a", score=1.0, subset="sql")
    )
    out_path = tmp_path / "report.html"
    html.write_report(run_dir, out_path)
    assert "tool-eval · sql" in out_path.read_text()
    with view.console.capture() as captured:
        view.render(run_dir)
    assert "subset" in captured.get()
    assert "sql" in captured.get()


def test_report_rows_of_an_ungrouped_run_claim_no_subset(tmp_path):
    """A run whose scorer grouped nothing has no subsets, and the rows must not
    invent one."""
    run_dir = _run_with_trace(tmp_path, records.Trace(index=0, request_hash="h", messages=[], content="a", score=1.0))
    out_path = tmp_path / "report.html"
    html.write_report(run_dir, out_path)
    assert "tool-eval ·" not in out_path.read_text()
