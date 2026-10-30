"""Terminal rendering of a saved run (eval or verify)."""

import pathlib

import rich.console
import rich.markup
import rich.panel
import rich.table
import rich.text

from beam_verifier import functional_tests, records
from beam_verifier.reporting import run_store

console = rich.console.Console()


def _category(failing_id: str) -> str:
    """The failing test module's human-readable category ("" when unknown)."""
    module_stem = pathlib.PurePosixPath(failing_id.partition("::")[0]).stem
    return functional_tests.TEST_MODULE_CATEGORIES.get(module_stem, "")


def _resolved_test_id(failing_id: str) -> rich.text.Text:
    """The failing test id with its file part resolved to a real path.

    Ids are stored relative to the functional_tests package; resolving the
    path makes the id's file prefix a plain path that editor terminals
    (VS Code, JetBrains, kitty) linkify themselves — including over remote
    sessions, unlike URL-scheme hyperlinks that hand off to an OS opener —
    and makes the whole string a directly rerunnable pytest node id. Falls
    back to the stored id when the packaged test file cannot be found (e.g.
    viewing a foreign run). Literal text: parametrized ids carry [brackets]
    that rich markup would otherwise swallow.
    """
    file_part, _, test_part = failing_id.partition("::")
    test_file = pathlib.Path(functional_tests.__file__).parent / file_part
    if not test_file.is_file():
        return rich.text.Text(failing_id)
    path = test_file.resolve()
    try:
        path = path.relative_to(pathlib.Path.cwd())
    except ValueError:
        pass
    return rich.text.Text(f"{path}::{test_part}")


def _num(x) -> str:
    return f"{x:.4g}" if isinstance(x, (int, float)) else "—"


def _trunc(x, n=22) -> str:
    s = "" if x is None else str(x)
    return s if len(s) <= n else s[: n - 1] + "…"


def render(run_dir: pathlib.Path, dataset=None, row=None, failed=False, errors=False, full=False):
    manifest, traces = run_store.load_run(run_dir)
    console.print(
        rich.panel.Panel.fit(
            f"[bold]{manifest.model}[/]  ·  {manifest.base_url}\n"
            f"{manifest.kind} · {manifest.created_at[:19]} · {manifest.duration_s}s · beam-verifier {manifest.version}\n"
            f"config: {manifest.config.model_dump(exclude_none=True)}"
            + "".join(f"\nresumed with: {applied}" for applied in manifest.resume_overrides),
            title=manifest.run_id,
        )
    )

    if manifest.verify is not None:
        render_verify(manifest.verify)
        return

    if row is not None:
        _render_row(run_dir, traces, dataset, row, full)
        return

    summary = rich.table.Table(title="Summary")
    for c in ("dataset", "grade", "status", "n", "ok/err", "finish"):
        summary.add_column(c, style="cyan" if c == "dataset" else None)
    for dataset_result in manifest.dataset_results:
        # A recomputed score covers only the samples that finished, and a failed dataset
        # has no score at all; say so rather than letting either read as a complete run.
        if dataset_result.status == "failed":
            status = f"[red]failed[/] ({dataset_result.error})" if dataset_result.error else "[red]failed[/]"
        elif dataset_result.status == "ok":
            status = "complete"
        else:
            status = f"[yellow]partial[/] ({dataset_result.num_ok}/{dataset_result.n} scored)"
        summary.add_row(
            dataset_result.dataset,
            _num(dataset_result.metrics.get(dataset_result.topline_metric_name)),
            status,
            str(dataset_result.n),
            f"{dataset_result.num_ok}/{dataset_result.num_error}",
            " ".join(f"{k}:{v}" for k, v in dataset_result.finish_reasons.items()),
        )
    console.print(summary)
    for dataset_result in manifest.dataset_results:
        if dataset_result.subset_metrics:
            _render_subset_metrics(dataset_result.dataset, dataset_result.subset_metrics)

    title = "Rows" + (" · failed" if failed else " · errors" if errors else "")
    rt = rich.table.Table(title=title)
    for c in ("idx", "dataset", "subset", "score", "expected", "parsed", "finish", "tok p/c", "ms", "error"):
        rt.add_column(c)
    shown = 0
    for name, tlist in traces.items():
        if dataset and name != dataset:
            continue
        for t in tlist:
            is_fail = records.trace_failed(t)
            if failed and not is_fail:
                continue
            if errors and not t.error_type:
                continue
            rt.add_row(
                str(t.index),
                name,
                t.subset or "—",
                _num(t.score),
                _trunc(t.expected_answer),
                _trunc(t.parsed_pred),
                t.finish_reason or "—",
                f"{t.prompt_tokens or '-'}/{t.completion_tokens or '-'}",
                str(t.latency_ms or "-"),
                t.error_type or "",
            )
            shown += 1
    console.print(rt)
    console.print(f"[dim]{shown} rows · full trace: beam-verifier view {run_dir} --row N[/dim]")


def _render_row(run_dir, traces, dataset, idx, full):
    name = dataset or (next(iter(traces)) if len(traces) == 1 else None)
    if name is None:
        console.print("[red]specify --dataset[/] (run has multiple): " + ", ".join(traces))
        return
    t = next((x for x in traces.get(name, []) if x.index == idx), None)
    if t is None:
        console.print(f"[red]no row {idx} in {name}[/]")
        return
    # Every value below is a dataset's or a model's own text, including its role,
    # so it is escaped before
    # reaching rich: unescaped, a `[...]` in it is read as console markup, which drops
    # the text it looks like a tag and raises on a stray closing tag.
    lines: list[str] = []
    for m in t.messages:
        role = rich.markup.escape(str(m.get("role", "")))
        reasoning = m.get("reasoning_content")
        if reasoning:
            shown = reasoning if full else _trunc(reasoning, 1500)
            lines.append(f"[grey50]{role} reasoning: {rich.markup.escape(shown)}[/]")
        text = records.turn_text(m)
        # A turn whose only output is its reasoning needs no empty text line.
        if text or not reasoning:
            lines.append(f"[bold cyan]{role}:[/] {rich.markup.escape(text)}")
    console.print(rich.panel.Panel("\n".join(lines), title=f"{name} · row {idx} · prompt"))
    if t.reasoning_content:
        rc = t.reasoning_content if full else _trunc(t.reasoning_content, 1500)
        console.print(
            rich.panel.Panel(
                rich.text.Text(rc),
                title=f"reasoning_content ({len(t.reasoning_content)} chars{'' if full else ', --full for all'})",
                border_style="grey50",
            )
        )
    answer = records.answer_text(t)
    console.print(rich.panel.Panel(rich.text.Text(answer) if answer else "[dim](empty)[/dim]", title="answer (graded)"))
    verdict = "[green]✓[/]" if (t.score or 0) > 0 else "[red]✗[/]"
    console.print(
        # The rows table truncates the extraction, so this is the only place a long
        # one can be read in full; the HTML report keeps it in each row's summary.
        f"expected=[cyan]{rich.markup.escape(str(t.expected_answer))}[/]  "
        f"parsed=[cyan]{rich.markup.escape(str(t.parsed_pred))}[/]  "
        f"score={t.score} {verdict}  finish={t.finish_reason}  tokens={t.prompt_tokens}/{t.completion_tokens}"
    )
    if t.error_type:
        console.print(f"[red]error[/] {t.error_type}: {rich.markup.escape(str(t.error_message))}")


def render_verify(v: records.VerifyResult) -> None:
    """Render a VerifyResult: per-suite table, exact failing ids, verdict."""
    if v.suites:
        suite_table = rich.table.Table(title="Functional tests")
        for c in ("suite", "status", "cases", "evidence"):
            suite_table.add_column(c, style="cyan" if c == "suite" else None)
        for s in v.suites:
            total = sum(s.counts.values())
            not_failing = total - len(s.failing_ids) - s.counts.get("skipped", 0)
            color = {"PASS": "green", "FAIL": "red"}.get(s.status, "yellow")
            evidence = "required" if s.required else "optional"
            if s.skip_reason:
                evidence += f" — {s.skip_reason}"
            suite_table.add_row(
                s.display_name, f"[{color}]{s.status}[/]", f"{not_failing}/{total}" if total else "—", evidence
            )
        console.print(suite_table)
        rules = {case["name"]: case.get("rule", "") for case in v.functional_cases}
        for s in v.suites:
            if s.failing_ids:
                fail_table = rich.table.Table(title=f"{s.display_name} · failing cases ({len(s.failing_ids)})")
                fail_table.add_column("category", overflow="fold")
                fail_table.add_column("rule", overflow="fold")
                # Fold, never crop: these ids must stay exact to be rerunnable.
                fail_table.add_column("failed test id", style="cyan", overflow="fold")
                previous_category = None
                previous_rule = None
                for index, name in enumerate(s.failing_ids):
                    # Consecutive cases share their test's rule and their
                    # module's category: show each once, and close a rule
                    # group with a section line.
                    category = _category(name)
                    rule = rules.get(name, "")
                    next_name = s.failing_ids[index + 1] if index + 1 < len(s.failing_ids) else None
                    fail_table.add_row(
                        "" if category == previous_category else category,
                        "" if rule == previous_rule else rule,
                        _resolved_test_id(name),
                        end_section=next_name is not None and rules.get(next_name, "") != rule,
                    )
                    previous_category = category
                    previous_rule = rule
                console.print(fail_table)
    else:
        # Pre-suite manifests (older runs) carry only flat outcome counts.
        console.print("Functional: " + (", ".join(f"{k}={val}" for k, val in v.functional.items()) or "n/a"))
        failed = [case for case in v.functional_cases if case.get("outcome") not in records.BENIGN_OUTCOMES]
        if failed:
            fail_table = rich.table.Table(title=f"Failed functional cases ({len(failed)})")
            fail_table.add_column("case", style="cyan")
            fail_table.add_column("outcome")
            for case in failed:
                fail_table.add_row(rich.text.Text(case.get("name", "")), case.get("outcome", ""))
            console.print(fail_table)
    console.print("[green]✓ verify passed[/]" if v.passed else "[red]✗ verify failed[/]")


def _render_subset_metrics(dataset_id: str, subset_metrics: dict[str, dict[str, float]]) -> None:
    """Prints a dataset's per-subset metrics.

    Args:
      dataset_id: The dataset the breakdown belongs to.
      subset_metrics: Subset name to its metrics.
    """
    metric_names = sorted({name for metrics in subset_metrics.values() for name in metrics})
    table = rich.table.Table(title=f"{dataset_id} · subsets")
    table.add_column("subset", style="cyan")
    for name in metric_names:
        table.add_column(name)
    for subset, metrics in subset_metrics.items():
        table.add_row(subset, *(_num(metrics.get(name)) for name in metric_names))
    console.print(table)
