"""Self-contained HTML report for a saved run (no server, no external assets)."""

import html as _html
import pathlib

from beam_verifier import records
from beam_verifier.reporting import run_store

_CSS = """
:root{--bg:#f3f6f3;--panel:#ffffff;--ink:#253b33;--muted:#63746b;--line:#dfe8e2;
--accent:#255e47;--mint:#e8f4e9;--good:#2f6b4c;--bad:#ab5757;--warn:#8d6a32;--chip:#fafcf9;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);-webkit-font-smoothing:antialiased;
font:400 15px/1.55 "IBM Plex Sans",Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:36px 36px 80px}
code,pre,.mono{font-family:"IBM Plex Mono","SF Mono",Menlo,Consolas,monospace}
h1{font-size:27px;font-weight:500;letter-spacing:-1.2px;margin:0 0 6px}
h2{font-size:20px;font-weight:500;letter-spacing:-.65px;margin:34px 0 10px}
.sub{color:var(--muted);font-size:13px}
.chips{margin:16px 0;display:flex;flex-wrap:wrap;gap:10px}
.chip{background:var(--chip);border:1px solid #dee7e1;border-radius:100px;padding:6px 13px;
font-size:12.5px;color:#3f5a4c}
table{width:100%;margin:14px 0;font-size:13.5px;background:var(--panel);border:1px solid var(--line);
border-radius:16px;border-collapse:separate;border-spacing:0}
th,td{text-align:left;padding:11px 14px;border-bottom:1px solid var(--line)}
th{background:#f4f8f4;color:var(--muted);text-transform:uppercase;font-size:11px;
letter-spacing:.06em;font-weight:500}
tr:first-child th:first-child{border-top-left-radius:15px}
tr:first-child th:last-child{border-top-right-radius:15px}
tr:last-child td{border-bottom:0}
.controls{margin:12px 0;color:var(--muted);font-size:13px}
details.row{border:1px solid var(--line);border-radius:14px;margin:10px 0;background:var(--panel);
box-shadow:0 4px 16px #29433004;transition:border-color .2s,box-shadow .2s}
details.row:hover{border-color:#bcd3c2;box-shadow:0 8px 28px #29433009}
details.row>summary{padding:13px 16px;cursor:pointer;list-style:none;display:flex;gap:12px;
align-items:center;flex-wrap:wrap}
summary::-webkit-details-marker{display:none}
summary.lbl{cursor:pointer}
details.row .ic{font-weight:700}.pass .ic{color:var(--good)}.fail .ic{color:var(--bad)}
.tag{color:var(--muted);font-size:12px}
.trace{padding:6px 16px 16px;border-top:1px solid var(--line)}
.lbl{color:var(--accent);font-size:11px;text-transform:uppercase;letter-spacing:.08em;margin:14px 0 6px}
pre{background:#f7faf7;border:1px solid var(--line);border-radius:10px;padding:13px;overflow-x:auto;
white-space:pre-wrap;word-break:break-word;font-size:12.5px;color:#31463c;max-height:480px}
.meta{color:var(--muted);font-size:12px;margin-top:12px}
.res-PASS{color:var(--good)}.res-FAIL{color:var(--bad)}.res-WARN{color:var(--warn)}
body.failonly details.row.pass{display:none}
.err{color:var(--bad)}
input[type=checkbox]{accent-color:var(--accent)}
summary:focus-visible,input:focus-visible{outline:2px solid #347f70;outline-offset:4px}
"""

_JS = """
function flt(cb){document.body.classList.toggle('failonly',cb.checked)}
"""


def _esc(x) -> str:
    return _html.escape(str(x)) if x is not None else ""


def _msgs(messages) -> str:
    return "\n\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in (messages or []))


def _turns(messages) -> str:
    out = []
    for m in messages:
        out.append(f"<div class='lbl'>{_esc(m.get('role', ''))}</div>")
        reasoning = m.get("reasoning_content")
        if reasoning:
            out.append(
                f"<details><summary class='lbl'>reasoning_content ({len(reasoning)} chars)</summary>"
                f"<pre>{_esc(reasoning)}</pre></details>"
            )
        text = records.turn_text(m)
        # A turn whose only output is its reasoning needs no empty text block.
        if text or not reasoning:
            out.append(f"<pre>{_esc(text)}</pre>")
    return "".join(out)


def write_report(run_dir: pathlib.Path, out_path: pathlib.Path) -> None:
    manifest, traces = run_store.load_run(run_dir)
    parts: list[str] = []
    parts.append(
        f"<!doctype html><html><head><meta charset='utf-8'><meta name='color-scheme' content='light'>"
        f"<title>beam-verifier · {_esc(manifest.run_id)}</title><style>{_CSS}</style></head><body><div class='wrap'>"
    )
    parts.append(f"<h1>{_esc(manifest.model)}</h1>")
    parts.append(
        f"<div class='sub'>{_esc(manifest.kind)} run · {_esc(manifest.base_url)} · "
        f"{_esc(manifest.created_at[:19])} · {_esc(manifest.duration_s)}s · beam-verifier {_esc(manifest.version)}</div>"
    )
    parts.append(
        "<div class='chips'>"
        + "".join(
            f"<span class='chip'>{_esc(k)}={_esc(v)}</span>"
            for k, v in manifest.config.model_dump(exclude_none=True).items()
        )
        + "".join(
            f"<span class='chip'>resumed {_esc(k)}={_esc(v)}</span>"
            for applied in manifest.resume_overrides
            for k, v in applied.items()
        )
        + "</div>"
    )

    if manifest.verify is not None:
        parts.append(_verify_html(manifest))
    else:
        parts.append(_eval_html(manifest, traces))

    parts.append(f"</div><script>{_JS}</script></body></html>")
    pathlib.Path(out_path).write_text("".join(parts))


def _eval_html(manifest, traces) -> str:
    out = [
        "<h2>Summary</h2><table><tr><th>dataset</th><th>grade</th><th>status</th>"
        "<th>n</th><th>ok/err</th><th>finish</th></tr>"
    ]
    for dataset_result in manifest.dataset_results:
        topline = dataset_result.metrics.get(dataset_result.topline_metric_name)
        grade = f"{topline:.4g}" if isinstance(topline, (int, float)) else "—"
        finish = " ".join(f"{k}:{v}" for k, v in dataset_result.finish_reasons.items())
        # A recomputed score covers only the samples that finished, and a failed dataset
        # has no score at all; say so rather than letting either read as a completed run.
        if dataset_result.status == "failed":
            reason = f" — {_esc(dataset_result.error)}" if dataset_result.error else ""
            status = f"<span class='res-FAIL'>failed{reason}</span>"
        elif dataset_result.status == "ok":
            status = "complete"
        else:
            status = f"<span class='res-WARN'>partial — {dataset_result.num_ok}/{dataset_result.n} scored</span>"
        out.append(
            f"<tr><td>{_esc(dataset_result.dataset)}</td><td>{grade}</td><td>{status}</td><td>{dataset_result.n}</td>"
            f"<td>{dataset_result.num_ok}/{dataset_result.num_error}</td><td>{_esc(finish)}</td></tr>"
        )
    out.append("</table>")
    for dataset_result in manifest.dataset_results:
        if dataset_result.subset_metrics:
            out.append(_subset_metrics_html(dataset_result.dataset, dataset_result.subset_metrics))
    out.append(
        "<h2>Traces</h2><div class='controls'><label>"
        "<input type='checkbox' onchange='flt(this)'> failed only</label></div>"
    )
    for name, tlist in traces.items():
        for t in tlist:
            failed = records.trace_failed(t)
            cls = "fail" if failed else "pass"
            ic = "✗" if failed else "✓"
            tok = f"{t.prompt_tokens or '-'}/{t.completion_tokens or '-'}"
            errbadge = f" <span class='err'>· {_esc(t.error_type)}</span>" if t.error_type else ""
            scope = f"{name} · {t.subset}" if t.subset else name
            summary = (
                f"<span class='ic'>{ic}</span> <b>#{t.index}</b> <span class='tag'>{_esc(scope)}</span> "
                f"<span class='tag'>exp {_esc(t.expected_answer)} → {_esc(t.parsed_pred)}</span> "
                f"<span class='tag'>{_esc(t.finish_reason)} · {tok} tok · {_esc(t.latency_ms)}ms</span>{errbadge}"
            )
            if any(m.get("role") == "tool" or m.get("tool_calls") or m.get("reasoning_content") for m in t.messages):
                body = [f"<div class='trace'>{_turns(t.messages)}"]
            else:
                body = [f"<div class='trace'><div class='lbl'>prompt</div><pre>{_esc(_msgs(t.messages))}</pre>"]
            if t.reasoning_content:
                body.append(
                    f"<details><summary class='lbl'>reasoning_content ({len(t.reasoning_content)} chars)</summary>"
                    f"<pre>{_esc(t.reasoning_content)}</pre></details>"
                )
            answer = records.answer_text(t)
            body.append(f"<div class='lbl'>answer (graded)</div><pre>{_esc(answer) or '<em>(empty)</em>'}</pre>")
            if t.expected_answer is not None:
                # What the grader extracted is the answer above, and the row's summary
                # already pairs it with this one; repeating it here says nothing new.
                body.append(f"<div class='lbl'>reference answer</div><pre>{_esc(t.expected_answer)}</pre>")
            meta = f"score={t.score} · finish={_esc(t.finish_reason)} · latency={t.latency_ms}ms"
            if t.error_type:
                meta += f" · <span class='err'>{_esc(t.error_type)}: {_esc(t.error_message)}</span>"
            body.append(f"<div class='meta'>{meta}</div></div>")
            out.append(f"<details class='row {cls}'><summary>{summary}</summary>{''.join(body)}</details>")
    return "".join(out)


def _verify_html(manifest) -> str:
    v = manifest.verify
    out = []
    if v.suites:
        out.append("<h2>Functional tests</h2>")
        out.append("<table><tr><th>suite</th><th>status</th><th>cases</th><th>evidence</th></tr>")
        for s in v.suites:
            total = sum(s.counts.values())
            not_failing = total - len(s.failing_ids) - s.counts.get("skipped", 0)
            evidence = "required" if s.required else "optional"
            if s.skip_reason:
                evidence += f" — {s.skip_reason}"
            cls = {"PASS": "res-PASS", "FAIL": "res-FAIL"}.get(s.status, "res-WARN")
            out.append(
                f"<tr><td>{_esc(s.display_name)}</td><td class='{cls}'>{_esc(s.status)}</td>"
                f"<td>{f'{not_failing}/{total}' if total else '—'}</td><td>{_esc(evidence)}</td></tr>"
            )
        out.append("</table>")
    fsum = ", ".join(f"{k}={val}" for k, val in (v.functional or {}).items()) or "n/a"
    out.append(f"<h2>Cases</h2><div class='sub'>{_esc(fsum)}</div>")
    out.append("<table><tr><th>suite</th><th>case</th><th>outcome</th></tr>")
    for c in v.functional_cases:
        oc = c.get("outcome", "")
        cls = "res-FAIL" if oc not in records.BENIGN_OUTCOMES else ("res-PASS" if oc == "passed" else "")
        out.append(
            f"<tr><td>{_esc(c.get('suite', ''))}</td><td class='mono'>{_esc(c.get('name'))}</td>"
            f"<td class='{cls}'>{_esc(oc)}</td></tr>"
        )
    out.append("</table>")
    out.append(f"<h2>{'✓ verify passed' if v.passed else '✗ verify failed'}</h2>")
    return "".join(out)


def _subset_metrics_html(dataset_id: str, subset_metrics: dict[str, dict[str, float]]) -> str:
    """Renders a dataset's per-subset metrics.

    Args:
      dataset_id: The dataset the breakdown belongs to.
      subset_metrics: Subset name to its metrics.

    Returns:
      An HTML fragment.
    """
    metric_names = sorted({name for metrics in subset_metrics.values() for name in metrics})
    out = [
        f"<h2>{_esc(dataset_id)} subsets</h2>",
        "<table><tr><th>subset</th>" + "".join(f"<th>{_esc(name)}</th>" for name in metric_names) + "</tr>",
    ]
    for subset, metrics in subset_metrics.items():
        cells = "".join(f"<td>{f'{metrics[name]:.4g}' if name in metrics else '—'}</td>" for name in metric_names)
        out.append(f"<tr><td>{_esc(subset)}</td>{cells}</tr>")
    out.append("</table>")
    return "".join(out)
