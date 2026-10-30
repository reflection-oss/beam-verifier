"""beam-verifier CLI"""

import datetime
import importlib.metadata
import json
import os
import pathlib
import subprocess
import sys
import urllib.parse
from collections.abc import Sequence
from typing import Any, override

import click
import rich.console
import rich.markup
import rich.table

from beam_verifier import options, records
from beam_verifier.benchmarks import config as dataset_config
from beam_verifier.benchmarks import registry as benchmarks_registry
from beam_verifier.benchmarks.scaffolds import base
from beam_verifier.functional_tests import constants
from beam_verifier.reporting import run_store

console = rich.console.Console()

# Accept --model_name and --model-name alike; add -h.
CTX = {"help_option_names": ["-h", "--help"], "token_normalize_func": lambda x: x.replace("_", "-")}


class _MultiValueCommand(click.Command):
    """Allow grouped values for repeatable options, which Click cannot express with `nargs`.

    Expand groups before Click parses them so its validation and parameter-source
    tracking still apply. Kwargs groups stop at bare dataset names; other groups
    stop at the next option or `--`.
    """

    @override
    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        option_by_flag: dict[str, click.Option] = {}
        for param in self.get_params(ctx):
            if isinstance(param, click.Option):
                option_by_flag.update(dict.fromkeys(param.opts, param))
        expanded: list[str] = []
        index = 0
        while index < len(args):
            arg = args[index]
            index += 1
            expanded.append(arg)
            if arg == "--":
                expanded.extend(args[index:])
                break
            name, separator, _ = arg.partition("=")
            if ctx.token_normalize_func:
                name = ctx.token_normalize_func(name)
            option = option_by_flag.get(name)
            if option is None or option.is_flag:
                continue
            if not separator:
                # Preserve optional values such as bare --resume, and scalar
                # values that happen to look like another option's name.
                if option.flag_value is not None and (
                    index == len(args) or (args[index].startswith("-") and len(args[index]) > 1)
                ):
                    continue
                expanded.extend(args[index : index + option.nargs])
                index += option.nargs
            if option.multiple:
                while index < len(args) and not args[index].startswith("-"):
                    if option.name and option.name.endswith("_kwargs") and "=" not in args[index]:
                        break
                    expanded.extend((name, args[index]))
                    index += 1
        return super().parse_args(ctx, expanded)


@click.group(context_settings=CTX)
@click.version_option(package_name="beam-verifier")
def cli():
    """beam-verifier — eval a model's scores, or verify an endpoint's functionality."""


def check_endpoint(base_url: str, model_name: str, api_key: str) -> bool:
    """Probes `/v1/models` and reports whether the endpoint can serve the model.

    An unreachable endpoint or a rejected key is fatal, as is an endpoint that
    lists its models authoritatively without listing this one. Anything else — no
    `/v1/models` route, an unexpected failure — is only a warning, since the
    endpoint may still serve chat completions.

    Args:
        base_url: OpenAI-compatible base URL, ending in `/v1`.
        model_name: Model id, exactly as the endpoint names it.
        api_key: Key for the endpoint.

    Returns:
        Whether the run should proceed.
    """
    import openai  # noqa: PLC0415

    models = None
    try:
        models = [model.id for model in openai.OpenAI(base_url=base_url, api_key=api_key).models.list()]
    except (openai.APIConnectionError, openai.AuthenticationError, openai.PermissionDeniedError) as error:
        console.print(f"[red]error:[/] {base_url}: {error}")
        return False
    except openai.NotFoundError:
        console.print(f"[yellow]warning:[/] {base_url} has no /v1/models route; skipping the model check.")
    except Exception as error:  # noqa: BLE001 - a failed check must not stop a servable endpoint
        console.print(f"[yellow]warning:[/] couldn't list models at {base_url} ({type(error).__name__}); proceeding.")
    if models and model_name not in models:
        console.print(f"[red]error:[/] model '{model_name}' not served at {base_url} (have: {', '.join(models[:6])}…).")
        return False
    return True


def validate_input_args(
    dataset_ids: list[str], num_examples: int | None = None, parallel_tasks: int | None = None
) -> None:
    """Validates the input run arguments.

    Args:
        dataset_ids: Dataset ids the run will execute.
        num_examples: Cap on examples per dataset, or None for every example.
        parallel_tasks: Tasks to run in parallel, or None for the default.

    Raises:
        ValueError: If a dataset is not registered, or a count is below one.
    """
    unsupported = [dataset_id for dataset_id in dataset_ids if dataset_id not in benchmarks_registry.DATASETS]
    if unsupported:
        raise ValueError(f"unknown dataset(s): {', '.join(unsupported)}. Try `beam-verifier list-datasets`.")
    if num_examples is not None and num_examples < 1:
        raise ValueError(f"--num-examples must be at least 1, got {num_examples}.")
    if parallel_tasks is not None and parallel_tasks < 1:
        raise ValueError(f"--parallel-tasks must be at least 1, got {parallel_tasks}.")


@cli.command(name="list-datasets")
def list_datasets_command():
    """List available datasets."""
    table = rich.table.Table(title="Datasets")
    table.add_column("dataset id", style="cyan")
    table.add_column("runner")
    table.add_column("status")
    table.add_column("description")
    for dataset in benchmarks_registry.DATASETS.values():
        table.add_row(
            dataset.dataset_id,
            dataset.runner.MODULE,
            dataset.development_status,
            (dataset.description or dataset.display_name or dataset.dataset_id)[:70],
        )
    console.print(table)


def _load_resume(
    resume: str,
    *,
    base_url: str,
    model_name: str,
    settings: records.EvalSettings,
    datasets: tuple[str, ...],
    output_dir: str | None,
) -> tuple[pathlib.Path, records.RunManifest, records.EvalSettings]:
    """Loads a saved eval plan and applies the settings the caller named again.

    A resume reuses the settings in effect for the run, which are the ones it was
    launched with plus every override a previous resume recorded. The runner rejects
    unsupported overrides before the saved manifest changes.
    """
    run_dir = run_store.resolve_run(None if resume == "latest" else resume)
    if run_dir is None:
        raise click.ClickException("--resume: no saved run found.")
    try:
        manifest = run_store.load_manifest(run_dir)
    except (OSError, ValueError) as error:
        raise click.ClickException(f"--resume: cannot read {run_dir}/run.json: {error}") from error
    if manifest.kind != "eval":
        raise click.ClickException("--resume requires an eval run.")
    if (model_name, base_url) != (manifest.model, manifest.base_url):
        raise click.ClickException("--resume requires the saved model and base URL.")
    if output_dir and pathlib.Path(output_dir).resolve() != run_dir.resolve():
        raise click.ClickException("--resume writes into the saved run; drop --output-dir.")
    selected_ids = list(dict.fromkeys(datasets)) or manifest.dataset_ids
    if manifest.dataset_ids and set(selected_ids) != set(manifest.dataset_ids):
        raise click.ClickException("--resume cannot change the saved dataset list.")
    if not manifest.dataset_ids or set(selected_ids) - manifest.dataset_settings.keys():
        raise click.ClickException("--resume requires saved dataset settings; start a new eval run.")
    context = click.get_current_context()
    in_effect = manifest.config
    for applied in manifest.resume_overrides:
        in_effect = dataset_config.merge(in_effect, records.EvalSettings.model_validate(applied))
    overrides = {
        name: getattr(settings, name)
        for name in records.EvalSettings.model_fields
        if context.get_parameter_source(name) == click.core.ParameterSource.COMMANDLINE
    }
    supplied = records.EvalSettings.model_validate(overrides)
    changed: set[str] = set()
    for dataset_id in _pending_datasets(manifest):
        saved = manifest.dataset_settings[dataset_id]
        requested = dataset_config.merge(saved, supplied)
        before, after = saved.model_dump(mode="json"), requested.model_dump(mode="json")
        # Native options can distinguish booleans from integers, such as shuffle seeds.
        dataset_changes = {
            name
            for name in overrides
            if json.dumps(before[name], sort_keys=True) != json.dumps(after[name], sort_keys=True)
        }
        changed_settings = records.EvalSettings.model_validate(
            {
                name: {
                    key: value
                    for key, value in after[name].items()
                    if key not in before[name]
                    or json.dumps(before[name][key], sort_keys=True) != json.dumps(value, sort_keys=True)
                }
                if name in ("harness_kwargs", "model_kwargs")
                else after[name]
                for name in dataset_changes
            }
        )
        unsupported = benchmarks_registry.DATASETS[dataset_id].runner.unsupported_resume_overrides(
            changed_settings, dataset_changes
        )
        if unsupported:
            raise click.ClickException(
                f"--resume: {dataset_id} cannot apply changed settings: {', '.join(sorted(unsupported))}; start a new run."
            )
        changed.update(dataset_changes)
        manifest.dataset_settings[dataset_id] = requested
    overrides = {name: value for name, value in overrides.items() if name in changed}
    for name, value in overrides.items():
        option = f"--{name.replace('_', '-')}"
        console.print(f"[yellow]warning:[/] --resume: {option}={value} overrides saved dataset settings")
    if overrides:
        manifest.resume_overrides.append(overrides)
    return run_dir, manifest, dataset_config.merge(in_effect, records.EvalSettings.model_validate(overrides))


@cli.command(name="eval", cls=_MultiValueCommand)
@options.endpoint_arguments
@click.argument("datasets", nargs=-1, metavar="[DATASET ...]")
@click.option("--num-examples", type=int, default=None, help="Cap examples per dataset.")
@click.option(
    "--parallel-tasks",
    type=int,
    default=None,
    help="Tasks run in parallel against the endpoint (a task may make more than one request). ",
)
@click.option(
    "--temperature",
    type=float,
    default=None,
    help="Sampling temperature. Unset sends nothing, leaving the choice to the endpoint.",
)
@click.option(
    "--max-tokens",
    type=int,
    default=None,
    help="Generation cap. Unset sends nothing, leaving the choice to the endpoint.",
)
@click.option("--top-p", type=float, default=None)
@click.option(
    "--reasoning-effort",
    type=click.Choice(["low", "medium", "high", "xhigh", "max"]),
    default=None,
    help="Reasoning effort sent with each request.",
)
@click.option(
    "--client-timeout",
    type=int,
    default=600,
    show_default=True,
    help="Per-request client timeout in seconds. A scaffold that streams responses applies it "
    "to the gap between chunks (a true hang) rather than to total generation time.",
)
@options.kwargs_option(
    "harness",
    "Keyword argument forwarded verbatim to the API of the eval library that runs the dataset "
    "(e.g. retry_on_error=5, epochs=2 for InspectAI), winning over the dedicated flags on a name "
    "collision.",
)
@options.model_kwargs_option
@click.option(
    "--sandbox",
    type=click.Choice(["docker", "daytona"]),
    default="docker",
    show_default=True,
    help="Sandbox provider for coding datasets. Daytona requires beam-verifier[daytona].",
)
@click.option(
    "--host-model-proxy",
    is_flag=True,
    help="Relay model requests through this host, keeping endpoint credentials outside sandboxes. "
    "Requires --sandbox daytona; local Docker does not support this flag.",
)
@click.option(
    "--output-dir", default=None, help="Run directory (default: .runs/<ts>-<model>-eval at the package root)."
)
@click.option(
    "--resume",
    is_flag=False,
    flag_value="latest",
    default=None,
    metavar="RUN",
    help="Continue a saved eval run; bare --resume uses the latest run. Saved settings are reused, and a setting named again overrides the saved value where the harness accepts it.",
)
def eval_command(
    base_url,
    model_name,
    api_key,
    datasets,
    num_examples,
    parallel_tasks,
    temperature,
    max_tokens,
    top_p,
    reasoning_effort,
    client_timeout,
    harness_kwargs,
    model_kwargs,
    sandbox,
    host_model_proxy,
    output_dir,
    resume,
):
    """Run evals on dataset(s), save a run (traces + results), and print scores.

    Specify the datasets to run, in order; omit them to run the default set.
    `list-datasets` prints the available dataset ids.
    """
    selected_ids = list(dict.fromkeys(datasets)) or [
        dataset_id for dataset_id, dataset in benchmarks_registry.DATASETS.items() if dataset.run_by_default
    ]
    try:
        parsed_harness_kwargs = options.parse_kwargs(harness_kwargs, option_name="--harness-kwargs")
        parsed_model_kwargs = options.parse_model_kwargs(model_kwargs)
        if resume is None:
            validate_input_args(selected_ids, num_examples, parallel_tasks)
    except ValueError as error:
        console.print(f"[red]error:[/] {error}")
        sys.exit(1)

    # Retain which values the caller supplied: a Click default must not overwrite
    # a dataset's YAML default, even when an explicit value equals Click's default.
    settings_values = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "reasoning_effort": reasoning_effort,
        "num_examples": num_examples,
        "parallel_tasks": parallel_tasks,
        "client_timeout": client_timeout,
        "harness_kwargs": parsed_harness_kwargs,
        "model_kwargs": parsed_model_kwargs,
        "sandbox": sandbox,
        "host_model_proxy": host_model_proxy,
    }
    context = click.get_current_context()
    settings = records.EvalSettings.model_validate(
        {
            name: value
            for name, value in settings_values.items()
            if context.get_parameter_source(name) not in (None, click.core.ParameterSource.DEFAULT)
        }
    )
    started = datetime.datetime.now()
    if resume is not None:
        run_dir, manifest, settings = _load_resume(
            resume,
            base_url=base_url,
            model_name=model_name,
            settings=settings,
            datasets=datasets,
            output_dir=output_dir,
        )
        selected_ids = manifest.dataset_ids
        console.print(f"Resuming {run_dir}")
    else:
        run_dir = pathlib.Path(output_dir) if output_dir else run_store.new_run_dir(model_name, "eval")
        if (run_dir / "run.json").exists():
            raise click.ClickException(f"{run_dir} already contains a run; use --resume or a fresh output directory.")
        try:
            dataset_settings = {
                name: benchmarks_registry.DATASETS[name].resolve_settings(settings) for name in selected_ids
            }
        except ValueError as error:
            raise click.ClickException(str(error)) from error
        manifest = records.RunManifest(
            run_id=run_dir.name,
            kind="eval",
            created_at=started.isoformat(),
            version=importlib.metadata.version("beam-verifier"),
            model=model_name,
            base_url=base_url,
            config=settings,
            dataset_ids=selected_ids,
            dataset_settings=dataset_settings,
        )

    try:
        validate_input_args(selected_ids, settings.num_examples, settings.parallel_tasks)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    pending_ids = _pending_datasets(manifest)

    if pending_ids and not check_endpoint(base_url, model_name, api_key):
        sys.exit(1)
    failed_ids = run_datasets(
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        manifest=manifest,
        run_dir=run_dir,
        resume=resume is not None,
    )

    table = rich.table.Table(title=model_name, title_style="bold yellow")
    for c in ("dataset", "grade", "status", "n", "ok/err", "finish"):
        table.add_column(c, style="cyan" if c == "dataset" else None)
    for dataset_result in manifest.dataset_results:
        if dataset_result.status == "failed":
            table.add_row(dataset_result.dataset, "—", "[red]failed[/]", "0", "0/0", "")
            continue
        topline = dataset_result.metrics.get(dataset_result.topline_metric_name)
        finish = " ".join(f"{k}:{v}" for k, v in dataset_result.finish_reasons.items())
        table.add_row(
            dataset_result.dataset,
            f"{topline:.4g}" if isinstance(topline, (int, float)) else "—",
            "complete"
            if dataset_result.status == "ok"
            else f"[yellow]partial[/] ({dataset_result.num_ok}/{dataset_result.n} scored)",
            str(dataset_result.n),
            f"{dataset_result.num_ok}/{dataset_result.num_error}",
            finish,
        )

    console.print("")
    console.print(table)
    errs: dict[str, int] = {}
    for dataset_result in manifest.dataset_results:
        for k, v in dataset_result.error_types.items():
            errs[k] = errs.get(k, 0) + v
    if errs:
        console.print("[yellow]errors:[/] " + ", ".join(f"{k}={v}" for k, v in errs.items()))
    console.print(f"\nRun saved to {run_dir}")
    console.print(f"  view:   beam-verifier view {run_dir}")
    console.print(f"  report: beam-verifier report {run_dir}")

    # Exit after the run is saved, so a caller keeps whatever did finish: the
    # scores are on disk either way, and only the status says they are complete.
    if failed_ids:
        console.print(f"\n[red]failed:[/] {', '.join(failed_ids)}")
        sys.exit(1)


def _pending_datasets(manifest: records.RunManifest) -> list[str]:
    """The manifest's datasets without a complete result, in run order."""
    results = {result.dataset: result for result in manifest.dataset_results}
    return [
        dataset_id
        for dataset_id in manifest.dataset_ids
        if dataset_id not in results or results[dataset_id].status != "ok"
    ]


def run_datasets(
    *,
    base_url: str,
    model_name: str,
    api_key: str,
    manifest: records.RunManifest,
    run_dir: pathlib.Path,
    resume: bool = False,
) -> list[str]:
    """Runs the manifest's datasets that are not complete yet, and returns those that failed or came back partial.

    The manifest already contains resolved settings for every dataset. It is saved
    before the first dataset and after each one, so an interrupted run can resume.
    """
    store = run_store.RunStore(run_dir)
    results = {result.dataset: result for result in manifest.dataset_results}
    pending_ids = _pending_datasets(manifest)
    if pending_ids:
        manifest.finished_at = None
        # A kill during the first dataset must still leave a resumable run plan.
        store.write_manifest(manifest)

    # Datasets that produced no score, or only a partial one.
    failed_ids: list[str] = []

    for dataset_id in pending_ids:
        console.print(f"\n[bold blue]Evaluating {dataset_id}[/bold blue]")
        dataset = benchmarks_registry.DATASETS[dataset_id]
        effective_settings = manifest.dataset_settings[dataset_id]
        try:
            outcome = dataset.runner(
                base.RunRequest(
                    base_url=base_url,
                    model_name=model_name,
                    api_key=api_key,
                    run_dir=run_dir,
                    settings=effective_settings,
                    resume=resume,
                )
            )
        except base.RunnerError as error:
            console.print(f"error: {error}", style="red", markup=False)
            failed_ids.append(dataset_id)
            # Record the failure: a saved run must tell an attempted dataset that failed
            # apart from one that was never asked for.
            results[dataset_id] = records.DatasetResult(
                dataset=dataset_id, topline_metric_name="", status="failed", error=str(error)
            )
            manifest.dataset_results = list(results.values())
            store.write_manifest(manifest)
            continue
        if outcome.warning:
            console.print(f"[red]error:[/] {dataset_id}: {outcome.warning}")
        # A partial result fails the run even when the eval itself reported
        # success: scoring can be skipped or yield no aggregate metric, leaving a row
        # with no number to read.
        if outcome.warning or outcome.result.status != "ok":
            failed_ids.append(dataset_id)
        dataset_result = outcome.result
        store.write_traces(dataset_id, outcome.traces)
        results[dataset_id] = dataset_result
        manifest.dataset_results = list(results.values())
        # Persist after each dataset: a long run that is interrupted later should not
        # lose the datasets that already finished.
        store.write_manifest(manifest)

    if pending_ids or manifest.finished_at is None:
        started = datetime.datetime.fromisoformat(manifest.created_at)
        finished = datetime.datetime.now(tz=started.tzinfo)
        manifest.finished_at = finished.isoformat()
        manifest.duration_s = round((finished - started).total_seconds(), 1)
        store.write_manifest(manifest)

    return list(
        dict.fromkeys(failed_ids + [result.dataset for result in manifest.dataset_results if result.status != "ok"])
    )


def _endpoint_url(ctx: click.Context, param: click.Parameter, value: str | None) -> str | None:
    """Require an absolute HTTP URL while preserving the exact path and query."""
    if value is not None:
        url = urllib.parse.urlsplit(value)
        if url.scheme not in ("http", "https") or not url.netloc or url.fragment:
            raise click.BadParameter("must be an absolute HTTP(S) URL without a fragment", ctx, param)
    return value


@cli.command(name="verify", cls=_MultiValueCommand, context_settings={**CTX, "ignore_unknown_options": True})
@click.option("--base-url", metavar="URL", help="OpenAI-compatible base URL, ending in /v1.")
@click.option("--model-name", "model_name", metavar="NAME", help="Model id; required for API tests.")
@click.option(
    "--parse-url", callback=_endpoint_url, help="Exact URL accepting token IDs and returning an assistant message."
)
@click.option("--format-url", callback=_endpoint_url, help="Exact URL accepting chat inputs and returning token IDs.")
@click.option(
    "--completions-url", callback=_endpoint_url, help="Exact raw completions URL; runs Harmony generation checks."
)
@click.option(
    "--api-key",
    envvar="REFLECTION_API_KEY",
    default="EMPTY",
    help="API key; defaults to $REFLECTION_API_KEY, else 'EMPTY'.",
)
@options.model_kwargs_option
@click.argument("pytest_args", nargs=-1, type=click.UNPROCESSED)
def verify_command(
    base_url: str | None,
    model_name: str | None,
    parse_url: str | None,
    format_url: str | None,
    completions_url: str | None,
    api_key: str,
    model_kwargs: tuple[str, ...],
    pytest_args: tuple[str, ...],
) -> None:
    """Verify endpoint behavior with the functional conformance tests.

    `--model-kwargs` reaches every generation the suite elicits, so one suite
    runs against any engine. The parser and formatter suites post token ids to
    native endpoints rather than issuing model requests, and are unaffected.
    """
    try:
        parsed_model_kwargs = options.parse_model_kwargs(model_kwargs)
    except ValueError as error:
        raise click.BadParameter(str(error), param_hint="--model-kwargs") from error
    suites_to_run = [
        name
        for name, url in (
            ("e2e", base_url),
            ("parser", parse_url),
            ("formatter", format_url),
            ("raw_completions", completions_url),
        )
        if url
    ] or ["e2e"]
    if "e2e" in suites_to_run:
        if not base_url:
            raise click.UsageError("--base-url is required for API tests")
        if not model_name:
            raise click.UsageError("--model-name is required for API tests")
        if not check_endpoint(base_url, model_name, api_key):
            sys.exit(1)

    started = datetime.datetime.now()
    run_dir = run_store.new_run_dir(model_name or "-".join(suites_to_run), "verify")
    store = run_store.RunStore(run_dir)

    # Functional tests, captured via pytest-json-report.
    console.rule("[bold]Functional tests")
    import beam_verifier.functional_tests as ft  # noqa: PLC0415

    tests_dir = pathlib.Path(ft.__file__).parent
    report_path = run_dir / "verify_report.json"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *(str(tests_dir / suite) for suite in suites_to_run),
        "-v",
        "--json-report",
        f"--json-report-file={report_path}",
    ]
    if base_url is not None:
        cmd.append(f"--base-url={base_url}")
    if model_name is not None:
        cmd.append(f"--model={model_name}")
    if parse_url is not None:
        cmd.append(f"--parse-url={parse_url}")
    if format_url is not None:
        cmd.append(f"--format-url={format_url}")
    if completions_url is not None:
        cmd.append(f"--completions-url={completions_url}")
    cmd.extend(f"--model-kwargs={pair}" for pair in model_kwargs)
    cmd.extend(pytest_args)
    env = os.environ.copy()
    env["PYTEST_API_KEY"] = api_key
    tests_rc = subprocess.run(cmd, env=env).returncode

    functional, cases = parse_pytest_report(report_path)
    suites = summarize_suites(cases, required_suites=suites_to_run)
    ok = tests_rc == 0 and all(s.status == "PASS" for s in suites if s.required)
    verify = records.VerifyResult(
        suites=suites,
        evidence=[s.suite for s in suites if s.status != "SKIP"],
        functional=functional,
        functional_cases=cases,
        passed=ok,
    )
    finished = datetime.datetime.now()
    manifest = records.RunManifest(
        run_id=run_dir.name,
        kind="verify",
        created_at=started.isoformat(),
        finished_at=finished.isoformat(),
        duration_s=round((finished - started).total_seconds(), 1),
        version=importlib.metadata.version("beam-verifier"),
        model=model_name or "",
        base_url=base_url or parse_url or format_url or completions_url or "",
        config=records.EvalSettings(
            client_timeout=constants.REQUEST_TIMEOUT_SECONDS,
            model_kwargs=parsed_model_kwargs,
            parse_url=parse_url,
            format_url=format_url,
            completions_url=completions_url,
        ),
        verify=verify,
    )
    store.write_manifest(manifest)

    console.rule("[bold]Result")
    from beam_verifier.reporting import view as view_mod  # noqa: PLC0415

    view_mod.render_verify(verify)
    console.print(f"\nRun saved to {run_dir}\n  view: beam-verifier view {run_dir}")
    sys.exit(0 if ok else 1)


def parse_pytest_report(path: pathlib.Path) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Return (counts, cases) from a pytest-json-report file.

    Each case is {"suite", "name", "outcome"}: `suite` is the test's top-level
    directory under functional_tests/, and `name` its exact node id from
    functional_tests/ down, so a failing case can be rerun with pytest as-is.
    """
    counts: dict[str, int] = {}
    cases: list[dict[str, Any]] = []
    try:
        data = json.loads(path.read_text())
    except Exception:
        return counts, cases
    for t in data.get("tests", []):
        outcome = t.get("outcome", "unknown")
        counts[outcome] = counts.get(outcome, 0) + 1
        nodeid = t.get("nodeid", "")
        file_part, _, test_part = nodeid.partition("::")
        parts = pathlib.PurePosixPath(file_part).parts
        suite = ""
        name = nodeid
        if "functional_tests" in parts:
            below = parts[parts.index("functional_tests") + 1 :]
            if len(below) > 1:
                suite = below[0]
            name = "/".join(below) + (f"::{test_part}" if test_part else "")
        metadata = t.get("metadata") or {}
        rule = metadata.get("rule", "")
        cases.append({"suite": suite, "name": name, "outcome": outcome, "rule": rule, "metadata": metadata})
    return counts, cases


def summarize_suites(
    cases: list[dict[str, Any]], *, required_suites: Sequence[str] | None = None
) -> list[records.SuiteResult]:
    """Aggregate parsed cases into per-suite results, in registry order.

    A suite whose cases all skipped (or that collected nothing) is SKIP; an
    optional suite's skip is explained by the endpoint option that gates it.
    Directories outside the registry are reported as optional suites so a new
    suite is never silently dropped from the report.
    """
    import beam_verifier.functional_tests as ft  # noqa: PLC0415

    by_suite: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_suite.setdefault(case.get("suite", ""), []).append(case)

    registered = {spec.directory for spec in ft.SUITES}
    specs = list(ft.SUITES) + [
        ft.SuiteSpec(directory=d, display_name=d, required=False) for d in sorted(by_suite) if d and d not in registered
    ]
    results = []
    for spec in specs:
        suite_cases = by_suite.get(spec.directory, [])
        counts_by_outcome: dict[str, int] = {}
        for case in suite_cases:
            counts_by_outcome[case["outcome"]] = counts_by_outcome.get(case["outcome"], 0) + 1
        failing = [case["name"] for case in suite_cases if case["outcome"] not in records.BENIGN_OUTCOMES]
        ran = [case for case in suite_cases if case["outcome"] != "skipped"]
        if failing:
            status, reason = "FAIL", None
        elif ran:
            incomplete_conformance = spec.directory in (
                "parser",
                "formatter",
                "raw_completions",
            ) and counts_by_outcome.get("skipped")
            status, reason = ("FAIL", "some cases skipped") if incomplete_conformance else ("PASS", None)
        elif suite_cases:
            status, reason = "SKIP", "all cases skipped"
        elif required_suites is not None and spec.directory not in required_suites:
            status, reason = "SKIP", "not selected"
        else:
            status, reason = "SKIP", f"no {spec.gate}" if spec.gate else "no tests ran"
        skip_reasons = sorted({case.get("metadata", {}).get("skip_reason", "") for case in suite_cases} - {""})
        if reason and skip_reasons:
            reason = "; ".join(skip_reasons)
        results.append(
            records.SuiteResult(
                suite=spec.directory,
                display_name=spec.display_name,
                required=spec.required if required_suites is None else spec.directory in required_suites,
                status=status,
                counts=counts_by_outcome,
                failing_ids=failing,
                skip_reason=reason,
            )
        )
    return results


@cli.command(name="view")
@click.argument("run", required=False)
@click.option("--dataset", default=None, help="Limit to one dataset.")
@click.option("--row", "row", type=int, default=None, help="Show the full trace for one row index.")
@click.option("--failed", is_flag=True, help="Only rows that scored 0 / errored.")
@click.option("--errors", is_flag=True, help="Only rows with an error.")
@click.option("--full", is_flag=True, help="Show full reasoning_content (default: truncated).")
def view_command(run, dataset, row, failed, errors, full):
    """View a saved run (defaults to the latest)."""
    from beam_verifier.reporting import view as view_mod  # noqa: PLC0415

    run_dir = run_store.resolve_run(run)
    if run_dir is None or not pathlib.Path(run_dir).exists():
        console.print("[red]error:[/] no run found. Pass a run dir, or run `beam-verifier eval` first.")
        sys.exit(1)
    view_mod.render(pathlib.Path(run_dir), dataset=dataset, row=row, failed=failed, errors=errors, full=full)


@cli.command(name="report")
@click.argument("run", required=False)
@click.option("-o", "--output", default=None, help="HTML output path (default: <run>/report.html).")
def report_command(run, output):
    """Generate a self-contained HTML report for a run (defaults to the latest)."""
    from beam_verifier.reporting import html as html_mod  # noqa: PLC0415

    run_dir = run_store.resolve_run(run)
    if run_dir is None or not pathlib.Path(run_dir).exists():
        console.print("[red]error:[/] no run found. Pass a run dir, or run `beam-verifier eval` first.")
        sys.exit(1)
    out = pathlib.Path(output) if output else pathlib.Path(run_dir) / "report.html"
    html_mod.write_report(pathlib.Path(run_dir), out)
    console.print(f"HTML report: {out}")


if __name__ == "__main__":
    cli()
