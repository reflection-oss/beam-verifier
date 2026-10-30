"""Offline tests for CLI options and functional-test discovery."""

import importlib.metadata
import json
import pathlib
import shutil

import click.testing
import pytest

from beam_verifier import cli, functional_tests, options, records
from beam_verifier.benchmarks import registry
from beam_verifier.benchmarks.scaffolds import base
from beam_verifier.reporting import run_store


@pytest.fixture(autouse=True)
def endpoint_checks(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record endpoint checks without making network requests."""
    checked: list[str] = []

    def check_endpoint(base_url: str, model_name: str, api_key: str) -> bool:
        del model_name, api_key  # Unused.
        checked.append(base_url)
        return True

    monkeypatch.setattr(cli, "check_endpoint", check_endpoint)
    return checked


def _option(command: click.Command, name: str) -> click.Option:
    """Finds one option on a command by its destination name.

    Args:
      command: Command to search.
      name: Destination parameter name, e.g. `base_url`.

    Returns:
      The matching option.

    Raises:
      AssertionError: If the command has no such option.
    """
    for param in command.params:
        if isinstance(param, click.Option) and param.name == name:
            return param
    raise AssertionError(f"{command.name} has no {name} option")


ENDPOINT_COMMANDS = ("eval", "verify")


def test_cli_exposes_endpoint_evaluation_and_reporting_commands():
    assert set(cli.cli.commands) == {"verify", "eval", "list-datasets", "view", "report"}


def test_version_reports_the_installed_distribution_without_an_endpoint():
    result = click.testing.CliRunner().invoke(cli.cli, ["--version"])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == f"cli, version {importlib.metadata.version('beam-verifier')}"


@pytest.mark.parametrize("command_name", ENDPOINT_COMMANDS)
def test_endpoint_arguments_have_exactly_one_spelling(command_name: str):
    """One spelling per value: an alias would give the same value two names to learn,
    and a second click.option would silently shadow the first."""
    command = cli.cli.commands[command_name]
    assert _option(command, "base_url").opts == ["--base-url"]
    assert _option(command, "model_name").opts == ["--model-name"]
    assert _option(command, "api_key").opts == ["--api-key"]


@pytest.mark.parametrize("command_name", ENDPOINT_COMMANDS)
@pytest.mark.parametrize(
    ("reflection_key", "key_args", "expected"),
    [
        ("synthetic-reflection-key", [], "synthetic-reflection-key"),
        ("synthetic-reflection-key", ["--api-key", "synthetic-explicit-key"], "synthetic-explicit-key"),
        (None, [], "EMPTY"),
    ],
)
def test_endpoint_commands_use_reflection_api_key(
    command_name: str,
    reflection_key: str | None,
    key_args: list[str],
    expected: str,
) -> None:
    env = {
        "REFLECTION_API_KEY": reflection_key,
        "OPENAI_API_KEY": "synthetic-ignored-key",
        "BEAM_VERIFIER_API_KEY": "synthetic-ignored-key",
    }
    command = cli.cli.commands[command_name]
    args = ["--base-url", "https://example.test/v1", "--model-name", "model", *key_args]
    with click.testing.CliRunner().isolation(env=env), command.make_context(command_name, args) as context:
        assert context.params["api_key"] == expected


@pytest.mark.parametrize(
    ("arguments", "missing"),
    [(["--model-name", "m"], "--base-url"), (["--base-url", "http://unused/v1"], "--model-name")],
)
def test_eval_requires_endpoint_and_model(arguments: list[str], missing: str) -> None:
    result = click.testing.CliRunner().invoke(cli.cli, ["eval", *arguments])
    assert result.exit_code == 2
    assert missing in result.output


def test_eval_rejects_numeric_reasoning_effort():
    result = click.testing.CliRunner().invoke(cli.cli, ["eval", "--reasoning-effort", "512"])
    assert result.exit_code == 2
    assert "'512' is not one of 'low', 'medium', 'high', 'xhigh', 'max'" in result.output


def test_eval_rejects_an_unknown_dataset(endpoint_checks: list[str]) -> None:
    result = click.testing.CliRunner().invoke(
        cli.cli, ["eval", "--base-url", "https://endpoint.example.com/v1", "--model-name", "m", "nope"]
    )
    assert result.exit_code == 1
    assert "nope" in result.output
    assert endpoint_checks == []


def test_parse_kwargs_decodes_yaml_values():
    parsed = options.parse_kwargs(("retry_on_error=5", "fail_on_error=false", "tags=[a, b]"))
    assert parsed == {"retry_on_error": 5, "fail_on_error": False, "tags": ["a", "b"]}


def test_parse_kwargs_turns_dashes_into_underscores():
    # A key typed the CLI way must land as the library's keyword argument.
    assert options.parse_kwargs(("retry-on-error=5",)) == {"retry_on_error": 5}


def test_parse_kwargs_without_separator_is_rejected():
    with pytest.raises(ValueError, match="expected KEY=VALUE"):
        options.parse_kwargs(("epochs",))


def test_parse_kwargs_with_malformed_yaml_is_rejected():
    with pytest.raises(ValueError, match="malformed --harness-kwargs"):
        options.parse_kwargs(("value={unclosed",))


@pytest.mark.parametrize("flag", ["--num-examples", "--parallel-tasks"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_eval_rejects_nonpositive_counts(flag: str, value: str, endpoint_checks: list[str]) -> None:
    result = click.testing.CliRunner().invoke(
        cli.cli,
        ["eval", "--base-url", "https://endpoint.example.com/v1", "--model-name", "m", flag, value],
    )
    assert result.exit_code == 1
    assert flag in result.output
    assert endpoint_checks == []


@pytest.mark.parametrize("option", ["--parse-url", "--format-url", "--completions-url"])
def test_conformance_url_does_not_require_base_url_or_model(option: str) -> None:
    command = cli.cli.commands["verify"]
    with command.make_context("verify", [option, "https://example.test/custom?version=2"]) as context:
        assert context.params["base_url"] is None
        assert context.params["model_name"] is None
        assert context.params[option[2:].replace("-", "_")] == "https://example.test/custom?version=2"


def test_api_suite_still_requires_base_url() -> None:
    result = click.testing.CliRunner().invoke(cli.cli, ["verify", "--model-name", "m"])
    assert result.exit_code == 2
    assert "--base-url is required for API tests" in result.output


@pytest.mark.parametrize(
    "extra_urls",
    [
        [],
        [
            "--parse-url",
            "https://example.test/parse",
            "--format-url",
            "https://example.test/format",
            "--completions-url",
            "https://example.test/v1/completions",
        ],
    ],
    ids=["api-only", "all-suites"],
)
def test_api_suite_still_requires_model(extra_urls: list[str]) -> None:
    result = click.testing.CliRunner().invoke(cli.cli, ["verify", "--base-url", "https://example.test/v1", *extra_urls])
    assert result.exit_code == 2
    assert "--model-name is required for API tests" in result.output


@pytest.mark.parametrize("option", ["--parse-url", "--completions-url"])
@pytest.mark.parametrize("url", ["/parse", "ftp://example.test/parse", "https://example.test/parse#fragment"])
def test_verify_rejects_invalid_endpoint_url(option: str, url: str) -> None:
    result = click.testing.CliRunner().invoke(cli.cli, ["verify", option, url])
    assert result.exit_code == 2
    assert "absolute HTTP(S) URL" in result.output


@pytest.mark.parametrize(
    ("command_name", "scope"),
    [("eval", "model"), ("eval", "harness"), ("verify", "model")],
)
def test_commands_accept_grouped_and_repeated_kwargs(command_name: str, scope: str) -> None:
    option = f"--{scope}-kwargs"
    with cli.cli.commands[command_name].make_context(
        command_name,
        [
            "--base-url",
            "http://localhost/v1",
            "--model-name",
            "model",
            option,
            "stop=[read, write]",
            "stream=false",
            "seed=-2",
            "extra_body={custom_option: false}",
            option,
            "stream=true",
            "user=Run tests = always.",
        ],
    ) as context:
        assert options.parse_kwargs(context.params[f"{scope}_kwargs"], option_name=option) == {
            "stop": ["read", "write"],
            "stream": True,
            "seed": -2,
            "extra_body": {"custom_option": False},
            "user": "Run tests = always.",
        }
        assert context.get_parameter_source(f"{scope}_kwargs") == click.core.ParameterSource.COMMANDLINE


def test_eval_grouped_kwargs_preserve_scopes_and_positional_datasets():
    with cli.eval_command.make_context(
        "eval",
        [
            "gpqa-diamond",
            "--base-url=http://localhost/v1",
            "--model-name=model",
            "--model_kwargs=top_k=50",
            "extra_body={custom_option: false}",
            "aime2026",
            "--harness-kwargs",
            "retry_on_error=2",
            "epochs=3",
            "--",
            "literal=dataset",
            "--model-kwargs",
        ],
        **cli.CTX,
    ) as context:
        assert context.params["datasets"] == ("gpqa-diamond", "aime2026", "literal=dataset", "--model-kwargs")
        assert context.params["model_kwargs"] == ("top_k=50", "extra_body={custom_option: false}")
        assert context.params["harness_kwargs"] == ("retry_on_error=2", "epochs=3")


def test_eval_preserves_option_names_used_as_scalar_values():
    with cli.eval_command.make_context(
        "eval",
        [
            "--base-url",
            "http://localhost/v1",
            "--model-name",
            "--model-kwargs",
            "--harness-kwargs",
            "retry_on_error=2",
            "epochs=3",
            "--temperature",
            "-0.1",
        ],
    ) as context:
        assert context.params["model_name"] == "--model-kwargs"
        assert context.params["model_kwargs"] == ()
        assert context.params["harness_kwargs"] == ("retry_on_error=2", "epochs=3")
        assert context.params["temperature"] == -0.1


@pytest.mark.parametrize(
    ("resume_args", "expected"),
    [(["--resume"], "latest"), (["--resume", "saved-run"], "saved-run"), (["--resume=saved-run"], "saved-run")],
)
def test_eval_grouped_kwargs_preserve_optional_resume_value(resume_args: list[str], expected: str):
    with cli.eval_command.make_context(
        "eval",
        [
            "--base-url",
            "http://localhost/v1",
            "--model-name",
            "model",
            *resume_args,
            "--harness-kwargs",
            "epochs=3",
            "retry_on_error=2",
        ],
    ) as context:
        assert context.params["resume"] == expected
        assert context.params["harness_kwargs"] == ("epochs=3", "retry_on_error=2")


def test_verify_preserves_pytest_argument_forwarding():
    with cli.verify_command.make_context(
        "verify",
        ["--base-url", "http://localhost/v1", "--model-name", "model", "-k", "tool and not stream", "-x", "-vv"],
    ) as context:
        assert context.params["pytest_args"] == ("-k", "tool and not stream", "-x", "-vv")


@pytest.mark.parametrize(
    ("endpoint", "suite"),
    [
        ("--base-url", "e2e"),
        ("--parse-url", "parser"),
        ("--format-url", "formatter"),
        ("--completions-url", "raw_completions"),
    ],
)
@pytest.mark.parametrize("ancestor_config", [False, True])
def test_verify_uses_bundled_config(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, endpoint: str, suite: str, ancestor_config: bool
) -> None:
    installed = tmp_path / "package" / "functional_tests"
    (installed / suite).mkdir(parents=True)
    bundled = pathlib.Path(functional_tests.__file__).parent
    shutil.copyfile(bundled / "conftest.py", installed / "conftest.py")
    shutil.copyfile(bundled.parent / "pytest.ini", installed.parent / "pytest.ini")
    if ancestor_config:
        (tmp_path / "pytest.ini").write_text("[pytest]\naddopts = -k unrelated_customer_tests\n")
    (installed / suite / "test_options.py").write_text('def test_model(model):\n    assert model == "model"\n')
    monkeypatch.setattr(functional_tests, "__file__", str(installed / "__init__.py"))
    monkeypatch.setattr(run_store, "default_runs_dir", lambda: tmp_path / "runs")
    # No shared ancestor with the installation except the filesystem root.
    monkeypatch.chdir(tmp_path.anchor)

    result = click.testing.CliRunner().invoke(
        cli.cli, ["verify", endpoint, "https://example.test/v1", "--model-name", "model"]
    )

    assert result.exit_code == 0, result.output
    report = next((tmp_path / "runs").glob("*/verify_report.json"))
    counts, cases = cli.parse_pytest_report(report)
    assert counts == {"passed": 1}
    assert cases[0]["suite"] == suite


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        (["--model-kwargs"], "requires an argument"),
        (["--model-kwargs", "temperature=1", "--unknown-option"], "No such option"),
        (["--model-kwargs", "temperature=1", "tools=[unclosed"], "malformed --model-kwargs"),
    ],
)
def test_eval_grouped_kwargs_preserve_argument_errors(extra_args: list[str], expected: str):
    result = click.testing.CliRunner().invoke(
        cli.cli, ["eval", "gpqa-diamond", "--base-url", "http://unused/v1", "--model-name", "model", *extra_args]
    )
    assert result.exit_code != 0
    assert expected in result.output
    assert "Proceed?" not in result.output


def test_eval_rejects_malformed_model_options_before_starting(endpoint_checks: list[str]) -> None:
    result = click.testing.CliRunner().invoke(
        cli.cli,
        ["eval", "gpqa-diamond", "--base-url", "http://unused/v1", "--model-name", "model", "--model-kwargs", "tools"],
    )
    assert result.exit_code == 1
    assert "malformed --model-kwargs" in result.output
    assert endpoint_checks == []


class _RecordingRunner(base.DatasetRunner):
    MODULE = "offline"

    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id
        self.requests: list[base.RunRequest] = []

    def __call__(self, request: base.RunRequest) -> base.RunOutcome:
        self.requests.append(request)
        result = records.DatasetResult(dataset=self.dataset_id, topline_metric_name="accuracy", metrics={"accuracy": 1})
        return base.RunOutcome(result=result, traces=[])


@pytest.mark.parametrize("sandbox_args, sandbox", [([], "docker"), (["--sandbox", "daytona"], "daytona")])
def test_eval_forwards_reflection_key_without_saving_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, sandbox_args: list[str], sandbox: str
) -> None:
    recording = _RecordingRunner("credential-test")
    monkeypatch.setitem(registry.DATASETS, "credential-test", registry.Dataset(runner=recording))
    result = click.testing.CliRunner().invoke(
        cli.cli,
        [
            "eval",
            "credential-test",
            "--base-url",
            "https://example.test/v1",
            "--model-name",
            "model",
            "--output-dir",
            str(tmp_path),
            *sandbox_args,
        ],
        env={"REFLECTION_API_KEY": "synthetic-reflection-key", "OPENAI_API_KEY": "synthetic-ignored-key"},
    )
    assert result.exit_code == 0, result.output
    assert recording.requests[0].api_key == "synthetic-reflection-key"
    assert recording.requests[0].settings.sandbox == sandbox
    manifest = json.loads((tmp_path / "run.json").read_text())
    assert manifest["dataset_settings"]["credential-test"]["sandbox"] == sandbox
    assert "synthetic-reflection-key" not in result.output + (tmp_path / "run.json").read_text()


@pytest.mark.parametrize(
    ("asked", "expected"), [(["--max-tokens", "16384"], 16384), (["--max-tokens", "4096"], 4096), ([], 8192)]
)
def test_dataset_token_defaults_are_overridable(tmp_path, monkeypatch, asked, expected):
    """A larger served context can use an explicit budget above the registry default."""
    recording = _RecordingRunner("long")
    monkeypatch.setitem(
        registry.DATASETS,
        "long",
        registry.Dataset(runner=recording, settings=records.EvalSettings(max_tokens=8192)),
    )
    args = [
        "eval",
        "long",
        "--base-url",
        "http://unused/v1",
        "--model-name",
        "m",
        "--output-dir",
        str(tmp_path),
        *asked,
    ]
    result = click.testing.CliRunner().invoke(cli.cli, args)
    assert result.exit_code == 0, result.output
    assert recording.requests[0].settings.max_tokens == expected
    assert "capped at" not in result.output


@pytest.mark.parametrize(
    "options", [[], ["--temperature", "0", "--reasoning-effort", "low", "--harness-kwargs", "epochs=1"]]
)
def test_cli_resolves_each_dataset_defaults_and_explicit_overrides(tmp_path, monkeypatch, options):
    science = _RecordingRunner("science-defaults")
    long = _RecordingRunner("long-defaults")
    monkeypatch.setitem(
        registry.DATASETS, science.dataset_id, registry.Dataset(runner=science, config="configs/gpqa-diamond.yaml")
    )
    monkeypatch.setitem(
        registry.DATASETS,
        long.dataset_id,
        registry.Dataset(runner=long, settings=records.EvalSettings(max_tokens=8192)),
    )
    result = click.testing.CliRunner().invoke(
        cli.cli,
        [
            "eval",
            science.dataset_id,
            long.dataset_id,
            "--base-url",
            "http://unused/v1",
            "--model-name",
            "model",
            "--output-dir",
            str(tmp_path),
            *options,
        ],
    )
    assert result.exit_code == 0, result.output
    science_settings = science.requests[0].settings
    assert science_settings.temperature == (0 if options else 0.7)
    assert science_settings.reasoning_effort == ("low" if options else "max")
    assert science_settings.harness_kwargs["epochs"] == (1 if options else 16)
    assert science_settings.top_p == 0.99
    assert science_settings.max_tokens is None
    assert long.requests[0].settings.max_tokens == 8192
    manifest = json.loads((tmp_path / "run.json").read_text())
    assert manifest["dataset_settings"][science.dataset_id] == science_settings.model_dump()
    assert manifest["dataset_settings"][long.dataset_id] == long.requests[0].settings.model_dump()


def test_verify_model_kwargs_do_not_swallow_trailing_pytest_arguments():
    with cli.verify_command.make_context(
        "verify",
        [
            "--base-url=http://localhost/v1",
            "--model-name=model",
            "--model-kwargs",
            "no_stop_trim=true",
            "e2e/test_tool_protocol.py",
            "-k",
            "named",
        ],
        **cli.CTX,
    ) as context:
        assert context.params["model_kwargs"] == ("no_stop_trim=true",)
        assert context.params["pytest_args"] == ("e2e/test_tool_protocol.py", "-k", "named")


def test_verify_rejects_model_kwargs_without_a_value():
    with pytest.raises(click.BadParameter, match="--model-kwargs"):
        cli.verify_command.main(
            ["--base-url=http://localhost/v1", "--model-name=model", "--model-kwargs", "no_stop_trim"],
            standalone_mode=False,
        )
