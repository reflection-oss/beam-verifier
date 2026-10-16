"""Offline tests for the CLI's option surface: no network, no subprocess."""

import importlib.metadata
import json
import pathlib

import click.testing
import pytest

from beam_verifier import cli, options, records
from beam_verifier.benchmarks import registry
from beam_verifier.benchmarks.scaffolds import base


@pytest.fixture(autouse=True)
def available_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CLI state and argument tests independent of a live endpoint."""
    monkeypatch.setattr(cli, "check_endpoint", lambda *args: True)


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


@pytest.mark.parametrize(("command_name", "extra_args"), [("eval", []), ("verify", [])])
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
    extra_args: list[str],
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
    args = ["--base-url", "https://example.test/v1", "--model-name", "model", *extra_args, *key_args]
    with click.testing.CliRunner().isolation(env=env), command.make_context(command_name, args) as context:
        assert context.params["api_key"] == expected


def test_eval_base_url_and_model_name_are_required():
    """An eval always requires the endpoint and model it will exercise."""
    command = cli.cli.commands["eval"]
    assert _option(command, "base_url").required
    assert _option(command, "model_name").required


def test_eval_rejects_a_run_without_a_base_url():
    result = click.testing.CliRunner().invoke(cli.cli, ["eval", "--model-name", "m"])
    assert result.exit_code == 2
    assert "--base-url" in result.output


def test_eval_rejects_numeric_reasoning_effort():
    result = click.testing.CliRunner().invoke(cli.cli, ["eval", "--reasoning-effort", "512"])
    assert result.exit_code == 2
    assert "'512' is not one of 'low', 'medium', 'high', 'xhigh', 'max'" in result.output


def test_eval_rejects_an_unknown_dataset():
    """Exit code 1 rather than click's own 2 is the assertion that matters: the command
    parsed its arguments and reached its own validation, which runs before the endpoint
    is probed. The output is rich markup, so only the offending id is matched."""
    result = click.testing.CliRunner().invoke(
        cli.cli, ["eval", "--base-url", "https://endpoint.example.com/v1", "--model-name", "m", "nope"]
    )
    assert result.exit_code == 1
    assert "nope" in result.output


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


@pytest.mark.parametrize(
    ("num_examples", "parallel_tasks"),
    [
        pytest.param(0, None, id="zero-examples"),
        pytest.param(-1, None, id="negative-examples"),
        pytest.param(None, 0, id="zero-tasks"),
        pytest.param(None, -1, id="negative-tasks"),
    ],
)
def test_validate_input_args_rejects_a_count_below_one(num_examples: int | None, parallel_tasks: int | None):
    """Zero is not a smaller run but no run at all, and both counts are read as
    `value or <fallback>` downstream — so zero would silently mean the whole dataset
    or the default task count instead of failing."""
    with pytest.raises(ValueError, match="at least 1"):
        cli.validate_input_args(["aime2026"], num_examples, parallel_tasks)


def test_validate_input_args_accepts_unset_counts():
    """Unset is how a run asks for the dataset's own size and the default task count,
    which is the value those fallbacks exist for."""
    cli.validate_input_args(["aime2026"], None, None)


def test_validate_input_args_rejects_an_unregistered_dataset():
    """The id is what the run looks up, so an unknown one is caught before the run
    starts rather than raising a KeyError partway through."""
    with pytest.raises(ValueError, match="nope"):
        cli.validate_input_args(["aime2026", "nope"], None, None)


@pytest.mark.parametrize("flag", ["--num-examples", "--parallel-tasks"])
def test_eval_rejects_a_zero_count_before_reaching_the_endpoint(flag: str):
    """A value the arguments alone can reject must not first probe the endpoint or
    leave an empty run directory behind, so the base URL here is never contacted."""
    result = click.testing.CliRunner().invoke(
        cli.cli,
        ["eval", "--base-url", "https://endpoint.example.com/v1", "--model-name", "m", flag, "0"],
    )
    assert result.exit_code == 1
    assert flag in result.output


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


def test_eval_accepts_repeated_model_kwargs():
    command = cli.cli.commands["eval"]
    with command.make_context(
        "eval",
        [
            "--base-url",
            "http://localhost/v1",
            "--model-name",
            "model",
            "--model-kwargs",
            "stop=[DONE]",
            "--model-kwargs",
            "extra_body={custom_option: false}",
        ],
    ) as context:
        assert options.parse_kwargs(context.params["model_kwargs"], option_name="--model-kwargs") == {
            "stop": ["DONE"],
            "extra_body": {"custom_option": False},
        }


@pytest.mark.parametrize("scope", ["model", "harness"])
def test_eval_accepts_grouped_and_repeated_kwargs(scope: str):
    option = f"--{scope}-kwargs"
    with cli.eval_command.make_context(
        "eval",
        [
            "gpqa-diamond",
            "--base-url",
            "http://localhost/v1",
            "--model-name",
            "model",
            option,
            "tags=[read, write]",
            "enabled=false",
            "offset=-2",
            option,
            "enabled=true",
            "prompt=Run tests = always.",
        ],
    ) as context:
        assert options.parse_kwargs(context.params[f"{scope}_kwargs"], option_name=option) == {
            "tags": ["read", "write"],
            "enabled": True,
            "offset": -2,
            "prompt": "Run tests = always.",
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


def test_eval_accepts_harness_kwargs():
    command = cli.cli.commands["eval"]
    with command.make_context(
        "eval", ["--base-url", "http://unused/v1", "--model-name", "model", "--harness-kwargs", "retry_on_error=2"]
    ) as context:
        assert options.parse_kwargs(context.params["harness_kwargs"]) == {"retry_on_error": 2}


def test_eval_rejects_malformed_model_options_before_starting():
    result = click.testing.CliRunner().invoke(
        cli.cli,
        ["eval", "gpqa-diamond", "--base-url", "http://unused/v1", "--model-name", "model", "--model-kwargs", "tools"],
    )
    assert result.exit_code == 1
    assert "malformed --model-kwargs" in result.output
    assert "Proceed?" not in result.output


class _RecordingRunner(base.DatasetRunner):
    MODULE = "offline"

    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id
        self.requests: list[base.RunRequest] = []

    def __call__(self, request: base.RunRequest) -> base.RunOutcome:
        self.requests.append(request)
        result = records.DatasetResult(dataset=self.dataset_id, topline_metric_name="accuracy", metrics={"accuracy": 1})
        return base.RunOutcome(result=result, traces=[])


def test_eval_forwards_reflection_key_without_saving_it(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
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
        ],
        env={"REFLECTION_API_KEY": "synthetic-reflection-key", "OPENAI_API_KEY": "synthetic-ignored-key"},
    )
    assert result.exit_code == 0, result.output
    assert recording.requests[0].api_key == "synthetic-reflection-key"
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


@pytest.mark.parametrize("command_name", ["eval", "verify"])
def test_every_endpoint_command_accepts_grouped_model_kwargs(command_name: str):
    """One spelling of `--model-kwargs` reaches every command that issues model requests."""
    command = cli.cli.commands[command_name]
    required = {
        "eval": ["--base-url=http://localhost/v1", "--model-name=model"],
        "verify": ["--base-url=http://localhost/v1", "--model-name=model"],
    }[command_name]
    with command.make_context(
        command_name,
        [*required, "--model-kwargs", "no_stop_trim=true", "include_stop_str_in_output=true"],
        **cli.CTX,
    ) as context:
        assert options.parse_kwargs(context.params["model_kwargs"], option_name="--model-kwargs") == {
            "no_stop_trim": True,
            "include_stop_str_in_output": True,
        }


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


def test_every_kwargs_channel_shares_one_documented_contract():
    """A channel's help names what it forwards to; `kwargs_option` appends the rest."""
    channels = ["harness", "model"]
    helps = {}
    for channel in channels:
        parameter = next(
            param
            for param in cli.eval_command.params
            if isinstance(param, click.Option) and param.name == f"{channel}_kwargs"
        )
        assert parameter.multiple, f"--{channel}-kwargs must be repeatable"
        assert parameter.metavar == "KEY=VALUE ...", f"--{channel}-kwargs has metavar {parameter.metavar!r}"
        assert parameter.help is not None
        helps[channel] = parameter.help
    contract = helps["model"].split(". ")[-1]
    assert all(text.endswith(contract) for text in helps.values()), (
        f"Channel help texts have drifted apart; none of them may restate {contract!r}: {helps}"
    )
    assert all(text != contract for text in helps.values()), "Each channel must also say what it forwards to"
