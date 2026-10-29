import pathlib

import pytest

from beam_verifier import records
from beam_verifier.benchmarks import config, prompts, registry
from beam_verifier.benchmarks.scaffolds import base
from beam_verifier.benchmarks.scaffolds.harbor import runner as harbor_runner
from beam_verifier.benchmarks.scaffolds.inspect_ai import runner as inspect_ai_runner


def test_explicit_default_values_override_dataset_settings():
    defaults = records.EvalSettings(temperature=1, client_timeout=100, max_tokens=8192)
    explicit = records.EvalSettings(temperature=0, client_timeout=600, max_tokens=None)
    assert config.merge(defaults, explicit) == explicit


@pytest.mark.parametrize(
    ("dataset_id", "harness_kwargs", "epochs"),
    [
        ("aime2026", {}, 64),
        ("gpqa-diamond", {}, 16),
        ("aime2026", {"epochs": 3}, 3),
        ("gpqa-diamond", {"epochs": 3}, 3),
    ],
)
def test_reference_defaults_and_overrides_reach_inspect_ai(
    tmp_path: pathlib.Path, dataset_id: str, harness_kwargs: dict[str, object], epochs: int
):
    dataset = registry.DATASETS[dataset_id]
    assert isinstance(dataset.runner, inspect_ai_runner.InspectAIRunner)
    kwargs = dataset.runner.build_eval_kwargs(
        base.RunRequest(
            base_url="http://unused/v1",
            model_name="beam",
            run_dir=tmp_path,
            settings=dataset.resolve_settings(records.EvalSettings(harness_kwargs=harness_kwargs)),
        )
    )
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.99
    assert kwargs["reasoning_effort"] == "max"
    assert kwargs["epochs"] == epochs
    assert kwargs["max_tokens"] is None
    assert "extra_body" not in kwargs


def test_native_override_channels_preserve_unrelated_options():
    defaults = records.EvalSettings(
        harness_kwargs={"epochs": 8, "model_args": {"stream": True, "client_timeout": 600}},
        model_kwargs={"extra_body": {"custom_option": True, "chat_template_kwargs": {"hint": "beam"}}},
    )
    explicit = records.EvalSettings(
        harness_kwargs={"model_args": {"stream": False}},
        model_kwargs={"extra_body": {"chat_template_kwargs": {"hint": None}}},
    )
    merged = config.merge(defaults, explicit)
    assert merged.harness_kwargs == {"epochs": 8, "model_args": {"stream": False}}
    assert merged.model_kwargs == {"extra_body": {"custom_option": True, "chat_template_kwargs": {"hint": None}}}
    assert defaults.harness_kwargs["model_args"] == {"stream": True, "client_timeout": 600}


def test_yaml_prompt_factory_survives_settings_serialization():
    settings = registry.DATASETS["aime2026"].resolve_settings()
    restored = records.EvalSettings.model_validate_json(settings.model_dump_json())
    assert isinstance(restored.prompt_factory, prompts.BeamPromptFactory)
    assert restored.prompt_factory.name == "beam-aime-v1"
    assert restored.prompt_factory.render_task_instruction("Compute {6 * 7}.") == (
        "Solve the following problem and output the final answer within \\boxed{}.\nCompute {6 * 7}."
    )


@pytest.mark.parametrize("factory", [None, prompts.BeamPromptFactory("custom-v1", user_template="Solve {prompt}")])
def test_prompt_factory_can_be_disabled_or_replaced(factory):
    defaults = registry.DATASETS["aime2026"].resolve_settings()
    merged = config.merge(defaults, records.EvalSettings(prompt_factory=factory))
    assert merged.prompt_factory == factory
    assert merged.temperature == 0.7
    assert defaults.prompt_factory is not None
    assert defaults.prompt_factory.name == "beam-aime-v1"


def test_swe_defaults_reach_harbor(tmp_path: pathlib.Path) -> None:
    dataset = registry.DATASETS["swebench-pro"]
    assert isinstance(dataset.runner, harbor_runner.HarborRunner)
    settings = dataset.resolve_settings()
    job = dataset.runner.build_job_config(base.RunRequest("http://unused/v1", "beam", tmp_path, settings=settings))
    assert job.n_attempts == 16
    assert job.environment.type == "docker"
    assert job.environment.force_build is True
    assert job.agents[0].kwargs["mirror_kwargs"] == {
        "api": "openai-chat-completions-non-streaming",
        "turn_timeout_seconds": 10800,
        "temperature": 0.7,
        "top_p": 0.99,
        "reasoning_effort": "max",
        "request_timeout_seconds": 600,
        "tools": "bash",
        "use_native_bash": True,
        "disable_compaction": True,
        "max_iterations": 1000000,
    }
    assert job.datasets[0].name == "swebenchpro"
    assert job.datasets[0].version == "1.0"
    assert dataset.run_by_default
    assert settings.prompt_factory is not None


def test_swe_prompt_formats_workdir_without_interpreting_task_text() -> None:
    factory = registry.DATASETS["swebench-pro"].resolve_settings().prompt_factory
    assert factory is not None
    assert factory.render_task_instruction("Fix {variable}; $(touch /tmp/unwanted)", workdir="/app") == (
        'Fix the following issue. The repository is checked out at `/app`. When you are finished, state "Task complete".\n\n'
        "Fix {variable}; $(touch /tmp/unwanted)"
    )
    with pytest.raises(ValueError, match="requires a sandbox working directory"):
        factory.render_task_instruction("Issue")


def test_prompt_factory_preserves_escaped_workdir_placeholder() -> None:
    factory = prompts.BeamPromptFactory("literal", "Use {{workdir}}: {prompt}")
    assert not factory.requires_workdir
    assert factory.render_task_instruction("task") == "Use {workdir}: task"


def test_prompt_factory_requires_workdir_for_converted_placeholder() -> None:
    factory = prompts.BeamPromptFactory("converted", "Use {workdir!r}: {prompt}")
    with pytest.raises(ValueError, match="requires a sandbox working directory"):
        factory.render_task_instruction("task")
    assert factory.render_task_instruction("task", workdir="/app") == "Use '/app': task"
