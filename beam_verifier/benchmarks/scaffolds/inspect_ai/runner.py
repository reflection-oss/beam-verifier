"""The InspectAI runner: calls InspectAI's `eval()` in this process."""

import inspect
from collections.abc import Collection
from typing import Any

from beam_verifier import options, records
from beam_verifier.benchmarks.scaffolds import base
from beam_verifier.benchmarks.scaffolds.inspect_ai import log

# OpenAI request field -> the name InspectAI's `GenerateConfig` declares for it.
OAI_TO_IAI_CONFIG_NAMES = {"stop": "stop_seqs", "n": "num_choices", "max_completion_tokens": "max_tokens"}


class InspectAIRunner(base.DatasetRunner):
    """Runs a dataset through InspectAI's `eval()` API.

    Attributes:
      dataset_id: The registry id this runner executes; InspectAI's task name by default.
      task: InspectAI task path, e.g. `inspect_evals/aime2026`. Defaults to the task the
        dataset id names.
      task_args: Native task options forwarded to InspectAI unchanged.
      limit: Samples to run when the run does not set `num_examples`: a fixed
        subset, sized for the statistics rather than the whole dataset.
      shuffle_seed: Shuffle with this seed before `limit`, so the subset is a
        random sample that is the same on every run.
    """

    MODULE = "inspect_ai"

    def __init__(
        self,
        dataset_id: str,
        task: str | None = None,
        task_args: dict[str, object] | None = None,
        limit: int | None = None,
        shuffle_seed: int | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.task = task
        self.task_args = dict(task_args or {})
        self.limit = limit
        self.shuffle_seed = shuffle_seed

    def __call__(self, request: base.RunRequest) -> base.RunOutcome:
        """Runs the dataset through InspectAI's `eval()` and parses its log.

        InspectAI renders its own progress display, so long runs stay visible, and
        records a failed run in the log rather than raising; what it does raise is a
        failure to start the run at all.

        Args:
          request: Endpoint, run directory and settings. Explicit `harness_kwargs` override
            options passed to InspectAI's `eval()`; nested options replace the
            corresponding mapping. Unset concurrency uses InspectAI's default.

        Returns:
          The dataset's result and traces, with a warning when InspectAI reported a
          run-level failure that leaves the score partial.

        Raises:
          RunnerError: If InspectAI failed to run, or wrote a log that cannot be
            read.
          KeyboardInterrupt: If the run was interrupted.
        """
        import inspect_ai  # noqa: PLC0415
        from inspect_ai import log as inspect_log  # noqa: PLC0415

        from beam_verifier.benchmarks.scaffolds.inspect_ai import (
            prompts as prompt_adapter,  # noqa: PLC0415
        )

        try:
            kwargs = self.build_eval_kwargs(request)
            prior = inspect_log.list_eval_logs(kwargs["log_dir"]) if request.resume else []
            if prior:
                # InspectAI restores the task and generation settings from its log,
                # but omits credentials, so the provider's key is supplied again.
                accepted = frozenset(inspect.signature(inspect_ai.eval_retry).parameters) - {"tasks"}
                with base.scoped_env("BEAM_VERIFIER_API_KEY", request.api_key):
                    logs = inspect_ai.eval_retry(
                        prior[0],
                        **{name: value for name, value in kwargs.items() if name in accepted and value is not None},
                    )
            else:
                if kwargs["tasks"] is prompt_adapter.beam_prompt_task:
                    # Resolve only for a new run: retry uses the saved arguments,
                    # and the original config file may no longer exist.
                    task_args = kwargs["task_args"]
                    task_args["task_args"] = prompt_adapter.task_arguments(task_args["task_args"])
                logs = inspect_ai.eval(**kwargs)
            if not logs or logs[0].status == "cancelled":
                # InspectAI reports Ctrl+C in its log; it must still stop the whole run.
                raise KeyboardInterrupt
            eval_log = logs[0]

            # Samples load lazily, so parsing can also fail on disk reads.
            result, traces = log.parse_log(eval_log, self.dataset_id)
        except Exception as error:  # noqa: BLE001 - isolation point: one dataset's failure must not end the run
            raise base.RunnerError(f"{self.dataset_id}: InspectAI failed: {error}") from error

        warning = None
        if eval_log.status != "success":
            message = eval_log.error.message[:300] if eval_log.error else ""
            failure = f"{eval_log.status}: {message}" if message else eval_log.status
            warning = f"the eval did not complete — {failure}"
        return base.RunOutcome(result=result, traces=traces, warning=warning)

    def build_eval_kwargs(self, request: base.RunRequest) -> dict[str, Any]:
        """Maps run settings to InspectAI's native evaluation arguments.

        Args:
          request: Endpoint and settings for this dataset. The run's
            `model_kwargs` merge over these defaults, under the names
            `GenerateConfig` declares — InspectAI takes model request options
            as top-level generate-config arguments — and explicit
            `harness_kwargs` then override. Native `model_args` retain caller
            credentials and transport defaults for keys they omit; other
            nested mappings are replaced as a whole.

        Returns:
          Arguments for `inspect_ai.eval`; unset options keep library defaults.
        """
        settings = request.settings
        model_args = {
            "api_key": request.api_key,
            "stream": True,
            "client_timeout": settings.client_timeout,
        }
        kwargs: dict[str, Any] = {
            "tasks": self.resolved_task,
            "model": f"openai-api/beam-verifier/{request.model_name}",
            "model_base_url": request.base_url,
            "model_args": model_args,
            "task_args": self.task_args,
            "log_dir": str(request.run_dir / "inspect_logs" / self.dataset_id),
            "max_connections": settings.parallel_tasks,
            "limit": settings.num_examples if settings.num_examples is not None else self.limit,
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "reasoning_effort": settings.reasoning_effort,
        }
        if self.shuffle_seed is not None:
            kwargs["sample_shuffle"] = self.shuffle_seed
        model_kwargs: dict[str, Any] = {}
        for key, value in settings.model_kwargs.items():
            name = OAI_TO_IAI_CONFIG_NAMES.get(key, key)
            # OpenAI accepts a bare stop string where InspectAI wants a sequence.
            model_kwargs[name] = [value] if name == "stop_seqs" and isinstance(value, str) else value
        kwargs = options.merge_options(kwargs, model_kwargs)
        kwargs.update(settings.harness_kwargs)
        if settings.prompt_factory is not None and "tasks" not in settings.harness_kwargs:
            from beam_verifier.benchmarks.scaffolds.inspect_ai import (
                prompts as prompt_adapter,  # noqa: PLC0415
            )

            kwargs["tasks"] = prompt_adapter.beam_prompt_task
            kwargs["task_args"] = {
                "task_name": self.resolved_task,
                "task_args": kwargs["task_args"],
                "prompt_factory": settings.prompt_factory.name,
                "user_template": settings.prompt_factory.user_template,
                "system_text": settings.prompt_factory.system_text,
            }
        if isinstance(kwargs["model_args"], dict):
            kwargs["model_args"] = model_args | kwargs["model_args"]
        return kwargs

    def unsupported_resume_overrides(self, settings: records.EvalSettings, names: Collection[str]) -> list[str]:
        """Names the overrides `eval_retry()` cannot apply.

        Args:
          settings: Changed settings only; option mappings contain only changed keys.
          names: `EvalSettings` field names the caller overrode on the command line.

        Returns:
          Descriptions of the overrides that will not reach InspectAI.
        """
        # `benchmarks/registry.py` imports every scaffold module, so importing
        # InspectAI at module scope would make `list-datasets` pay its cost.
        import inspect_ai  # noqa: PLC0415

        accepted = frozenset(inspect.signature(inspect_ai.eval_retry).parameters)
        unsupported: list[str] = []
        for name in names:
            if name == "harness_kwargs":
                unsupported.extend(f"--harness-kwargs {key}" for key in settings.harness_kwargs if key not in accepted)
            elif {"parallel_tasks": "max_connections"}.get(name, name) not in accepted:
                unsupported.append(f"--{name.replace('_', '-')}")
        return unsupported

    @property
    def resolved_task(self) -> str:
        """The configured InspectAI task, or the one this dataset's id names.

        `inspect_evals` names its tasks after the datasets, with underscores where
        these ids use dashes.
        """
        return self.task or f"inspect_evals/{self.dataset_id.replace('-', '_')}"
