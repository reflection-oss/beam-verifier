"""Run coding datasets through Harbor."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

from beam_verifier import options
from beam_verifier.benchmarks.scaffolds import base
from beam_verifier.benchmarks.scaffolds.harbor.mirror import config as mirror_config

if TYPE_CHECKING:
    from harbor.models.job import config as job_config


class HarborRunner(base.DatasetRunner):
    """Run one dataset with Harbor and Mirror agents inside task sandboxes.

    Args:
        dataset_id: Dataset name used in the registry and output directory.
        dataset: Harbor dataset settings, including the source version.
        mirror_kwargs: Mirror options supplied by the dataset.
    """

    MODULE = "harbor"

    def __init__(
        self, dataset_id: str, dataset: dict[str, object], mirror_kwargs: dict[str, object] | None = None
    ) -> None:
        self.dataset_id = dataset_id
        self.dataset = dict(dataset)
        self.mirror_kwargs = dict(mirror_kwargs or {})

    def build_job_config(self, request: base.RunRequest) -> job_config.JobConfig:
        """Build a Harbor job configuration for this dataset and the Mirror agent.

        Caller-supplied Harbor options override generated settings. Agent
        options merge into the existing Mirror configuration.

        Args:
            request: Model endpoint and resolved dataset settings.

        Returns:
            Validated Harbor job settings.

        Raises:
            ValueError: If agent, sandbox, or authentication settings are unsupported.
        """
        from harbor.models import environment_type  # noqa: PLC0415
        from harbor.models.job import config as job_config  # noqa: PLC0415

        dataset = dict(self.dataset)
        if request.settings.num_examples is not None:
            dataset["n_tasks"] = request.settings.num_examples
        agent = mirror_config.build_agent_config(request, self.mirror_kwargs).model_dump()
        overrides = dict(request.settings.harness_kwargs)
        if "agents" in overrides:
            supplied = overrides.pop("agents")
            if not isinstance(supplied, list) or len(supplied) != 1 or not isinstance(supplied[0], dict):
                raise ValueError("Harbor datasets require exactly one Mirror agent")
            agent = options.merge_options(agent, supplied[0])
        defaults: dict[str, object] = {
            "job_name": self.dataset_id,
            "jobs_dir": request.run_dir / "harbor_jobs",
            "environment": {"type": request.settings.sandbox},
            "datasets": [dataset],
            "agents": [agent],
        }
        if request.settings.parallel_tasks is not None:
            defaults["n_concurrent_trials"] = request.settings.parallel_tasks
        config = job_config.JobConfig.model_validate(options.merge_options(defaults, overrides))
        if request.settings.host_model_proxy and config.environment.type != environment_type.EnvironmentType.DAYTONA:
            raise ValueError("--host-model-proxy requires --sandbox daytona; local Docker connects directly")
        if (
            config.environment.type == environment_type.EnvironmentType.DOCKER
            and "force_build" not in config.environment.model_fields_set
        ):
            # Task Dockerfiles can reset a prebuilt image's entrypoint, as in SWE-bench Pro.
            config.environment.force_build = True
        if request.api_key not in ("", "EMPTY") and not (
            request.settings.host_model_proxy and config.environment.type == environment_type.EnvironmentType.DAYTONA
        ):
            raise ValueError(
                "Authenticated Harbor endpoints require a Daytona host proxy; host API keys cannot enter sandboxes"
            )
        self.validate_job_config(config)
        return config

    def __call__(self, request: base.RunRequest) -> base.RunOutcome:
        """Run Harbor in a separate process and read its results.

        Args:
            request: Model endpoint and resolved settings for this dataset.

        Returns:
            Harbor's metrics and a trace for each trial.

        Raises:
            RunnerError: If setup, execution, or reading results fails.
        """
        from harbor.environments import factory  # noqa: PLC0415
        from harbor.models import environment_type  # noqa: PLC0415
        from harbor.models.job import config as job_config  # noqa: PLC0415

        from beam_verifier.benchmarks.proxies import endpoint_proxy  # noqa: PLC0415
        from beam_verifier.benchmarks.scaffolds.harbor import log, worker  # noqa: PLC0415

        try:
            config = self.build_job_config(request)
            use_proxy = (
                request.settings.host_model_proxy
                and config.environment.type == environment_type.EnvironmentType.DAYTONA
            )
            job_dir = config.jobs_dir / config.job_name
            resuming = request.resume and job_dir.exists()
            if job_dir.exists() and not request.resume:
                raise FileExistsError(f"Refusing to reuse Harbor job directory: {job_dir}")
            if resuming:
                saved = job_config.JobConfig.model_validate_json((job_dir / "config.json").read_text())
                if (saved.jobs_dir / saved.job_name).resolve() != job_dir.resolve():
                    raise ValueError("Harbor resume requires the original job directory")
                self.validate_job_config(saved)
                saved_proxy = (
                    saved.environment.type == environment_type.EnvironmentType.DAYTONA
                    and saved.agents[0].kwargs.get("base_url") == endpoint_proxy.SANDBOX_MODEL_URL
                )
                if saved_proxy != use_proxy:
                    raise ValueError("Harbor resume cannot change host model routing")
                if saved_proxy:
                    config.agents[0].kwargs["base_url"] = endpoint_proxy.SANDBOX_MODEL_URL
                if config != saved:
                    raise ValueError("Harbor resume requires unchanged job settings")
            try:
                factory.EnvironmentFactory.run_preflight(config.environment.type)
            except ImportError as error:
                if config.environment.type != environment_type.EnvironmentType.DAYTONA:
                    raise
                raise RuntimeError("Daytona requires: uv tool install 'beam-verifier[daytona]'") from error
            except SystemExit as error:
                raise RuntimeError(f"{config.environment.type} preflight failed: {error}") from error
            config.jobs_dir.mkdir(parents=True, exist_ok=True)
            with contextlib.ExitStack() as cleanup:
                proxy_port = None
                if use_proxy:
                    proxy = cleanup.enter_context(
                        endpoint_proxy.EndpointProxy(
                            request.base_url,
                            model_name=config.agents[0].model_name or request.model_name,
                            api_key=request.api_key,
                            port=0,
                            max_connections=config.n_concurrent_trials,
                        )
                    )
                    if failure := proxy.check_upstream():
                        raise RuntimeError(f"Model endpoint unreachable from host: {failure}")
                    config.agents[0].kwargs["base_url"] = endpoint_proxy.SANDBOX_MODEL_URL
                    proxy_port = proxy.port
                if resuming:
                    log.archive_ungraded_trials(job_dir)
                asyncio.run(worker.HarborWorker(config, proxy_port=proxy_port).run_in_subprocess())
            return log.parse_job(job_dir, self.dataset_id)
        except Exception as error:  # noqa: BLE001 - isolate one dataset's failure
            raise base.RunnerError(f"{self.dataset_id}: Harbor failed: {error}") from error

    @staticmethod
    def validate_job_config(config: job_config.JobConfig) -> None:
        """Check that a Harbor job uses supported sandbox and agent settings.

        Args:
            config: Harbor job configuration to check.

        Raises:
            ValueError: If sandbox, agent, or agent environment settings are unsupported.
        """
        from harbor.models import environment_type  # noqa: PLC0415

        if (
            config.environment.type
            not in {environment_type.EnvironmentType.DAYTONA, environment_type.EnvironmentType.DOCKER}
            or config.environment.import_path is not None
        ):
            raise ValueError("Harbor datasets support Daytona or local Docker")
        if config.user_agent is not None:
            raise ValueError("Harbor datasets do not support a separate user agent")
        if len(config.agents) != 1:
            raise ValueError("Harbor datasets require exactly one Mirror agent")
        configured_agent = config.agents[0]
        if configured_agent.name != mirror_config.IMPORT_PATH or configured_agent.import_path is not None:
            raise ValueError("Harbor datasets require the standalone Mirror agent")
        if configured_agent.env:
            raise ValueError(
                "Mirror agent environment overrides are unsupported; host credentials must stay on the host"
            )
