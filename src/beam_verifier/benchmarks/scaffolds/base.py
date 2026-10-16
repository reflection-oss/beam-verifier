"""The `DatasetRunner` base class: one request in, one outcome out."""

import contextlib
import dataclasses
import os
import pathlib
from collections.abc import Collection, Iterator

from beam_verifier import records


class RunnerError(Exception):
    """A dataset could not be run, or its results could not be read."""


@dataclasses.dataclass
class RunRequest:
    """Run request sent to a runner.

    Attributes:
      base_url: OpenAI-compatible base URL ending in `/v1`.
      model_name: Model id as served by the endpoint.
      run_dir: Run directory; runner artifacts are kept beneath it.
      api_key: Key for the endpoint. OpenAI clients require a key even for
        unauthenticated servers, hence the placeholder default.
      settings: Settings applied to every dataset in the run; a runner uses those
        that apply to it and ignores the rest.
      resume: Reuse completed samples from this run's harness artifacts.
    """

    base_url: str
    model_name: str
    run_dir: pathlib.Path
    api_key: str = "EMPTY"
    settings: records.EvalSettings = dataclasses.field(default_factory=records.EvalSettings)
    resume: bool = False


@dataclasses.dataclass
class RunOutcome:
    """The scored result of one dataset and the traces behind it.

    Attributes:
      result: Aggregate result, as saved in the run manifest.
      traces: Per-sample traces, in dataset order.
      warning: A problem worth reporting even though the run produced a result,
        such as a failure leaving the score partial.
    """

    result: records.DatasetResult
    traces: list[records.Trace]
    warning: str | None = None


class DatasetRunner:
    """Executes one dataset and reports its result.

    A runner is constructed with its dataset's identity and configuration, and its
    module must stay cheap to import: load heavy dependencies inside `__call__`
    rather than at module scope.

    Every runner that sends tunable model requests forwards the run's
    `model_kwargs` to those request options through
    `options.merge_options`.

    Attributes:
      MODULE: The scaffold's own module, shown in the `list-datasets` runner column.
        A scaffold is named after the eval library it drives, so for those it is also
        that library's importable name.
      dataset_id: Id of the dataset this runner runs, as named on the `eval`
        command line.
    """

    MODULE: str

    dataset_id: str

    def unsupported_resume_overrides(self, settings: records.EvalSettings, names: Collection[str]) -> list[str]:
        """Setting overrides this runner cannot apply when resuming.

        Args:
          settings: Effective settings for this dataset, overrides included.
          names: `EvalSettings` field names the caller overrode on the command line.

        Returns:
          Descriptions of the overrides this runner will not apply. The default is
          every name, for a harness whose resume restores its saved configuration.
        """
        del settings  # Unused.
        return list(names)

    def __call__(self, request: RunRequest) -> RunOutcome:
        """Runs this runner's dataset.

        Args:
          request: Endpoint, run directory and settings.

        Returns:
          The dataset's result and traces.

        Raises:
          RunnerError: If the dataset cannot be run or its results cannot be read.
        """
        raise NotImplementedError("Subclasses must implement this method")


@contextlib.contextmanager
def scoped_env(key: str, value: str) -> Iterator[None]:
    """Sets one environment variable for the block, restoring it on exit."""
    saved = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = saved
