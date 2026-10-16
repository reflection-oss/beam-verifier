"""Every dataset beam-verifier can run, keyed by the id named on the command line."""

import dataclasses
from typing import Literal

from beam_verifier import records
from beam_verifier.benchmarks import config as dataset_config
from beam_verifier.benchmarks.scaffolds import base
from beam_verifier.benchmarks.scaffolds.inspect_ai import runner as inspect_ai


@dataclasses.dataclass
class Dataset:
    """One dataset: what it is, and the runner configured to execute it.

    Attributes:
      runner: The runner configured to execute this dataset.
      display_name: Human-readable name; reports fall back to the id when unset.
      development_status: How far this dataset's scores have been validated.
      description: One-line description, shown by `list-datasets`.
      scope_note: Set when coverage is narrower than the name suggests.
      config: Packaged YAML defaults, relative to the benchmarks package.
        Explicit run settings override them.
      settings: Inline defaults for datasets without a YAML config.
      run_by_default: Whether `eval` with no dataset named includes it. Off for a
        dataset that needs something only the caller can supply (an option, an
        extra, a long-context server).
    """

    runner: base.DatasetRunner
    display_name: str = ""
    development_status: Literal["verified", "unverified"] = "unverified"
    description: str = ""
    scope_note: str = ""
    config: str | None = None
    settings: records.EvalSettings | None = None
    run_by_default: bool = True

    @property
    def dataset_id(self) -> str:
        """Id named on the `eval` command line and used for trace file names."""
        return self.runner.dataset_id

    def resolve_settings(self, overrides: records.EvalSettings | None = None) -> records.EvalSettings:
        """Loads this dataset's defaults and applies explicitly supplied options.

        Args:
          overrides: Caller settings; omitted fields keep their dataset defaults.

        Returns:
          Resolved settings without importing the evaluation harness.
        """
        if self.config:
            defaults = dataset_config.load(self.config)
        else:
            defaults = self.settings.model_copy(deep=True) if self.settings else records.EvalSettings()
        return dataset_config.merge(defaults, overrides) if overrides is not None else defaults


DATASETS: dict[str, Dataset] = {
    dataset.dataset_id: dataset
    for dataset in (
        Dataset(
            runner=inspect_ai.InspectAIRunner(dataset_id="gpqa-diamond"),
            config="configs/gpqa-diamond.yaml",
            display_name="GPQA Diamond",
            development_status="verified",
            description="Graduate-level science questions from the expert-validated 'Diamond' subset",
        ),
        Dataset(
            runner=inspect_ai.InspectAIRunner(dataset_id="aime2026"),
            config="configs/aime2026.yaml",
            display_name="AIME 2026",
            development_status="verified",
            description="American Invitational Mathematics Examination, 2026",
        ),
        Dataset(
            runner=inspect_ai.InspectAIRunner(dataset_id="gsm8k"),
            display_name="GSM8K",
            description="Grade-school math word problems, 10-shot",
            scope_note="All 1319 test problems",
            run_by_default=False,
        ),
        Dataset(
            runner=inspect_ai.InspectAIRunner(
                dataset_id="infinite-bench-kv-retrieval",
                task="inspect_evals/infinite_bench_kv_retrieval",
            ),
            settings=records.EvalSettings(
                num_examples=310,
                harness_kwargs={"sample_shuffle": 0},
            ),
            display_name="InfiniteBench KV retrieval",
            description="Look up a key's value in a ~128k-token JSON of UUID pairs",
            scope_note="Fixed random 310 of 500; needs a long-context server",
            run_by_default=False,
        ),
        Dataset(
            runner=inspect_ai.InspectAIRunner(
                dataset_id="infinite-bench-math-find",
                task="inspect_evals/infinite_bench_math_find",
            ),
            settings=records.EvalSettings(max_tokens=8192),
            display_name="InfiniteBench Math.Find",
            description="Find the max, min or median of a ~128k-token list of numbers",
            scope_note="All 350; needs a long-context server",
            run_by_default=False,
        ),
        Dataset(
            runner=inspect_ai.InspectAIRunner(
                dataset_id="infinite-bench-longbook-choice",
                task="inspect_evals/infinite_bench_longbook_choice_eng",
            ),
            settings=records.EvalSettings(max_tokens=8192),
            display_name="InfiniteBench En.MC",
            description="Multiple-choice questions over a novel truncated to ~128k tokens",
            scope_note="All 229; needs a long-context server",
            run_by_default=False,
        ),
    )
}
