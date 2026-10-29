"""Evaluation settings defaults."""

import dataclasses
from importlib import resources

from beam_verifier import options, records
from beam_verifier.benchmarks import prompts


def load(path: str) -> records.EvalSettings:
    """Loads a registry config resource, rejecting misspelled setting names.

    Args:
      path: Resource path relative to the benchmarks package.

    Returns:
      Dataset defaults; null generation values delegate to the harness.

    Raises:
      ValueError: If the resource is not a settings mapping or has unknown keys.
    """
    import yaml  # noqa: PLC0415

    data = yaml.safe_load(resources.files(__package__).joinpath(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Dataset config {path!r} must be a settings mapping")
    unknown = data.keys() - records.EvalSettings.model_fields.keys()
    if unknown:
        raise ValueError(f"Unknown settings in {path!r}: {sorted(unknown)}")
    if isinstance(factory := data.get("prompt_factory"), dict):
        unknown = factory.keys() - {field.name for field in dataclasses.fields(prompts.BeamPromptFactory)}
        if unknown:
            raise ValueError(f"Unknown prompt_factory settings in {path!r}: {sorted(unknown)}")
    return records.EvalSettings.model_validate(data)


def merge(defaults: records.EvalSettings, overrides: records.EvalSettings) -> records.EvalSettings:
    """Applies explicitly supplied fields, preserving zero, false and null.

    Native evaluation options merge by option name; each supplied option
    replaces its value. Model request mappings merge recursively.
    """
    values = defaults.model_dump()
    for name, value in overrides.model_dump(exclude_unset=True).items():
        if name == "model_kwargs":
            values[name] = options.merge_options(values[name], value)
        elif name == "harness_kwargs":
            values[name] = values[name] | value
        else:
            values[name] = value
    return records.EvalSettings.model_validate(values)
