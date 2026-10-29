"""Shared CLI options, and the decoding and merging behind them.

Every command declares its endpoint and `KEY=VALUE` options here, so a channel
is spelled once however many commands offer it. `--model-kwargs` is checked once here, at the
boundary: an OpenAI request field is named directly and every other engine
option goes in `extra_body`, so each layer below can merge and send.
"""

import inspect
from typing import Any

import click


def endpoint_arguments(func):
    """Shared `--base-url` / `--model-name` / `--api-key` options."""
    func = click.option(
        "--base-url",
        "base_url",
        required=True,
        metavar="URL",
        help="OpenAI-compatible base URL, ending in /v1.",
    )(func)
    func = click.option(
        "--model-name",
        "model_name",
        required=True,
        metavar="NAME",
        help="Model id, exactly as the endpoint names it.",
    )(func)
    func = click.option(
        "--api-key",
        "api_key",
        envvar="REFLECTION_API_KEY",
        default="EMPTY",
        help="API key; defaults to $REFLECTION_API_KEY, else 'EMPTY'.",
    )(func)
    return func


def kwargs_option(channel: str, description: str):
    """Declares `--<channel>-kwargs`, which forwards pairs to a lower layer.

    `description` says what the channel forwards to; the shared
    repeat-and-decode sentence is appended. The `_kwargs` suffix is what
    `_MultiValueCommand` reads to end a grouped value list.
    """

    def decorator(func):
        return click.option(
            f"--{channel}-kwargs",
            f"{channel}_kwargs",
            multiple=True,
            metavar="KEY=VALUE ...",
            help=f"{description} Accepts multiple KEY=VALUE pairs; repeatable, and values are YAML-decoded.",
        )(func)

    return decorator


# Shared by every command that issues model requests.
model_kwargs_option = kwargs_option(
    "model",
    "Request field sent with each model request, named as the OpenAI schema names it "
    "(e.g. temperature=0.6). An engine option outside that schema goes in extra_body, which every "
    "engine forwards into the request body: extra_body={no_stop_trim: true} for SGLang, "
    "extra_body={include_stop_str_in_output: true} for vLLM. Merged over the request options the "
    "command already builds: nested mappings merge per key and any other value replaces.",
)


def parse_kwargs(kwargs: tuple[str, ...], *, option_name: str = "--harness-kwargs") -> dict[str, object]:
    """Decodes `KEY=VALUE` pairs, dashes to underscores and values as YAML.

    `option_name` names the flag in error messages. Raises ValueError on a pair
    with no `=` or a value that is not valid YAML.
    """
    import yaml  # noqa: PLC0415

    parsed: dict[str, object] = {}
    for pair in kwargs:
        key, separator, raw = pair.partition("=")
        if not separator:
            raise ValueError(f"malformed {option_name} {pair!r}; expected KEY=VALUE.")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as error:
            raise ValueError(f"malformed {option_name} {pair!r}: {error}") from error
        parsed[key.replace("-", "_")] = value
    return parsed


def merge_options(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merges `overrides` over `defaults`: nested mappings per key, else replace.

    Typed for unpacking into the request APIs below it.
    """
    merged = dict(defaults)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = merge_options(current, value)
        else:
            merged[key] = value
    return merged


def parse_model_kwargs(kwargs: tuple[str, ...]) -> dict[str, object]:
    """Decodes `--model-kwargs` and checks each option is named the one way that works.

    An engine option the OpenAI schema does not define belongs in `extra_body`,
    which every engine forwards into the request body; checking here means one
    message before any work starts. An OpenAI field is accepted either way: a
    library may not implement one (InspectAI's `GenerateConfig` lacks `stop`)
    and then asks for it in `extra_body` itself, so refusing that would leave
    the caller with no spelling that works.

    Raises:
        ValueError: If a pair is malformed, or an option is named the wrong way.
    """
    options = parse_kwargs(kwargs, option_name="--model-kwargs")
    extra_body = options.get("extra_body", {})
    if not isinstance(extra_body, dict):
        raise ValueError(f"extra_body must be a mapping, got {extra_body!r}")
    import openai  # noqa: PLC0415

    # The SDK declares the OpenAI request fields; whatever it does not name is
    # an engine option. Imported here so `list-datasets` does not pay for it.
    sdk_fields = frozenset(inspect.signature(openai.resources.chat.completions.Completions.create).parameters)
    engine_options = sorted(key for key in options if key != "extra_body" and key not in sdk_fields)
    if engine_options:
        inner = ", ".join(f"{key}: ..." for key in engine_options)
        raise ValueError(
            f"{', '.join(engine_options)}: not an OpenAI request field. Engine options go in extra_body: "
            f"--model-kwargs 'extra_body={{{inner}}}'"
        )
    return options
