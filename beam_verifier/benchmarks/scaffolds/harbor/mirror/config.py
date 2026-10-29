"""Configure Mirror for a Harbor trial."""

from __future__ import annotations

import dataclasses
import math
import re
from typing import TYPE_CHECKING

from beam_verifier import options
from beam_verifier.benchmarks.scaffolds import base

if TYPE_CHECKING:
    from harbor.models.trial import config as trial_config

IMPORT_PATH = f"{__package__}.agent:MirrorAgent"


@dataclasses.dataclass
class MirrorConfig:
    """Mirror settings used by the Harbor adapter.

    One Mirror turn handles the full task instruction, including all model
    calls and tool use. Its default limit is three hours; Harbor may stop it
    sooner. Other Mirror options are passed through to the binary.
    """

    model: str
    base_url: str
    mirror_kwargs: dict[str, object] = dataclasses.field(default_factory=dict)
    turn_timeout_seconds: float = 10800
    workspace: str = "."

    def __post_init__(self) -> None:
        self.mirror_kwargs = {name.replace("-", "_"): value for name, value in self.mirror_kwargs.items()}
        for name in self.mirror_kwargs:
            if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                raise ValueError(f"invalid Mirror option name: {name!r}")
            if name in {"model", "base_url", "bind", "output"}:
                raise ValueError(f"Mirror option {name!r} is managed by beam-verifier")
        workspace = self.mirror_kwargs.get("workspace", self.workspace)
        timeout = self.mirror_kwargs.get("turn_timeout_seconds", self.turn_timeout_seconds)
        if not isinstance(workspace, str):
            raise ValueError("Mirror workspace must be a string")
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("Mirror turn_timeout_seconds must be finite and positive")
        self.workspace = workspace
        self.turn_timeout_seconds = timeout

    def settings(self) -> dict[str, object]:
        """Merge Mirror options with the default API mode and turn timeout."""
        return options.merge_options(
            {"api": "openai-chat-completions-non-streaming", "turn_timeout_seconds": self.turn_timeout_seconds},
            self.mirror_kwargs,
        )


def build_agent_config(request: base.RunRequest, mirror_kwargs: dict[str, object]) -> trial_config.AgentConfig:
    """Build Harbor's Mirror agent configuration without copying the host API key.

    Args:
        request: Model endpoint and resolved dataset settings.
        mirror_kwargs: Mirror options supplied by the dataset.

    Returns:
        Harbor agent settings with the endpoint, model options, and prompt.
    """
    from harbor.models.trial import config as trial_config  # noqa: PLC0415

    settings = request.settings
    generation: dict[str, object] = {
        name: value
        for name in ("temperature", "top_p", "max_tokens", "reasoning_effort")
        if (value := getattr(settings, name)) is not None
    }
    # Mirror's extra_body is the final flat request body; the shared model channel
    # nests engine fields under extra_body before they reach this native boundary.
    model_options = dict(settings.model_kwargs)
    engine_options = model_options.pop("extra_body", {})
    if not isinstance(engine_options, dict):
        raise ValueError("model_kwargs.extra_body must be a mapping")
    body = options.merge_options(model_options, engine_options)
    generation["request_timeout_seconds"] = settings.client_timeout
    if body:
        generation["extra_body"] = body
    resolved = MirrorConfig(request.model_name, request.base_url, options.merge_options(mirror_kwargs, generation))
    kwargs: dict[str, object] = {"base_url": request.base_url, "mirror_kwargs": resolved.settings()}
    if settings.prompt_factory is not None:
        kwargs["prompt_factory"] = dataclasses.asdict(settings.prompt_factory)
    return trial_config.AgentConfig(name=IMPORT_PATH, model_name=request.model_name, kwargs=kwargs)
