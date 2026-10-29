"""HTTP transport for raw completions."""

from typing import Literal

import httpx
import pydantic


class Completion(pydantic.BaseModel):
    """One unprocessed completion, including its termination status."""

    text: pydantic.StrictStr
    finish_reason: Literal["stop", "length"]


class CompletionResponse(pydantic.BaseModel):
    """The single-choice response requested by the raw-completions suite."""

    choices: list[Completion] = pydantic.Field(min_length=1, max_length=1)


def generate_completion(client: httpx.Client, url: str, body: dict[str, object]) -> Completion:
    """POST to the supplied URL and preserve raw text without parsing chat messages."""
    response = client.post(url, json=body)
    response.raise_for_status()
    return CompletionResponse.model_validate(response.json()).choices[0]
