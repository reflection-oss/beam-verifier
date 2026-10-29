import pathlib
from typing import Any

import pytest
import yaml

from beam_verifier.functional_tests import completions


@pytest.fixture
def chat_examples() -> dict[str, Any]:
    """Loads fresh examples so tests can change them independently."""
    return yaml.safe_load((pathlib.Path(__file__).parent / "fixtures/chat.yaml").read_text())


@pytest.fixture
def completion(chat_examples: dict[str, Any]) -> completions.Completion:
    return completions.Completion(**chat_examples["completion"])


@pytest.fixture
def tool_call(chat_examples: dict[str, Any]) -> dict[str, Any]:
    return chat_examples["tool_call"]


@pytest.fixture
def weather_tool(chat_examples: dict[str, Any]) -> dict[str, Any]:
    return chat_examples["weather_tool"]
