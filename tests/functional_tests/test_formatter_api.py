"""Check the formatter contract using an httpx client over in-process HTTP."""

import copy
import json
from collections.abc import Callable

import httpx
import pydantic
import pytest

from beam_verifier.functional_tests import formatter_api


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer test-key"},
    )


@pytest.mark.parametrize("model", [None, "test-model"])
def test_formatter_preserves_exact_url_and_fixture_fields(model: str | None) -> None:
    url = "https://example.test/custom/render?version=2"
    chat: dict[str, object] = {
        "messages": [{"role": "assistant", "content": "Answer", "reasoning_content": "Think"}],
        "tools": [{"type": "function", "function": {"name": "weather"}}],
        "chat_template_kwargs": {"reasoning_effort": "low"},
        "add_generation_prompt": True,
        "add_special_tokens": False,
    }
    original = copy.deepcopy(chat)
    expected = {**chat, "model": model} if model is not None else chat

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == url
        assert request.headers["authorization"] == "Bearer test-key"
        assert json.loads(request.content) == expected
        return httpx.Response(200, json={"tokens": [151669, 42]})

    with _client(respond) as client:
        assert formatter_api.Formatter(client, url, model).render(chat) == [151669, 42]
    assert chat == original


@pytest.mark.parametrize("payload", [{"tokens": []}, {"tokens": [[1]]}, {"tokens": [True]}, {"token_ids": [1]}])
def test_invalid_tokenization_cannot_pass(payload: dict[str, object]) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    with _client(respond) as client, pytest.raises(pydantic.ValidationError):
        formatter_api.Formatter(client, "https://example.test/format").render({"messages": []})
