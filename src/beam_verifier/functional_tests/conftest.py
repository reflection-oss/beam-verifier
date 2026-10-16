"""Pytest options and endpoint fixtures shared by the functional-test suites."""

import inspect
import os
from collections.abc import Iterator

import httpx
import openai
import pytest

from beam_verifier import options
from beam_verifier.functional_tests import completions, constants


@pytest.hookimpl(optionalhook=True)
def pytest_json_runtest_metadata(item, call):
    """Attach each test's one-line docstring to the JSON report as its rule.

    Reports render that rule next to a failing test id.
    """
    del call  # Unused.
    doc = inspect.getdoc(item.function) or ""
    return {"rule": doc.splitlines()[0] if doc else ""}


def pytest_addoption(parser):
    parser.addoption("--base-url", action="store", help="OpenAI-compatible base URL")
    parser.addoption("--api-key", action="store", default=None, help="API key (else $REFLECTION_API_KEY / EMPTY)")
    parser.addoption("--model", action="store", help="Model id under test")
    parser.addoption("--parse-url", help="Exact URL for token IDs to an assistant message")
    parser.addoption("--format-url", help="Exact URL for chat messages to token IDs")
    parser.addoption("--completions-url", help="Exact URL for raw Harmony completions")
    parser.addoption(
        "--model-kwargs",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Option sent with every model request, e.g. no_stop_trim=true (SGLang) or "
        "include_stop_str_in_output=true (vLLM). Repeatable; values are YAML-decoded.",
    )


@pytest.fixture(autouse=True, scope="session")
def model_kwargs(request: pytest.FixtureRequest) -> dict[str, object]:
    """The session's `--model-kwargs`, also applied to the chat gateway.

    `generate_completion` is the single gateway for every chat generation, so
    setting it here is what lets a test honor the flag without taking it.
    """
    try:
        kwargs = options.parse_kwargs(tuple(request.config.getoption("--model-kwargs")), option_name="--model-kwargs")
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error
    completions.MODEL_KWARGS = kwargs
    return kwargs


@pytest.fixture
def base_url(request):
    return request.config.getoption("--base-url")


@pytest.fixture
def api_key(request):
    return (
        request.config.getoption("--api-key")
        or os.getenv("PYTEST_API_KEY")
        or os.getenv("REFLECTION_API_KEY")
        or "EMPTY"
    )


@pytest.fixture
def model(request):
    return request.config.getoption("--model")


@pytest.fixture(params=[False, True], ids=["non_streaming", "streaming"])
def stream(request):
    """Streaming axis: a test that takes `stream` runs once per transport mode."""
    return request.param


@pytest.fixture
def endpoint_client(api_key: str) -> Iterator[httpx.Client]:
    """Share authenticated HTTP setup for the direct endpoint suites."""
    with httpx.Client(
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=httpx.Timeout(constants.REQUEST_TIMEOUT_SECONDS, connect=constants.CONNECT_TIMEOUT_SECONDS),
    ) as client:
        yield client


@pytest.fixture
def http_client(base_url, api_key):
    """Raw OpenAI SDK client for the endpoint under test.

    No wrappers beyond the SDK's own, so tests observe the HTTP contract
    as-is: retries stay at the SDK default (connection errors / 408 / 429 /
    5xx), and the read timeout is bounded.
    """
    return openai.OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=openai.Timeout(constants.REQUEST_TIMEOUT_SECONDS, connect=constants.CONNECT_TIMEOUT_SECONDS),
    )
