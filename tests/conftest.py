"""Pytest options for the one integration test among these unit tests.

`tests/` runs with no network by default. The integration test is opt-in and takes
the same options the functional suite does, so one spelling points either suite at
an endpoint:

    uv run pytest tests --base-url <url>/v1 --model <model-id>
"""

import os

import pytest


# The same spellings the functional suite takes, so one vocabulary points either
# suite at an endpoint. Collect one suite at a time: pytest loads both conftests when
# handed both directories, and the second registration of an option is an error.
def pytest_addoption(parser):
    parser.addoption(
        "--base-url",
        action="store",
        default=None,
        help="OpenAI-compatible base URL. Without it the integration test skips.",
    )
    parser.addoption(
        "--api-key",
        action="store",
        default=None,
        help="API key (else $REFLECTION_API_KEY / EMPTY)",
    )
    parser.addoption("--model", action="store", default=None, help="Model id under test")


@pytest.fixture
def api_key(request) -> str:
    """Key for the endpoint under test; endpoints that ignore it still need one."""
    return request.config.getoption("--api-key") or os.getenv("REFLECTION_API_KEY") or "EMPTY"


@pytest.fixture
def model(request) -> str:
    """Model id under test, required once an endpoint is given."""
    model_name = request.config.getoption("--model")
    if not model_name:
        pytest.fail("--base-url was given without --model; name the model the endpoint serves.")
    return model_name


@pytest.fixture
def live_endpoint(request, api_key) -> str:
    """Base URL of an endpoint that answered, skipping the test when none was given.

    Passing `--base-url` is a statement that an endpoint should be there, so an
    unreachable one fails rather than skipping quietly.
    """
    base_url = request.config.getoption("--base-url")
    if not base_url:
        pytest.skip("pass --base-url and --model to run the integration test")

    import openai

    try:
        openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=10).models.list()
    except Exception as error:  # noqa: BLE001 - reported as a test failure, whatever the client raised
        pytest.fail(f"--base-url {base_url} is not reachable: {type(error).__name__}: {error}")
    return base_url
