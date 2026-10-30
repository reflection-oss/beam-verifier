"""Opt-in access to a provider's formatter endpoint without generation."""

import httpx
import pytest

from beam_verifier.functional_tests import formatter_api


@pytest.fixture
def formatter_client(
    request: pytest.FixtureRequest, endpoint_client: httpx.Client, model: str | None
) -> formatter_api.Formatter:
    url = request.config.getoption("--format-url")
    if not url:
        pytest.skip("no --format-url")
    return formatter_api.Formatter(endpoint_client, url, model)
