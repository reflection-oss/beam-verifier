"""Opt-in access to a provider's parser endpoint without generation."""

import httpx
import pytest

from beam_verifier.functional_tests import parser_api


@pytest.fixture
def parser_client(
    request: pytest.FixtureRequest, endpoint_client: httpx.Client, model: str | None
) -> parser_api.Parser:
    url = request.config.getoption("--parse-url")
    if not url:
        pytest.skip("no --parse-url")
    return parser_api.Parser(endpoint_client, url, model)
