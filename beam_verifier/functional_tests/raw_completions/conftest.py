"""Direct access to a user's raw completions endpoint."""

import pytest


@pytest.fixture
def completions_url(request: pytest.FixtureRequest) -> str:
    url: str | None = request.config.getoption("--completions-url")
    if not url:
        pytest.skip("no --completions-url")
    return url
