"""Opt-in endpoint options for the benchmark integration test."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--base-url", help="Model endpoint URL; the live test skips when omitted")
    parser.addoption("--model", help="Model served by the endpoint")
    parser.addoption("--api-key", help="Model API key; defaults to REFLECTION_API_KEY or EMPTY")
