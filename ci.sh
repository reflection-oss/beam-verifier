#!/usr/bin/env bash
# Check and test the installed wheel using the checked-in dependency lock.
# Publishing separately checks installation without that lock or its overrides.
set -euo pipefail

if [[ "$(uv --version | cut -d ' ' -f2)" != "0.10.10" ]]; then
  echo 'error: uv 0.10.10 is required' >&2
  exit 1
fi
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
export UV_PROJECT_ENVIRONMENT="$WORK_DIR/.venv"
export PYTHONDONTWRITEBYTECODE=1
WHEEL_DIR="${REFLECTION_CI_WHEEL_DIR:-$WORK_DIR/wheels}"

uv sync --locked --extra dev --python 3.12
uv run --no-sync ruff check --no-cache .
uv run --no-sync ruff format --check --no-cache .
uv run --no-sync mypy --cache-dir "$WORK_DIR/mypy"
# Source discovery has a different contract in an editable installation.
uv run --no-sync pytest -q -p no:cacheprovider tests "$@"
# Build the wheel from the sdist so both release artifacts contain usable source.
uv build --out-dir "$WHEEL_DIR"
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps --reinstall "$WHEEL_DIR"/*.whl
uv run --no-sync pytest -q -p no:cacheprovider -m 'not source_checkout' tests "$@"
