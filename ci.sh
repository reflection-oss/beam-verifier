#!/usr/bin/env bash
# Check and test the installed wheel using the checked-in dependency lock.
# Publishing separately checks installation without that lock or its overrides.
set -euo pipefail

# Pinned to the first uv release with webpki-root-certs 1.0.10, which drops the
# compromised eMudhra root (INC-2436); any later release is safe.
if [[ "$(uv --version | cut -d ' ' -f2)" != "0.13.4" ]]; then
  echo 'error: uv 0.13.4 is required' >&2
  exit 1
fi
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
export UV_PROJECT_ENVIRONMENT="$WORK_DIR/.venv"
export PYTHONDONTWRITEBYTECODE=1
WHEEL_DIR="${REFLECTION_CI_WHEEL_DIR:-$WORK_DIR/wheels}"

# Checkout builds compile the static sandbox connector. Release sdists already
# carry it, so installing a published package never needs this toolchain.
if ! command -v go >/dev/null && [[ -z "${BEAM_VERIFIER_TUNNEL_BINARIES:-}" ]]; then
  case "$(uname -sm)" in
    'Linux x86_64')
      GO_ARCH=amd64
      GO_SHA256=234828b7a89e0e303d2556310ee549fbcf253d28de937bac3da13d6294262ac1
      ;;
    'Linux aarch64')
      GO_ARCH=arm64
      GO_SHA256=8b5884aef89600aef5b0b051fb971f11f49bb996521e911f30f02a66884f7bd2
      ;;
    *)
      echo 'error: install Go 1.25+ before running checkout CI' >&2
      exit 1
      ;;
  esac
  uv run --no-project --python 3.12 python - "$GO_ARCH" "$WORK_DIR/go.tar.gz" <<'PYTHON'
import shutil
import sys
import urllib.request

with urllib.request.urlopen(f"https://go.dev/dl/go1.25.12.linux-{sys.argv[1]}.tar.gz", timeout=60) as source:
    with open(sys.argv[2], "wb") as target:
        shutil.copyfileobj(source, target)
PYTHON
  echo "$GO_SHA256  $WORK_DIR/go.tar.gz" | sha256sum --check --status
  tar -xzf "$WORK_DIR/go.tar.gz" -C "$WORK_DIR"
  export PATH="$WORK_DIR/go/bin:$PATH"
fi

if command -v go >/dev/null; then
  go -C beam_verifier/benchmarks/proxies/connector vet ./...
fi

uv sync --locked --extra dev --python 3.12
uv run --no-sync ruff check --no-cache .
uv run --no-sync ruff format --check --no-cache .
uv run --no-sync mypy --cache-dir "$WORK_DIR/mypy"
# Source discovery has a different contract in an editable installation.
uv run --no-sync pytest -q -p no:cacheprovider tests "$@"
# Build the wheel from the sdist so both release artifacts contain usable source.
uv build --out-dir "$WHEEL_DIR"
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps --reinstall "$WHEEL_DIR"/*.whl
# Run outside the checkout so a flat source package cannot shadow the wheel.
cp -R tests pyproject.toml "$WORK_DIR/"
cd "$WORK_DIR"
"$UV_PROJECT_ENVIRONMENT/bin/python" -m pytest -q -p no:cacheprovider -m 'not source_checkout' tests "$@"
