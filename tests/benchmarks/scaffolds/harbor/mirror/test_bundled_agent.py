"""Check bundled Mirror binaries and runtime files in the installed package."""

import hashlib
import pathlib
import platform
import subprocess
import sys
from importlib import metadata

import pytest

import beam_verifier


def installed_binary(architecture: str) -> pathlib.Path:
    """Find the installed Mirror binary, including with an editable installation."""
    package = beam_verifier.__name__.replace(".", "/")
    distribution = metadata.distribution("beam-verifier")
    return pathlib.Path(str(distribution.locate_file(f"{package}/bin/mirror-agent/linux-{architecture}/mirror-agent")))


@pytest.mark.parametrize(
    ("architecture", "checksum"),
    [
        ("x86_64", "84662333f0c72b2c77720f5fe9c72099a0800a949bedce7e7b9dce1acd88355c"),
        ("aarch64", "b2f4fbadcf1079d5efa8943df91392ae5f323df2d1bcfe7e0dc44b31e98918a8"),
    ],
)
def test_installed_binary_matches_published_release(architecture: str, checksum: str) -> None:
    binary = installed_binary(architecture)
    with binary.open("rb") as source:
        assert hashlib.file_digest(source, "sha256").hexdigest() == checksum
    assert binary.stat().st_mode & 0o111, f"Not executable: {binary}"
    assert {path.name for path in binary.parent.iterdir()} == {
        "mirror-agent",
        "LICENSE",
        "NOTICE",
        "RIPGREP-LICENSE-MIT",
        "THIRD_PARTY_NOTICES.txt",
        "runtime",
    }
    assert {path.name for path in (binary.parent / "runtime").iterdir()} == {
        "ld.so",
        "libc.so.6",
        "libm.so.6",
        "libpthread.so.0",
        "libdl.so.2",
        "libgcc_s.so.1",
    }


@pytest.mark.skipif(
    sys.platform != "linux" or platform.machine() not in {"x86_64", "aarch64"},
    reason="The sandbox executable requires a compatible Linux host for this smoke test",
)
def test_installed_binary_reports_release_version(tmp_path: pathlib.Path) -> None:
    binary = installed_binary(platform.machine())
    runtime = binary.parent / "runtime"
    result = subprocess.run(
        [str(runtime / "ld.so"), "--library-path", str(runtime), str(binary), "--version"],
        cwd=tmp_path,
        env={"HOME": str(tmp_path), "TMPDIR": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert result.stdout.strip() == "mirror-agent 0.1.8"
