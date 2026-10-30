"""Test package builds with synthetic Mirror and runtime archives, offline."""

import hashlib
import io
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest

import beam_verifier

pytestmark = [pytest.mark.source_checkout]


@pytest.fixture
def project(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a temporary project with the real build hook and synthetic archives."""
    shutil.copyfile(pathlib.Path(__file__).parents[1] / "hatch_build.py", tmp_path / "hatch_build.py")
    for architecture in ("x86_64", "aarch64"):
        binary = tmp_path / "tunnel-connector-binaries" / f"linux-{architecture}" / "tunnel-connector"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"synthetic connector")
        binary.chmod(0o755)
    archives = tmp_path / "mirror-agent-archives"
    archives.mkdir()
    archive_path = archives / "mirror-agent-0.0.1-linux-x86_64.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name in ("mirror-agent", "LICENSE", "NOTICE", "RIPGREP-LICENSE-MIT", "THIRD_PARTY_NOTICES.txt"):
            content = f"synthetic {name}".encode()
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            entry.mode = 0o755 if name == "mirror-agent" else 0o644
            archive.addfile(entry, io.BytesIO(content))
    checksum = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    runtime = tmp_path / "mirror-runtime-archives"
    runtime.mkdir()
    pins = ""
    for name, libraries in {
        "libc6": ("ld-linux-x86-64.so.2", "libc.so.6", "libm.so.6", "libpthread.so.0", "libdl.so.2"),
        "libgcc-s1": ("libgcc_s.so.1",),
    }.items():
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:xz") as archive:
            for library in libraries:
                content = f"synthetic {library}".encode()
                member = tarfile.TarInfo(f"./lib/x86_64-linux-gnu/{library}")
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
        data = buffer.getvalue()
        header = f"{'data.tar.xz':<16}{0:<12}{0:<6}{0:<6}{0o100644:<8o}{len(data):<10}`\n".encode()
        deb = runtime / f"{name}_amd64.deb"
        deb.write_bytes(b"!<arch>\n" + header + data + (b"\n" if len(data) % 2 else b""))
        pins += f'''
[tool.hatch.build.hooks.custom.runtime.{name}]
url = "https://example.invalid/{name}_{{architecture}}.deb"
sha256 = {{x86_64 = "{hashlib.sha256(deb.read_bytes()).hexdigest()}"}}
'''
    (tmp_path / "runtime-licenses").mkdir()
    (tmp_path / "runtime-licenses" / "LICENSE.txt").write_text("Synthetic runtime license\n")
    package = tmp_path / "src" / "example"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""Synthetic package."""\n')
    (tmp_path / "pyproject.toml").write_text(f"""
[project]
name = "example"
version = "1.0.0"
[tool.hatch.build.targets.wheel]
packages = ["src/example"]
sources = ["src"]
[tool.hatch.build.targets.sdist]
include = ["src", "hatch_build.py", "pyproject.toml"]
[tool.hatch.build.hooks.custom]
path = "hatch_build.py"
agent-version = "0.0.1"
archive-sha256 = {{x86_64 = "{checksum}"}}
{pins}
""")
    return tmp_path


def build(project: pathlib.Path, target: str, archives: pathlib.Path | None = None) -> subprocess.CompletedProcess[str]:
    """Build using supplied archives, with downloads disabled."""
    return subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", target],
        cwd=project,
        env={
            **os.environ,
            "BEAM_VERIFIER_MIRROR_ARCHIVES": str(archives or project / "mirror-agent-archives"),
            "BEAM_VERIFIER_RUNTIME_ARCHIVES": str(project / "mirror-runtime-archives"),
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_build_wheel_bundles_executable_and_notices(project: pathlib.Path) -> None:
    result = build(project, "wheel")
    assert result.returncode == 0, result.stderr
    with zipfile.ZipFile(next((project / "dist").glob("*.whl"))) as wheel:
        prefix = beam_verifier.__name__.replace(".", "/") + "/bin/mirror-agent/linux-x86_64/"
        assert wheel.read(prefix + "mirror-agent") == b"synthetic mirror-agent"
        assert wheel.read(prefix + "THIRD_PARTY_NOTICES.txt") == b"synthetic THIRD_PARTY_NOTICES.txt"
        assert wheel.getinfo(prefix + "mirror-agent").external_attr >> 16 & 0o111
        assert wheel.read(prefix + "runtime/ld.so") == b"synthetic ld-linux-x86-64.so.2"
        assert wheel.read(prefix + "runtime/libgcc_s.so.1") == b"synthetic libgcc_s.so.1"
        assert wheel.getinfo(prefix + "runtime/ld.so").external_attr >> 16 & 0o111
        package = beam_verifier.__name__.replace(".", "/")
        connector = f"{package}/bin/tunnel-connector/linux-aarch64/tunnel-connector"
        assert wheel.read(connector) == b"synthetic connector"
        assert wheel.getinfo(connector).external_attr >> 16 & 0o111


def test_build_sdist_rebuilds_wheel_without_network(project: pathlib.Path) -> None:
    result = build(project, "sdist")
    assert result.returncode == 0, result.stderr
    with tarfile.open(next((project / "dist").glob("*.tar.gz"))) as sdist:
        sdist.extractall(project / "rebuild", filter="data")
    environment = os.environ.copy()
    environment.pop("BEAM_VERIFIER_MIRROR_ARCHIVES", None)
    environment.pop("BEAM_VERIFIER_RUNTIME_ARCHIVES", None)
    rebuilt = project / "rebuild" / "example-1.0.0"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
def reject_network(event, args):
    if event == "socket.connect":
        raise RuntimeError("Source rebuild attempted network access")
sys.addaudithook(reject_network)
from hatchling import build
build.build_wheel("dist")
""",
        ],
        cwd=rebuilt,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert next((rebuilt / "dist").glob("*.whl")).is_file()


@pytest.mark.parametrize(
    "archive",
    ["mirror-agent-archives/mirror-agent-0.0.1-linux-x86_64.tar.gz", "mirror-runtime-archives/libc6_amd64.deb"],
)
def test_build_rejects_corrupt_staged_archive(project: pathlib.Path, archive: str) -> None:
    (project / archive).write_bytes(b"corrupt")
    result = build(project, "wheel")
    assert result.returncode != 0
    assert "Mirror archive checksum mismatch" in result.stderr


def test_build_rejects_missing_staged_archive(project: pathlib.Path) -> None:
    result = build(project, "wheel", project / "missing")
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr
