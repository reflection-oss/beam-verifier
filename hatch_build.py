"""Bundle Mirror and its Linux runtime for execution inside task sandboxes."""

import hashlib
import os
import pathlib
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from typing import Any

import arpy
from hatchling.builders.hooks.plugin import interface

PAYLOAD = ("mirror-agent", "LICENSE", "NOTICE", "RIPGREP-LICENSE-MIT", "THIRD_PARTY_NOTICES.txt")
RUNTIME_LIBRARIES = {
    "libc6": ("libc.so.6", "libm.so.6", "libpthread.so.0", "libdl.so.2"),
    "libgcc-s1": ("libgcc_s.so.1",),
}
RUNTIME_PLATFORMS = {
    "x86_64": ("amd64", "ld-linux-x86-64.so.2"),
    "aarch64": ("arm64", "ld-linux-aarch64.so.1"),
}


def verified_archive(source: pathlib.Path, url: str, checksum: str, *, download: bool) -> pathlib.Path:
    """Check an archive's SHA-256, downloading it first if allowed and missing.

    Args:
        source: Local archive path, also used to save downloads.
        url: Archive download URL.
        checksum: Expected SHA-256.
        download: Whether to download a missing archive.

    Returns:
        The verified archive path.

    Raises:
        ValueError: If the archive's SHA-256 does not match.
        OSError: If the archive cannot be read or downloaded.
    """
    if not source.exists() and download:
        source.parent.mkdir(parents=True, exist_ok=True)
        # Publish only a complete, verified download to the shared build cache.
        with tempfile.TemporaryDirectory(dir=source.parent) as temporary:
            candidate = pathlib.Path(temporary) / source.name
            with urllib.request.urlopen(url, timeout=60) as response, candidate.open("wb") as output:
                shutil.copyfileobj(response, output)
            verified_archive(candidate, url, checksum, download=False)
            candidate.replace(source)
    with source.open("rb") as archive:
        actual = hashlib.file_digest(archive, "sha256").hexdigest()
    if actual != checksum:
        raise ValueError(f"Mirror archive checksum mismatch for {source.name}: expected {checksum}, got {actual}")
    return source


def unpack_archive(archive: pathlib.Path, destination: pathlib.Path) -> None:
    """Extract the Mirror binary and license notices from a verified archive.

    Args:
        archive: Verified release archive.
        destination: Directory receiving the files listed in `PAYLOAD`.

    Raises:
        ValueError: If files are missing, unexpected, duplicated, empty, or links.
    """
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        if sorted(member.name for member in members) != sorted(PAYLOAD) or any(
            not member.isfile() or member.size == 0 for member in members
        ):
            raise ValueError(f"Unexpected Mirror release payload in {archive.name}")
        destination.mkdir(parents=True, exist_ok=True)
        bundle.extractall(destination, members=members, filter="data")


def unpack_runtime(archive: pathlib.Path, destination: pathlib.Path, members: dict[str, str]) -> None:
    """Extract selected runtime files without installing the Debian package.

    Args:
        archive: Checksum-verified Debian package.
        destination: Directory for Mirror's separate runtime.
        members: Archive paths mapped to output filenames.

    Raises:
        ValueError: If a selected entry is empty or is not a regular file.
    """
    destination.mkdir(parents=True, exist_ok=True)
    with arpy.Archive(str(archive)) as package:
        package.read_all_headers()
        with tarfile.open(fileobj=package.archived_files[b"data.tar.xz"], mode="r:xz") as payload:
            for source, name in members.items():
                member = payload.getmember(source)
                if not member.isfile() or not member.size:
                    raise ValueError(f"Invalid runtime library {source!r} in {archive.name}")
                with payload.extractfile(member) as content, (destination / name).open("wb") as target:
                    shutil.copyfileobj(content, target)
                (destination / name).chmod(0o755 if name == "ld.so" else 0o644)


class MirrorBuildHook(interface.BuildHookInterface):
    """Bundle Linux sandbox binaries independently of the installation host."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        del version  # Unused: the agent version is independent of the package version.
        root = pathlib.Path(self.root)
        agent_version = self.config["agent-version"]
        embedded = root / "mirror-agent-archives"
        staged = os.environ.get("BEAM_VERIFIER_MIRROR_ARCHIVES")
        cache_root = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache"))
        cache = cache_root / "beam-verifier" / "mirror-agent" / agent_version
        archives = pathlib.Path(staged) if staged else embedded if embedded.is_dir() else cache
        self._staging = tempfile.TemporaryDirectory(prefix="beam-verifier-agent-")
        for architecture, checksum in self.config["archive-sha256"].items():
            filename = f"mirror-agent-{agent_version}-linux-{architecture}.tar.gz"
            url = f"https://github.com/reflection-oss/mirror/releases/download/mirror-agent-v{agent_version}/{filename}"
            archive = verified_archive(
                archives / filename, url, checksum, download=staged is None and not embedded.is_dir()
            )
            if self.target_name == "sdist":
                # Source releases rebuild offline and verify the same pins as checkouts.
                build_data["force_include"][str(archive)] = f"mirror-agent-archives/{filename}"
            else:
                destination = pathlib.Path(self._staging.name) / architecture
                unpack_archive(archive, destination)
                # These binaries run in the task sandbox, not the installation host.
                target = f"beam_verifier/bin/mirror-agent/linux-{architecture}"
                for name in PAYLOAD:
                    build_data["force_include"][str(destination / name)] = f"{target}/{name}"
        self._include_runtime(build_data)
        self._include_connectors(build_data)

    def _include_runtime(self, build_data: dict[str, Any]) -> None:
        """Bundle the Linux loader and libraries Mirror uses in task sandboxes."""
        root = pathlib.Path(self.root)
        embedded = root / "mirror-runtime-archives"
        staged = os.environ.get("BEAM_VERIFIER_RUNTIME_ARCHIVES")
        cache = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache"))
        archives = (
            pathlib.Path(staged) if staged else embedded if embedded.is_dir() else cache / "beam-verifier/runtime"
        )
        for architecture in self.config["archive-sha256"]:
            debian_arch, loader = RUNTIME_PLATFORMS[architecture]
            destination = pathlib.Path(self._staging.name) / architecture / "runtime"
            for name, libraries in RUNTIME_LIBRARIES.items():
                pinned = self.config["runtime"][name]
                url = pinned["url"].format(architecture=debian_arch)
                filename = url.rsplit("/", 1)[1]
                archive = verified_archive(
                    archives / filename,
                    url,
                    pinned["sha256"][architecture],
                    download=staged is None and not embedded.is_dir(),
                )
                if self.target_name == "sdist":
                    build_data["force_include"][str(archive)] = f"mirror-runtime-archives/{filename}"
                else:
                    members = {f"./lib/{architecture}-linux-gnu/{library}": library for library in libraries}
                    if name == "libc6":
                        members[f"./lib/{architecture}-linux-gnu/{loader}"] = "ld.so"
                    unpack_runtime(archive, destination, members)
            if self.target_name != "sdist":
                target = f"beam_verifier/bin/mirror-agent/linux-{architecture}/runtime"
                build_data["force_include"][str(destination)] = target
        target = "beam_verifier/bin/mirror-agent/runtime-licenses"
        build_data["force_include"][str(root / "runtime-licenses")] = (
            "runtime-licenses" if self.target_name == "sdist" else target
        )

    def _include_connectors(self, build_data: dict[str, Any]) -> None:
        """Bundle connector binaries, compiling them only when not already supplied."""
        root = pathlib.Path(self.root)
        source = root / "beam_verifier/benchmarks/proxies/connector"
        embedded = root / "tunnel-connector-binaries"
        staged = os.environ.get("BEAM_VERIFIER_TUNNEL_BINARIES")
        supplied = pathlib.Path(staged) if staged else embedded if embedded.is_dir() else None
        for architecture, goarch in (("x86_64", "amd64"), ("aarch64", "arm64")):
            binary = (
                supplied / f"linux-{architecture}" / "tunnel-connector"
                if supplied is not None
                else pathlib.Path(self._staging.name) / f"tunnel-{architecture}"
            )
            if supplied is None:
                if shutil.which("go") is None:
                    raise RuntimeError("Building a checkout requires Go 1.25+ or BEAM_VERIFIER_TUNNEL_BINARIES")
                subprocess.run(
                    [
                        "go",
                        "build",
                        "-mod=readonly",
                        "-trimpath",
                        "-buildvcs=false",
                        "-ldflags=-s -w",
                        "-o",
                        str(binary),
                        ".",
                    ],
                    cwd=source,
                    env={**os.environ, "CGO_ENABLED": "0", "GOOS": "linux", "GOARCH": goarch},
                    check=True,
                )
            if not binary.is_file() or not os.access(binary, os.X_OK):
                raise FileNotFoundError(f"Missing executable tunnel connector: {binary}")
            target = (
                f"tunnel-connector-binaries/linux-{architecture}/tunnel-connector"
                if self.target_name == "sdist"
                else f"beam_verifier/bin/tunnel-connector/linux-{architecture}/tunnel-connector"
            )
            build_data["force_include"][str(binary)] = target

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        del version, build_data, artifact_path  # Unused.
        self._staging.cleanup()
