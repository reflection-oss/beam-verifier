"""Test the bundled Bash scripts locally without a model endpoint or sandbox."""

import pathlib
import shlex
import subprocess
import sys
from importlib import resources

import portpicker
import pytest

from beam_verifier.benchmarks.scaffolds.harbor import mirror


@pytest.mark.parametrize(("status", "exit_code"), [(204, 0), (500, 1)])
def test_turn_script_posts_instruction_bytes_and_propagates_failure(
    tmp_path: pathlib.Path, status: int, exit_code: int
) -> None:
    instruction = tmp_path / "instruction ' $(touch forbidden).txt"
    instruction.write_text("Fix café. $(touch forbidden) `echo literal`\n" * 10000)
    received = tmp_path / "received.txt"
    script = resources.files(mirror).joinpath("turn.sh").read_text(encoding="utf-8")
    port = portpicker.pick_unused_port()
    result = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "turn.sh",
            str(instruction),
            str(tmp_path / "server ' log"),
            f"127.0.0.1:{port}",
            "10",
            "10",
            sys.executable,
            str(pathlib.Path(__file__).parent / "fixtures/mirror_protocol_server.py"),
            str(port),
            str(status),
            str(received),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == exit_code, result.stderr
    assert received.read_bytes() == instruction.read_bytes()
    assert not (tmp_path / "forbidden").exists()


def test_snapshot_processes_ignores_zombies_and_preserves_start_time(tmp_path: pathlib.Path) -> None:
    running = tmp_path / "42"
    zombie = tmp_path / "43"
    running.mkdir()
    zombie.mkdir()
    (running / "stat").write_text("42 (name with spaces) S 1 " + "0 " * 17 + "500\n")
    (zombie / "stat").write_text("43 (zombie) Z 1 " + "0 " * 17 + "600\n")
    script = resources.files(mirror).joinpath("processes.sh").read_text(encoding="utf-8")
    result = subprocess.run(
        ["bash", "-c", script, "processes.sh", "snapshot", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["42 500"]


def test_stopping_a_turn_kills_a_reused_pid_and_preserves_the_original_process(tmp_path: pathlib.Path) -> None:
    reused = tmp_path / "42"
    original = tmp_path / "43"
    reused.mkdir()
    original.mkdir()
    (reused / "stat").write_text("42 (synthetic) S 1 " + "0 " * 17 + "500\n")
    (original / "stat").write_text("43 (synthetic) S 1 " + "0 " * 17 + "100\n")
    # Exercise the real shell cleanup with a synthetic process table and no signals.
    record_signal = f"""
      kill() {{
        printf 'signal %s pid %s\\n' "$1" "$2"
        [[ "$2" == 42 ]] || exit 99
        command rm -f {shlex.quote(str(reused / "stat"))}
      }}
    """
    script = resources.files(mirror).joinpath("processes.sh").read_text(encoding="utf-8")
    result = subprocess.run(
        ["bash", "-c", record_signal + script, "processes.sh", "stop", str(tmp_path), "1 1", "42 100", "43 100"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["signal -TERM pid 42", "signal -CONT pid 42"]
    assert (original / "stat").is_file()


@pytest.mark.parametrize(
    ("baseline", "error"),
    [
        ([], "empty baseline"),
        (["1 0", "rm -rf /"], "not a baseline line"),
        (["1 0", "42 $(touch forbidden)"], "not a baseline line"),
        (["1 0", "42 500\n"], "not a baseline line"),
    ],
)
def test_stop_script_rejects_invalid_baselines_before_signaling(
    tmp_path: pathlib.Path, baseline: list[str], error: str
) -> None:
    process = tmp_path / "42"
    process.mkdir()
    (process / "stat").write_text("42 (synthetic) S 1 " + "0 " * 17 + "500\n")
    script = resources.files(mirror).joinpath("processes.sh").read_text(encoding="utf-8")
    result = subprocess.run(
        [
            "bash",
            "-c",
            'kill() { echo "unexpected signal"; exit 99; };\n' + script,
            "processes.sh",
            "stop",
            str(tmp_path),
            *baseline,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1, result.stderr
    assert error in result.stderr
    assert not result.stdout, "Baseline validation must finish before any signals are sent"
    assert not (tmp_path / "forbidden").exists()
