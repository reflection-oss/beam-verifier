"""Read/write a run on disk: run.json (manifest) + traces/<dataset>.jsonl (one Trace per line)."""

import datetime
import hashlib
import json
import pathlib
from typing import Any

from beam_verifier import records

RUNS_DIR_NAME = ".runs"


def package_root() -> pathlib.Path | None:
    """The package root, identified by the `pyproject.toml` above this module.

    The module sits at a different depth in the internal monorepo than in the
    public tree, so the root is discovered by walking up rather than by a fixed
    number of parents.

    Returns:
        The nearest ancestor directory containing `pyproject.toml`, or None when
        the package is installed without one (a built wheel, typically).
    """
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


def default_runs_dir() -> pathlib.Path:
    """The directory new runs are written to when no output directory is given.

    Returns:
        `.runs` under the package root, falling back to `.runs` under the current
        working directory when the package root cannot be located.
    """
    return (package_root() or pathlib.Path.cwd()) / RUNS_DIR_NAME


def request_hash(messages: list[dict[str, Any]]) -> str:
    return hashlib.md5(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _slug(model: str) -> str:
    return model.strip("/").replace("/", "-").replace("_", "-")[:48] or "model"


def new_run_dir(model: str, kind: str, base: pathlib.Path | None = None) -> pathlib.Path:
    base = base or default_runs_dir()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base / f"{stamp}-{_slug(model)}-{kind}"
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)
    return run_dir


class RunStore:
    def __init__(self, run_dir: pathlib.Path):
        self.dir = pathlib.Path(run_dir)
        (self.dir / "traces").mkdir(parents=True, exist_ok=True)

    def write_traces(self, dataset: str, traces: list[records.Trace]) -> None:
        path = self.dir / "traces" / f"{dataset}.jsonl"
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as f:
            for t in traces:
                f.write(t.model_dump_json() + "\n")
        temporary.replace(path)

    def write_manifest(self, manifest: records.RunManifest) -> None:
        temporary = self.dir / "run.json.tmp"
        temporary.write_text(manifest.model_dump_json(indent=2))
        temporary.replace(self.dir / "run.json")


def _upgrade_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Rename the keys of a manifest written under an earlier schema.

    Earlier runs named the results `evalsets` (later `datasets`), gave each
    result an `evalset` name, called the base URL `endpoint`, and recorded the
    verifier version under a package-specific key. Renaming them on read keeps
    those runs viewable, so a rename in the code does not strand a run already
    on disk.

    Args:
        payload: Decoded `run.json` contents.

    Returns:
        The payload, with any legacy keys renamed.
    """
    for legacy_results_key in ("evalsets", "datasets"):
        if legacy_results_key in payload:
            payload["dataset_results"] = payload.pop(legacy_results_key)
    if "endpoint" in payload:
        payload["base_url"] = payload.pop("endpoint")
    if "reflection_eval_version" in payload:
        payload["version"] = payload.pop("reflection_eval_version")
    for result in payload.get("dataset_results") or []:
        if isinstance(result, dict) and "evalset" in result:
            result["dataset"] = result.pop("evalset")
    return payload


def load_manifest(run_dir: pathlib.Path) -> records.RunManifest:
    """Reads a run's settings and results without loading its sample traces."""
    payload = _upgrade_manifest(json.loads((run_dir / "run.json").read_text()))
    return records.RunManifest.model_validate(payload)


def load_run(run_dir: pathlib.Path) -> tuple[records.RunManifest, dict[str, list[records.Trace]]]:
    run_dir = pathlib.Path(run_dir)
    manifest = load_manifest(run_dir)
    traces: dict[str, list[records.Trace]] = {}
    tdir = run_dir / "traces"
    if tdir.exists():
        for f in sorted(tdir.glob("*.jsonl")):
            # Split on "\n" only — NOT str.splitlines(), which also breaks on Unicode line
            # separators ( , \x85, ...) that JSON leaves unescaped inside string values.
            traces[f.stem] = [
                records.Trace.model_validate_json(line) for line in f.read_text().split("\n") if line.strip()
            ]
    return manifest, traces


def latest_run(base: pathlib.Path | None = None) -> pathlib.Path | None:
    base = base or default_runs_dir()
    if not base.exists():
        return None
    runs = [d for d in base.iterdir() if d.is_dir() and (d / "run.json").exists()]
    return max(runs, key=lambda d: d.stat().st_mtime) if runs else None


def resolve_run(arg: str | None, base: pathlib.Path | None = None) -> pathlib.Path | None:
    """A run path/dir, or the latest run if arg is None."""
    if arg:
        return pathlib.Path(arg)
    return latest_run(base)
