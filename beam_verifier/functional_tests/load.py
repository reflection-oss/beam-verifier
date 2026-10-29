"""Loads YAML test-case fixtures."""

import pathlib
import re
from typing import Any

import yaml


def load_test_cases(name: str, fixture_dir: pathlib.Path) -> dict[str, Any]:
    """Load `<fixture_dir>/<name>.yaml`; each suite directory owns its fixtures."""
    yaml_path = fixture_dir / f"{name}.yaml"
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def case_id_generator(case: dict[str, Any]) -> str | None:
    """Generate a test ID from the case's 'case_id' field, or use a default."""
    case_id = case.get("case_id")
    if isinstance(case_id, str | int):
        return re.sub(r"\W|^(?=\d)", "_", str(case_id))
    return None
