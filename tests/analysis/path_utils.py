"""Helpers for resolving paths in analysis tests."""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path) -> Path:
    """Find the repository root by locating tests/ and qualibration_graphs/."""
    current = start
    while current != current.parent:
        if (current / "tests").is_dir() and (current / "qualibration_graphs").is_dir():
            return current
        current = current.parent
    raise FileNotFoundError("Could not locate repo root containing tests/ and qualibration_graphs/.")
