"""Lightweight file-system cache for pre-computed RB circuit sequences.

The cache is keyed on the three parameters that fully determine the
StandardRB output: seed, circuit_lengths, and num_circuits_per_length.
Each entry is a small JSON file stored under a configurable directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def cache_key(
    seed: int,
    circuit_lengths: list[int],
    num_circuits_per_length: int,
    *,
    target_gate: str | None = None,
) -> str:
    """Return a hex SHA-256 digest that uniquely identifies an RB config.

    When *target_gate* is supplied the hash includes it, so standard and
    interleaved caches never collide.  When omitted the blob is byte-identical
    to the original implementation — existing caches stay valid.
    """
    blob_dict = {
        "seed": seed,
        "circuit_lengths": sorted(circuit_lengths),
        "num_circuits_per_length": num_circuits_per_length,
    }
    if target_gate is not None:
        blob_dict["target_gate"] = target_gate
    blob = json.dumps(blob_dict, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def try_load(cache_dir: Path, key: str) -> dict[str, Any] | None:
    """Return the cached data dict, or *None* on a cache miss."""
    path = Path(cache_dir) / f"{key}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def save(cache_dir: Path, key: str, data: dict[str, Any]) -> None:
    """Atomically write *data* as JSON (write to tmp then rename)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{key}.json"
    fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp_path, target)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
