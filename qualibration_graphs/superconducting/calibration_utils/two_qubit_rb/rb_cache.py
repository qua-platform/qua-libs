"""Lightweight file-system cache for pre-computed RB circuit sequences.

The cache is keyed on the parameters that determine StandardRB / interleaved
output plus the circuit encoding version. Each entry is a small JSON file
stored under a configurable directory.

``RB_ENCODING_VERSION = 3`` is analog-XY gate-only lists (opcodes 0–37) plus
``basis_gates`` in the cache key. Packed packets are built after load so a
later change in chunk capacity does not retranspile. Unversioned, v1
(marker-terminated), and v2 ZX files are acquisition misses and are left on
disk; they must not be fed to the analog-XY executor. Reanalysis may still
read statistics via :func:`try_load_statistics`.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .packing import (
    RB_ENCODING_VERSION,
    CircuitPackingError,
    validate_circuit_list,
)

_STATISTICS_KEYS = (
    "average_gates_per_clifford",
    "avg_1q_per_clifford",
    "avg_cz_per_clifford",
    "depth_summaries",
)

# Default analog-XY basis; included in versioned cache keys so ZX and XY
# encodings cannot share an acquisition file.
DEFAULT_RB_BASIS_GATES: tuple[str, ...] = ("cz", "sx", "x", "ry", "y")


def cache_key(
    seed: int,
    circuit_depths: list[int],
    num_circuits_per_depth: int,
    *,
    target_gate: str | None = None,
    encoding_version: int | None = RB_ENCODING_VERSION,
    basis_gates: Sequence[str] | None = DEFAULT_RB_BASIS_GATES,
) -> str:
    """Return a hex SHA-256 digest that uniquely identifies an RB config.

    When *target_gate* is supplied the hash includes it, so standard and
    interleaved caches never collide. *encoding_version* defaults to
    :data:`RB_ENCODING_VERSION` so v3 analog-XY acquisition does not share
    keys with v2 ZX or marker-terminated (unversioned) files. Versioned keys
    also include *basis_gates* (default analog XY). Pass
    ``encoding_version=None`` to reconstruct a legacy key for statistics-only
    lookup (no version, no basis). Pass ``basis_gates=None`` with an explicit
    old *encoding_version* to reconstruct a pre-basis versioned key.
    """
    blob_dict: dict[str, Any] = {
        "seed": seed,
        "circuit_depths": sorted(circuit_depths),
        "num_circuits_per_depth": num_circuits_per_depth,
    }
    if target_gate is not None:
        blob_dict["target_gate"] = target_gate
    if encoding_version is not None:
        blob_dict["encoding_version"] = encoding_version
        if basis_gates is not None:
            blob_dict["basis_gates"] = sorted(basis_gates)
    blob = json.dumps(blob_dict, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _read_json(cache_dir: Path, key: str) -> dict[str, Any] | None:
    path = Path(cache_dir) / f"{key}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _payload_encoding_version(data: dict[str, Any]) -> int | None:
    if "encoding_version" not in data:
        return None
    value = data["encoding_version"]
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def try_load(cache_dir: Path, key: str) -> dict[str, Any] | None:
    """Return cached v3 analog-XY gate-only data, or *None* on a cache miss.

    Missing files, JSON errors, unversioned/v1/v2 ZX encodings, and malformed
    v3 payloads are misses. Old files are not deleted and are not reinterpreted
    as analog-XY sequences.
    """
    data = _read_json(cache_dir, key)
    if data is None:
        return None
    if _payload_encoding_version(data) != RB_ENCODING_VERSION:
        return None
    circuits = data.get("circuits_as_ints")
    if not isinstance(circuits, list):
        return None
    try:
        data["circuits_as_ints"] = validate_circuit_list(circuits)
    except CircuitPackingError:
        return None
    return data


def try_load_statistics(cache_dir: Path, key: str) -> dict[str, Any] | None:
    """Return gate-count summaries from any encoding, never circuit opcodes.

    Intended for ``load_data`` / IRB overlay reanalysis of legacy cache files.
    The returned dict has no ``circuits_as_ints`` key so callers cannot feed
    marker-terminated or ZX-v2 lists into the analog-XY executor.
    """
    data = _read_json(cache_dir, key)
    if data is None:
        return None
    stats = {k: data[k] for k in _STATISTICS_KEYS if k in data}
    if not stats:
        return None
    return stats


def try_load_legacy_statistics(
    cache_dir: Path,
    seed: int,
    circuit_depths: list[int],
    num_circuits_per_depth: int,
    *,
    target_gate: str | None = None,
) -> dict[str, Any] | None:
    """Statistics-only lookup using the unversioned cache key (pre-v2 files)."""
    legacy_key = cache_key(
        seed,
        circuit_depths,
        num_circuits_per_depth,
        target_gate=target_gate,
        encoding_version=None,
    )
    return try_load_statistics(cache_dir, legacy_key)


def save(cache_dir: Path, key: str, data: dict[str, Any]) -> None:
    """Atomically write *data* as JSON (write to tmp then rename).

    Ensures ``encoding_version`` is :data:`RB_ENCODING_VERSION`. Validates
    ``circuits_as_ints`` when present so the on-disk payload is gate-only.
    """
    payload = dict(data)
    payload["encoding_version"] = RB_ENCODING_VERSION
    circuits = payload.get("circuits_as_ints")
    if circuits is not None:
        payload["circuits_as_ints"] = validate_circuit_list(circuits)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{key}.json"
    fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp_path, target)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
