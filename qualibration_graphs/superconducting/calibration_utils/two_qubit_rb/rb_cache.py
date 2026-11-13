"""Lightweight file-system cache for pre-computed RB circuit sequences.

The cache is keyed on the parameters that determine StandardRB / interleaved
output, the single-qubit basis, and the circuit encoding version. Each entry is
a small JSON file stored under a configurable directory.

Versioned payloads hold gate-only opcode lists (0–37) per basis: XY is version 3
and XZ is version 4 (:data:`~calibration_utils.two_qubit_rb.packing.RB_ENCODING_VERSIONS`).
A payload is a hit only if its version and its ``single_qubit_basis`` stamp both match
the request. Packed packets are built after load so a later change in chunk capacity does
not retranspile. Unversioned, v1 (marker-terminated), and v2 ZX files are acquisition
misses and are left on disk; they must not be fed to the executor. Reanalysis may still
read statistics via :func:`try_load_statistics`.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .packing import (
    RB_ENCODING_VERSIONS,
    CircuitPackingError,
    SingleQubitBasis,
    validate_circuit_list,
)

_STATISTICS_KEYS = (
    "average_gates_per_clifford",
    "avg_1q_per_clifford",
    "avg_cz_per_clifford",
    "depth_summaries",
)

# Gate sets per single-qubit basis. XZ: ``rz`` is a virtual frame change, not a pulse.
RB_BASIS_GATES: dict[str, tuple[str, ...]] = {
    "XZ": ("cz", "rz", "sx", "x"),
    "XY": ("cz", "sx", "x", "ry", "y"),
}


def _key_blob(
    seed: int,
    circuit_depths: list[int],
    num_circuits_per_depth: int,
    target_gate: str | None,
) -> dict[str, Any]:
    blob_dict: dict[str, Any] = {
        "seed": seed,
        "circuit_depths": sorted(circuit_depths),
        "num_circuits_per_depth": num_circuits_per_depth,
    }
    if target_gate is not None:
        blob_dict["target_gate"] = target_gate
    return blob_dict


def _digest(blob_dict: dict[str, Any]) -> str:
    blob = json.dumps(blob_dict, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def cache_key(
    seed: int,
    circuit_depths: list[int],
    num_circuits_per_depth: int,
    *,
    single_qubit_basis: SingleQubitBasis,
    target_gate: str | None = None,
) -> str:
    """Return a hex SHA-256 digest that uniquely identifies an RB config and single-qubit basis.

    When *target_gate* is supplied the hash includes it, so standard and interleaved caches never
    collide. The basis, its encoding version, and its gate set are always hashed, so XZ and XY
    acquisitions never share a key.
    """
    blob_dict = _key_blob(seed, circuit_depths, num_circuits_per_depth, target_gate)
    blob_dict["single_qubit_basis"] = single_qubit_basis
    blob_dict["encoding_version"] = RB_ENCODING_VERSIONS[single_qubit_basis]
    blob_dict["basis_gates"] = sorted(RB_BASIS_GATES[single_qubit_basis])
    return _digest(blob_dict)


def legacy_cache_key(
    seed: int,
    circuit_depths: list[int],
    num_circuits_per_depth: int,
    *,
    target_gate: str | None = None,
) -> str:
    """Return the unversioned key (pre-v2 files) used for statistics-only lookup."""
    return _digest(_key_blob(seed, circuit_depths, num_circuits_per_depth, target_gate))


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


def try_load(
    cache_dir: Path,
    key: str,
    *,
    single_qubit_basis: SingleQubitBasis,
) -> dict[str, Any] | None:
    """Return cached gate-only data for *single_qubit_basis*, or *None* on a cache miss.

    A hit needs the encoding version of *single_qubit_basis* and a ``single_qubit_basis`` stamp equal
    to it. Missing files, JSON errors, other versions (including unversioned, v1, and v2 ZX),
    unstamped or other-basis payloads, and malformed payloads are misses. Old files are not deleted
    and are not reinterpreted as gate sequences.
    """
    data = _read_json(cache_dir, key)
    if data is None:
        return None
    if _payload_encoding_version(data) != RB_ENCODING_VERSIONS[single_qubit_basis]:
        return None
    if data.get("single_qubit_basis") != single_qubit_basis:
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
    marker-terminated or ZX-v2 lists into the executor.
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
    single_qubit_basis: SingleQubitBasis,
    target_gate: str | None = None,
) -> dict[str, Any] | None:
    """Statistics-only lookup under the unversioned key, used only when the basis is known to match.

    Unversioned files carry no ``single_qubit_basis`` stamp, so their basis-dependent statistics
    (gates and 1Q/CZ counts per Clifford) are skipped and this returns *None*.
    """
    key = legacy_cache_key(seed, circuit_depths, num_circuits_per_depth, target_gate=target_gate)
    data = _read_json(cache_dir, key)
    if data is None or data.get("single_qubit_basis") != single_qubit_basis:
        return None
    return try_load_statistics(cache_dir, key)


def save(
    cache_dir: Path,
    key: str,
    data: dict[str, Any],
    *,
    single_qubit_basis: SingleQubitBasis,
) -> None:
    """Atomically write *data* as JSON (write to tmp then rename).

    Stamps ``encoding_version`` and ``single_qubit_basis`` for *single_qubit_basis*. Validates
    ``circuits_as_ints`` when present so the on-disk payload is gate-only.
    """
    payload = dict(data)
    payload["encoding_version"] = RB_ENCODING_VERSIONS[single_qubit_basis]
    payload["single_qubit_basis"] = single_qubit_basis
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
