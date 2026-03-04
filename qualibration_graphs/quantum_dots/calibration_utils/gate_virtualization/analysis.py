"""Shared analysis helpers for gate virtualization calibration nodes.

Contains common operations (I/Q processing, matrix updates) used by all
three calibration nodes.  Node-specific analysis lives in the dedicated
modules:
- ``sensor_compensation_analysis``
- ``virtual_plunger_analysis``
- ``barrier_compensation_analysis``
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import xarray as xr


def _discover_iq_pairs(ds: xr.Dataset) -> List[Tuple[str, str]]:
    """Return matching (I, Q) variable names from a dataset."""
    if "I" in ds.data_vars and "Q" in ds.data_vars:
        return [("I", "Q")]

    pairs: List[Tuple[str, str]] = []
    for var_name in ds.data_vars:
        match = re.fullmatch(r"I(\d+)", var_name)
        if match is None:
            continue
        q_name = f"Q{match.group(1)}"
        if q_name in ds.data_vars:
            pairs.append((var_name, q_name))
    return sorted(pairs)


def process_raw_dataset(ds_raw: xr.Dataset, node) -> xr.Dataset:
    """Convert raw I/Q data to amplitude and phase.

    Parameters
    ----------
    ds_raw : xr.Dataset
        Raw dataset with ``I``/``Q``-like variables.
    node : QualibrationNode
        The active calibration node (unused but kept for API compatibility).

    Returns
    -------
    xr.Dataset
        Dataset augmented with ``amplitude`` and ``phase`` variables.
    """
    del node  # API compatibility: keep ``node`` in signature.
    ds = ds_raw.copy(deep=False)

    if "amplitude" in ds.data_vars and "phase" in ds.data_vars:
        return ds

    iq_pairs = _discover_iq_pairs(ds)
    if not iq_pairs:
        return ds

    if len(iq_pairs) == 1:
        i_name, q_name = iq_pairs[0]
        ds["amplitude"] = np.hypot(ds[i_name], ds[q_name])
        ds["phase"] = xr.apply_ufunc(np.arctan2, ds[q_name], ds[i_name])
        return ds

    # Multi-sensor fallback: keep per-sensor amplitude/phase datasets.
    for idx, (i_name, q_name) in enumerate(iq_pairs):
        ds[f"amplitude_{idx}"] = np.hypot(ds[i_name], ds[q_name])
        ds[f"phase_{idx}"] = xr.apply_ufunc(np.arctan2, ds[q_name], ds[i_name])

    # Preserve backward compatibility by exposing first pair under generic keys.
    first_i, first_q = iq_pairs[0]
    ds["amplitude"] = np.hypot(ds[first_i], ds[first_q])
    ds["phase"] = xr.apply_ufunc(np.arctan2, ds[first_q], ds[first_i])
    return ds


def _resolve_virtual_gate_set(node, virtual_gate_set_id: str | None = None):
    """Resolve the active VirtualGateSet from a node and optional explicit id."""
    requested_id = virtual_gate_set_id or getattr(node.parameters, "virtual_gate_set_id", None)
    virtual_gate_sets = getattr(node.machine, "virtual_gate_sets", {})
    if requested_id:
        if requested_id not in virtual_gate_sets:
            raise KeyError(
                f"VirtualGateSet '{requested_id}' not found. " f"Available: {list(virtual_gate_sets.keys())}"
            )
        return virtual_gate_sets[requested_id]

    if len(virtual_gate_sets) == 1:
        return next(iter(virtual_gate_sets.values()))

    raise ValueError("virtual_gate_set_id is required when multiple VirtualGateSets exist.")


def _resolve_layer(vgs, layer_id: str | None = None):
    """Resolve a virtualization layer by id, defaulting to the last layer."""
    if not vgs.layers:
        raise ValueError(f"VirtualGateSet '{vgs.id}' has no virtualization layers.")

    if layer_id is None:
        return vgs.layers[-1]

    for layer in vgs.layers:
        if layer.id == layer_id:
            return layer
    raise KeyError(
        f"Layer '{layer_id}' not found in VirtualGateSet '{vgs.id}'. "
        f"Available layers: {[layer.id for layer in vgs.layers]}"
    )


def _indices_from_gate_names(
    source_gates: Sequence[str],
    target_gates: Sequence[str],
    row_names: Iterable[str],
    col_names: Iterable[str],
) -> Tuple[List[int], List[int]]:
    """Convert row/column gate names into matrix indices."""
    row_indices = []
    for row in row_names:
        if row not in source_gates:
            raise KeyError(f"Row gate '{row}' not present in layer source_gates: {list(source_gates)}")
        row_indices.append(source_gates.index(row))

    col_indices = []
    for col in col_names:
        if col not in target_gates:
            raise KeyError(f"Column gate '{col}' not present in layer target_gates: {list(target_gates)}")
        col_indices.append(target_gates.index(col))

    return row_indices, col_indices


def update_compensation_submatrix(
    node,
    row_names: Sequence[str],
    col_names: Sequence[str],
    values: np.ndarray,
    layer_id: str | None = None,
) -> Dict[str, object]:
    """Write a dense submatrix into the selected virtual-gate layer.

    Parameters
    ----------
    node : QualibrationNode
        Active node containing ``machine`` and ``parameters``.
    row_names : sequence of str
        Row gate names in the layer ``source_gates`` basis.
    col_names : sequence of str
        Column gate names in the layer ``target_gates`` basis.
    values : np.ndarray
        Matrix values with shape ``(len(row_names), len(col_names))``.
    layer_id : str, optional
        Explicit layer id. If omitted, the last layer is used.

    Returns
    -------
    dict
        Metadata about the matrix update.
    """
    arr = np.asarray(values, dtype=float)
    expected_shape = (len(row_names), len(col_names))
    if arr.shape != expected_shape:
        raise ValueError(f"Submatrix shape mismatch. Expected {expected_shape}, got {arr.shape}.")

    vgs = _resolve_virtual_gate_set(node)
    active_layer_id = layer_id or getattr(node.parameters, "matrix_layer_id", None)
    layer = _resolve_layer(vgs, active_layer_id)

    matrix = np.asarray(layer.matrix, dtype=float)
    row_idx, col_idx = _indices_from_gate_names(
        list(layer.source_gates),
        list(layer.target_gates),
        row_names,
        col_names,
    )
    for r_local, r_global in enumerate(row_idx):
        for c_local, c_global in enumerate(col_idx):
            matrix[r_global, c_global] = arr[r_local, c_local]

    # Persist back to the mutable layer object used by QuAM/Qualibrate.
    layer.matrix = matrix.tolist()
    return {
        "virtual_gate_set_id": vgs.id,
        "layer_id": layer.id,
        "row_names": list(row_names),
        "col_names": list(col_names),
        "shape": matrix.shape,
    }


def update_compensation_matrix(
    node,
    row_name: str,
    col_name: str,
    coefficient: float,
) -> None:
    """Write a single coefficient into the virtual gate compensation matrix.

    Parameters
    ----------
    node : QualibrationNode
        The active calibration node (provides ``node.machine``).
    row_name : str
        Virtual gate name for the matrix row.
    col_name : str
        Virtual gate name for the matrix column.
    coefficient : float
        The cross-talk compensation coefficient to set.
    """
    update_compensation_submatrix(
        node=node,
        row_names=[row_name],
        col_names=[col_name],
        values=np.array([[float(coefficient)]], dtype=float),
        layer_id=getattr(node.parameters, "matrix_layer_id", None),
    )
