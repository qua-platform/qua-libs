"""Shared analysis helpers for gate virtualization calibration nodes.

Contains common operations (I/Q processing, matrix updates) used by all
three calibration nodes.  Node-specific analysis lives in the dedicated
modules:
- ``sensor_compensation_analysis``
- ``virtual_plunger_analysis``
- ``barrier_compensation_analysis``
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import xarray as xr


def process_raw_dataset(
    ds_raw: xr.Dataset,
    node: Optional[object] = None,
) -> xr.Dataset:
    """Convert raw I/Q data to amplitude and phase.

    Parameters
    ----------
    ds_raw : xr.Dataset
        Raw dataset with ``I`` and ``Q`` data variables.
    node : QualibrationNode, optional
        The active calibration node (reserved for future per-node scaling).

    Returns
    -------
    xr.Dataset
        Copy of *ds_raw* augmented with ``amplitude`` and ``phase`` variables.
    """
    ds = ds_raw.copy(deep=True)
    I = ds["I"].values  # noqa: E741
    Q = ds["Q"].values

    ds["amplitude"] = xr.DataArray(
        np.sqrt(I**2 + Q**2),
        dims=ds["I"].dims,
        coords=ds["I"].coords,
    )
    ds["phase"] = xr.DataArray(
        np.arctan2(Q, I),
        dims=ds["I"].dims,
        coords=ds["I"].coords,
    )
    return ds


def update_compensation_matrix(
    node: object,
    row_name: str,
    col_name: str,
    coefficient: float,
) -> None:
    """Write a single coefficient into the virtual gate compensation matrix.

    The compensation matrix lives on ``VirtualGateSet.layers[0]``
    (the "compensation_layer").  Rows and columns are indexed by the
    layer's ``source_gates`` list, which contains virtual gate names
    (e.g. ``"virtual_sensor_1"``, ``"virtual_dot_1"``).

    Parameters
    ----------
    node : QualibrationNode
        The active calibration node (provides ``node.machine``).
    row_name : str
        Virtual gate name for the matrix row (the gate whose operating
        point we want to stabilise, e.g. the sensor gate).
    col_name : str
        Virtual gate name for the matrix column (the gate whose
        movement causes the cross-talk, e.g. the device gate).
    coefficient : float
        The compensation coefficient to write.  Because the hardware
        applies M⁻¹ (not M) when resolving virtual→physical voltages,
        the entry should equal the fitted cross-talk slope ``alpha``
        directly (not negated).
    """
    machine = getattr(node, "machine", None)
    if machine is None:
        raise ValueError("Node does not have a machine attribute.")

    virtual_gate_sets = getattr(machine, "virtual_gate_sets", None)
    if virtual_gate_sets is None:
        raise ValueError("Machine does not have virtual_gate_sets.")

    for vgs_id, vgs in virtual_gate_sets.items():
        layers = getattr(vgs, "layers", None)
        if not layers:
            continue
        layer = layers[0]
        src = list(layer.source_gates)
        if row_name in src and col_name in src:
            row_idx = src.index(row_name)
            col_idx = src.index(col_name)
            layer.matrix[row_idx][col_idx] = coefficient
            return

    raise ValueError(
        f"No virtual gate set found containing both '{row_name}' and '{col_name}' "
        f"in layers[0].source_gates."
    )
