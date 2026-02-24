"""Shared analysis helpers for gate virtualization calibration nodes.

Contains common operations (I/Q processing, matrix updates) used by all
three calibration nodes.  Node-specific analysis lives in the dedicated
modules:
- ``sensor_compensation_analysis``
- ``virtual_plunger_analysis``
- ``barrier_compensation_analysis``
"""

from typing import Dict

import numpy as np
import xarray as xr


def process_raw_dataset(ds_raw: xr.Dataset, node) -> xr.Dataset:
    """Convert raw I/Q data to amplitude and phase.

    Parameters
    ----------
    ds_raw : xr.Dataset
        Raw dataset with ``I`` and ``Q`` data variables.
    node : QualibrationNode
        The active calibration node.

    Returns
    -------
    xr.Dataset
        Dataset augmented with ``amplitude`` and ``phase`` variables.
    """
    # TODO: implement ADC-to-volts conversion + amplitude/phase computation
    pass


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
    # TODO: implement matrix update via node.machine.virtual_gate_sets[...].compensation_matrix
    pass
