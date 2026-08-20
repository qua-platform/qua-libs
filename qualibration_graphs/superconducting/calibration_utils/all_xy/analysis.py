"""Analysis utilities for All-XY calibration.

All-XY does not update machine state. Success is decided by how closely the
measured staircase matches the ideal ground / superposition / excited pattern.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V

from .sequences import IDEAL_ALL_XY


@dataclass
class FitParameters:
    """All-XY outcome vs ideal staircase."""

    success: bool
    rms_error: float


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log per-qubit RMS error against the ideal All-XY pattern."""
    if log_callable is None:
        return
    for q, result in fit_results.items():
        success = result["success"] if isinstance(result, dict) else result.success
        rms_error = result["rms_error"] if isinstance(result, dict) else result.rms_error
        status = "successful" if success else "failed"
        log_callable(f"All-XY qubit {q}: RMS={rms_error:.4f} --> {status}")


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert I/Q to volts and add IQ amplitude/phase when not using state discrimination."""
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
        ds = add_amplitude_and_phase(ds, "sequence_index")
    return ds


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Score each qubit against the ideal All-XY staircase (no pulse-parameter fit)."""
    threshold = node.parameters.rms_threshold
    fit_results: Dict[str, FitParameters] = {}
    rms_by_qubit = []
    success_by_qubit = []
    qubit_names = []

    data_var = "state" if "state" in ds.data_vars else "IQ_abs"
    for q in node.namespace["qubits"]:
        y = np.asarray(ds[data_var].sel(qubit=q.name).values, dtype=float).ravel()
        y_span = float(np.max(y) - np.min(y))
        rms_error = float("inf") if y_span == 0.0 else float(
            np.sqrt(np.mean(((y - np.min(y)) / y_span - IDEAL_ALL_XY) ** 2))
        )
        success = rms_error < threshold
        fit_results[q.name] = FitParameters(success=success, rms_error=rms_error)
        qubit_names.append(q.name)
        rms_by_qubit.append(rms_error)
        success_by_qubit.append(success)

    ds_fit = xr.Dataset(
        {
            "rms_error": ("qubit", rms_by_qubit),
            "success": ("qubit", success_by_qubit),
        },
        coords={"qubit": qubit_names},
    )
    return ds_fit, fit_results
