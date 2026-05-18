"""Analysis utilities for XY-Coupler delay calibration."""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode

__all__ = [
    "FitParameters",
    "process_raw_dataset",
    "fit_raw_data",
    "log_fitted_results",
]


@dataclass
class FitParameters:
    """Stores XY-Z delay fit parameters: success status and extracted flux delay."""

    success: bool
    flux_delay: int


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        success = res["success"] if isinstance(res, dict) else res.success
        flux_delay = res["flux_delay"] if isinstance(res, dict) else res.flux_delay
        status = "SUCCESS" if success else "FAIL"
        log_callable(f"Qubit {q}: {status} | flux_delay = {flux_delay} ns")


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Convert IQ data to voltage and compute difference between |e> and |g> states."""
    if not node.parameters.use_state_discrimination:
        # Build readout lengths keyed by the coupler name so the coordinate
        # matches the 'qubit' dimension already in ds (which uses coupler names).
        measured_qubits = node.namespace["measured_qubits"]
        qubit_pairs = node.namespace["qubit_pairs"]
        readout_lengths = xr.DataArray(
            [q.resonator.operations["readout"].length for q in measured_qubits],
            coords=[("qubit", [qp.name for qp in qubit_pairs])],
        )
        ds = ds.assign({key: ds[key] * 2**12 / readout_lengths for key in ("I", "Q")})

    data = "state" if hasattr(ds, "state") else "I"

    difference = ds[data].sel(init_state="e") - ds[data].sel(init_state="g")

    if data == "I":
        difference -= difference.mean()

    ds["difference"] = difference
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the data to estimate a delay between XY and Z.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node_parameters : Parameters
        Parameters related to the node, including whether state discrimination is used.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    """

    dfit = ds.groupby("qubit").apply(fit_routine, node=node)

    fit_results = _extract_relevant_fit_parameters(dfit, node)

    return dfit, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    fit_results = {}
    for q in fit.qubit.data:
        fit_results[q] = FitParameters(
            success=fit.success.sel(qubit=q).data, flux_delay=fit.flux_delay.sel(qubit=q).data
        )
    return fit_results


def fit_routine(da, node):
    """Extract flux delay by finding zero-crossing midpoint in the difference signal."""
    x = da.relative_time.data
    y = da.difference.data[0]

    try:
        sign_changes = np.sign(y)
        crossings = np.where(np.diff(sign_changes))[0]
        assert len(crossings) == 2, "Expected exactly two sign change points in the data."
        flux_delay = x[(crossings[1] + crossings[0]) // 2]
        da = da.assign(flux_delay=flux_delay)
        da = da.assign(success=True)

    except AssertionError as e:
        print(f"Error processing {da.qubit.data}: {e}")
        flux_delay = 0
        da = da.assign(flux_delay=flux_delay)
        da = da.assign(success=False)
        return da

    print(f"Flux delay for {da.qubit.data}: {flux_delay:.2f} ns")

    return da
