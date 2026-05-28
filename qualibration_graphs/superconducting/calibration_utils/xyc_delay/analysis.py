"""Analysis utilities for XY-Coupler delay calibration."""

from dataclasses import dataclass
from typing import Tuple

import xarray as xr
from qualibrate import QualibrationNode
from calibration_utils.xyz_delay.analysis import (
    _extract_relevant_fit_parameters,
    fit_delay_trace,
    log_fitted_results,
)

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
    Fit the qubit XY-Coupler Z line delay response for each qubit and extract flux-delay parameters.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing per-qubit `difference` traces versus `relative_time`.
    node : QualibrationNode
        Node context used by the fitter (e.g., measured qubit objects and config).

    Returns:
    --------
    Tuple[xr.Dataset, dict[str, FitParameters]]
        `dfit`: dataset with fitted curve (`fit`) and extracted values
        (`flux_delay`, `flux_delay_std`, `success`) per qubit.
        `fit_results`: summarized fit outcomes keyed by qubit name.
    """

    dfit = ds.groupby("qubit").apply(fit_routine, node=node)

    fit_results = _extract_relevant_fit_parameters(dfit, node)

    return dfit, fit_results


def fit_routine(da, node):
    measured_qubit_name = da.measured_qubit_name.item()
    qubit = next(q for q in node.namespace["measured_qubits"] if q.id == measured_qubit_name)
    xy_duration = qubit.xy.operations["x180"].length  # ns
    return fit_delay_trace(da, xy_duration)
