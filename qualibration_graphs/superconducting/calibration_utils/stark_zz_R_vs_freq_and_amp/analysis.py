import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Literal
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from qualibration_libs.analysis import fit_oscillation_decay_exp
from calibration_utils.data_process_utils import reshape_control_target_val2dim


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""
    best_detuning: float
    best_amp_scaling: float
    best_R: float
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit xarray Dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the fitted results for all qubits.
    log_callable : callable, optional
        Callable for logging the fitted results. If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for qp, params in fit_results.items():
        msg = (
            f"Results for qubit pair {qp}:\n"
            f"\tmax R = {params['best_R']:.4f}\n"
            f"\tat amp_scaling = {params['best_amp_scaling']:.3f}\n"
            f"\tat detuning    = {params['best_detuning']/1e6:.3f} MHz\n"
        )
        log_callable(msg)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if node.parameters.use_state_discrimination:
        ds = reshape_control_target_val2dim(ds, state_discrimination=node.parameters.use_state_discrimination)
    else:
        ds = reshape_control_target_val2dim(ds, state_discrimination=node.parameters.use_state_discrimination)
        ds = convert_IQ_to_V(ds, qubits=None, qubit_pairs=node.namespace["qubit_pairs"])
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the frequency detuning and T2 decay of the Ramsey oscillations for each qubit.

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
    ds_fit = ds
    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(ds_fit: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Post-process modified-echo fit results:
      - frequency (oscillation due to second-interval detuning)
      - T2_modified_echo = -1/decay
      - Error propagation from 'decay_decay' as in your echo code
    """
    val = "state" if "state" in ds_fit.data_vars else "I"

    # Work with DataArrays, not Datasets
    bloch_t_c0 = 1 - 2 * ds_fit[val].sel(control_target="t", control_state=0)
    bloch_t_c1 = 1 - 2 * ds_fit[val].sel(control_target="t", control_state=1)

    # R is a DataArray
    R = 0.5 * xr.apply_ufunc(
        np.sqrt,
        ((bloch_t_c0 - bloch_t_c1) ** 2).sum(dim="target_basis"),
    ).rename("R")

    # Argmax over stacked 2D grid -> imax is a DataArray
    R_stack = R.stack(search=("detuning", "amp_scaling"))
    imax = R_stack.argmax("search")

    # Pull coords at argmax (each is a DataArray over remaining dims, e.g. qubit_pair)
    best_detuning = R_stack["detuning"].isel(search=imax).rename("best_detuning")
    best_amp_scaling = R_stack["amp_scaling"].isel(search=imax).rename("best_amp_scaling")
    best_R = R_stack.isel(search=imax).rename("best_R")  # value at argmax

    # Attach to dataset for convenience/plotting
    ds_fit = ds_fit.assign(
        {
            "R": R,
            "best_detuning": best_detuning,
            "best_amp_scaling": best_amp_scaling,
            "best_R": best_R,
        }
    )
    ds_fit = ds_fit.drop_vars("search")

    # Reuse 'freq_offset' to carry the frequency shift (in Hz) for compatibility with Ramsey output
    fit_results: Dict[str, FitParameters] = {
        qp: FitParameters(
            best_detuning=best_detuning.sel(qubit_pair=qp).item(),
            best_amp_scaling=best_amp_scaling.sel(qubit_pair=qp).item(),
            best_R=best_R.sel(qubit_pair=qp).item(),
            success=True,
        )
        for qp in ds_fit.qubit_pair.values
    }

    return ds_fit, fit_results
