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

    best_relative_phase: float
    best_state_c0: float
    best_state_c1: float
    residual_mse: float

    success: bool


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

    for qp in fit_results.keys():
        s_qubit_pair = f"Results for qubit pair {qp}: "
        s_rel_phase = f"\toptimal relative_phase={fit_results[qp]['best_relative_phase']}\n"
        s_state_c0 = f"\t\texpectation of Qt state with qc=|0>: {fit_results[qp]['best_state_c0']:.3f}\n"
        s_state_c1 = f"\t\texpectation of Qt state with qc=|1>: {fit_results[qp]['best_state_c1']:.3f}\n"
        s_state_mse = f"\t\tresidual MSE: {fit_results[qp]['residual_mse']:.3f}\n"
        if fit_results[qp]["success"]:
            s_qubit_pair += " SUCCESS!\n"
        else:
            s_qubit_pair += " FAIL!\n"
        log_callable(s_qubit_pair\
            + s_rel_phase + s_state_c0 + s_state_c1 + s_state_mse
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    try:
        if node.parameters.use_state_discrimination:
            ds = reshape_control_target_val2dim(
                ds, state_discrimination=node.parameters.use_state_discrimination
            )
        else:
            ds = reshape_control_target_val2dim(
                ds, state_discrimination=node.parameters.use_state_discrimination
            )
            ds = convert_IQ_to_V(
                ds, qubits=None, qubit_pairs=node.namespace["qubit_pairs"]
            )
    except KeyError:
        pass
    return ds


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
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
    data_t = ds_fit.sel(control_target="t")

    def find_optimal_relative_phase(ds):
        # Target values per control_state: 0 -> 0, 1 -> 1
        target = xr.DataArray(ds.control_state, dims=["control_state"]).astype(ds.state.dtype)

        # Squared error to targets at each (qubit_pair, relative_phase, control_state)
        err = (ds.state - target) ** 2

        # Sum over control_state -> cost per (qubit_pair, relative_phase)
        cost = err.sum(dim="control_state")

        # Argmin index along relative_phase for each qubit_pair
        idx = cost.argmin(dim="relative_phase")  # dims: (qubit_pair,)

        # 1) The best relative_phase (label), per qubit_pair
        best_relative_phase = ds.relative_phase.isel(relative_phase=idx).rename("best_relative_phase")

        # 2) The states at that best phase (for both control_state=0 and 1)
        best_states = ds.state.isel(relative_phase=idx).rename("state_at_best")
        # dims: (qubit_pair, control_state)

        # 3) The residual (sum of squared errors) at the best phase
        residual = cost.isel(relative_phase=idx).rename("residual_mse")
        
        return best_relative_phase, best_states, residual
    
    best_relative_phase_t, best_states_t, residual_t = find_optimal_relative_phase(data_t)

    # Optional: attach to your dataset
    ds_fit = ds_fit.assign(
        {            
            "best_relative_phase": best_relative_phase_t,
            "best_state": best_states_t,
            "residual_mse": residual_t,
        },
    )

    fit_results: Dict[str, FitParameters] = {
        qp: FitParameters(
            best_relative_phase=best_relative_phase_t.sel(qubit_pair=qp).item(),
            best_state_c0=best_states_t.sel(qubit_pair=qp, control_state=0).item(),
            best_state_c1=best_states_t.sel(qubit_pair=qp, control_state=1).item(),
            residual_mse=residual_t.sel(qubit_pair=qp).item(),
            success=True,
        )
        for qp in ds_fit.qubit_pair.values
    }

    return ds_fit, fit_results
