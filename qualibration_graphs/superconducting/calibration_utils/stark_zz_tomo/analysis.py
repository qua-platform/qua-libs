import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Literal
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from qualibration_libs.analysis import fit_oscillation_decay_exp
from ..data_process_utils import reshape_control_target_val2dim


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    zz_coeff: float
    detuning_target_qc0: float
    detuning_target_qc1: float
    T2_modified_echo_target_qc0: float
    T2_modified_echo_error_target_qc0: float
    T2_modified_echo_target_qc1: float
    T2_modified_echo_error_target_qc1: float
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
        s_detuning = (
            f"\tFitted detuning of target qubit with qc=|0>: {1e-6 * fit_results[qp]['detuning_target_qc0']:.3f} MHz\n"
            + f"\tFitted detuning of target qubit with qc=|1>: {1e-6 * fit_results[qp]['detuning_target_qc1']:.3f} MHz\n"
        )
        s_zz_coeff = f"\tExtracted ZZ coefficient: {1e-6 * fit_results[qp]['zz_coeff']:.3f} MHz\n"
        if fit_results[qp]["success"]:
            s_qubit_pair += " SUCCESS!\n"
        else:
            s_qubit_pair += " FAIL!\n"
        log_callable(s_qubit_pair + s_detuning + s_zz_coeff)


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

    bloch_t_c0 = 1 - 2 * ds_fit.sel(control_target="t", control_state=0)
    bloch_t_c1 = 1 - 2 * ds_fit.sel(control_target="t", control_state=1)
    
    R = 0.5 * (bloch_t_c0 - bloch_t_c1) ** 2

    # Attach to dataset for convenience/plotting
    ds_fit = ds_fit.assign(
        {
            "T2_modified_echo": T2_modified_echo,
            "T2_modified_echo_error": T2_modified_echo_error,
            "freq_offset": fitted_frequency,
            "zz_coeff": zz_coeff,
        }
    )

    # --- Success mask (finite values for both frequency and T2)
    nan_success = (
        xr.ufuncs.isnan(fitted_frequency) | xr.ufuncs.isnan(T2_modified_echo) | xr.ufuncs.isnan(T2_modified_echo_error)
    )
    success_criteria = ~nan_success
    ds_fit = ds_fit.assign({"success": success_criteria})

    # --- Package into FitParameters (align with your previous keys)
    # Reuse 'freq_offset' to carry the frequency shift (in Hz) for compatibility with Ramsey output
    fit_results: Dict[str, FitParameters] = {
        qp: FitParameters(
            # frequency from fit ('f') — consistent with your Ramsey code using 1e9 multiplier
            zz_coeff=1e9 * float(zz_coeff.sel(qubit_pair=qp)),
            detuning_target_qc0=1e9 * float(fitted_frequency.sel(qubit_pair=qp, control_target="t", control_state=0)),
            detuning_target_qc1=1e9 * float(fitted_frequency.sel(qubit_pair=qp, control_target="t", control_state=1)),
            # also provide a T2_modified_echo-style time for the modified echo (convert to seconds)
            T2_modified_echo_target_qc0=1e-9
            * float(T2_modified_echo.sel(qubit_pair=qp, control_target="t", control_state=0)),
            T2_modified_echo_error_target_qc0=1e-9
            * float(T2_modified_echo_error.sel(qubit_pair=qp, control_target="t", control_state=0)),
            T2_modified_echo_target_qc1=1e-9
            * float(T2_modified_echo.sel(qubit_pair=qp, control_target="t", control_state=0)),
            T2_modified_echo_error_target_qc1=1e-9
            * float(T2_modified_echo_error.sel(qubit_pair=qp, control_target="t", control_state=0)),
            success=bool(success_criteria.sel(qubit_pair=qp, control_target="t", control_state=1)),
        )
        for qp in ds_fit.qubit_pair.values
    }

    return ds_fit, fit_results
