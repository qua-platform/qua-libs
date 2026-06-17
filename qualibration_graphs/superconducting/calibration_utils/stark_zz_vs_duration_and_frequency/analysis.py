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
    zz_coeff: float
    frequency_shift_target_qc0: float
    frequency_shift_target_qc1: float
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
        s_detuning = f"\tmax zz_coeff at detuning={fit_results[qp]['best_detuning']}\n"
        s_zz_coeff = f"\tExtracted ZZ coefficient: {1e-6 * fit_results[qp]['zz_coeff']:.3f} MHz\n"
        s_frequency_shift = (
            f"\t\tFitted frequency shift of target qubit with qc=|0>: {1e-6 * fit_results[qp]['frequency_shift_target_qc0']:.3f} MHz\n"
            + f"\t\tFitted frequency shift of target qubit with qc=|1>: {1e-6 * fit_results[qp]['frequency_shift_target_qc1']:.3f} MHz\n"
        )
        if fit_results[qp]["success"]:
            s_qubit_pair += " SUCCESS!\n"
        else:
            s_qubit_pair += " FAIL!\n"
        log_callable(s_qubit_pair + s_detuning + s_zz_coeff + s_frequency_shift)


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
    if node.parameters.use_state_discrimination:
        fit = fit_oscillation_decay_exp(ds.state, "idle_time")
    else:
        fit = fit_oscillation_decay_exp(ds.I, "idle_time")

    ds_fit = xr.merge([ds, fit.rename("fit")])

    ds_fit, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return ds_fit, fit_results


def _extract_relevant_fit_parameters(ds_fit: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Post-process modified-echo fit results:
      - frequency (oscillation due to second-interval detuning)
      - T2_modified_echo = -1/decay
      - Error propagation from 'decay_decay' as in your echo code
    """
    # --- Frequency of oscillation (DataArray)
    # NOTE: Your other pipeline labels frequency as "MHz" but converts with 1e9 to Hz.
    fitted_frequency = ds_fit.fit.sel(fit_vals="f")
    fitted_frequency.attrs = {"long_name": "fitted fitted_frequency", "units": "MHz"}
    # fitted_frequency = fitted_frequency.where(fitted_frequency > 0, drop=True)

    # --- Decay (envelope) and its residual metric
    decay = ds_fit.fit.sel(fit_vals="decay")
    decay.attrs = {"long_name": "envelope decay coefficient", "units": "1/ns"}  # sign is typically negative
    decay_res = ds_fit.fit.sel(fit_vals="decay_decay")
    decay_res.attrs = {"long_name": "decay residual (variance-like)", "units": "1/ns^2"}

    # T2_modified_echo-like time: match your standard echo convention (T2_modified_echo = -1/decay)
    T2_modified_echo = -1.0 / decay
    T2_modified_echo.attrs = {"long_name": "T2 echo (modified for CZ)", "units": "ns"}

    # Propagate uncertainty similar to your echo code:
    # T2_modified_echo_error = -T2 * (sqrt(decay_res) / decay)
    T2_modified_echo_error = -T2_modified_echo * (np.sqrt(decay_res) / decay)
    T2_modified_echo_error.attrs = {"long_name": "T2 echo (modified for CZ) error", "units": "ns"}

    zz_coeff = fitted_frequency.sel(control_target="t", control_state=0) - fitted_frequency.sel(
        control_target="t", control_state=1
    )

    # Best zz drive detuning per qubit_pair
    best_detuning = zz_coeff.idxmax(dim="detuning").rename("best_detuning")
    best_zz_coeff = zz_coeff.max(dim="detuning").rename("best_zz_coeff")

    # Attach to dataset for convenience/plotting
    ds_fit = ds_fit.assign(
        {
            "T2_modified_echo": T2_modified_echo,
            "T2_modified_echo_error": T2_modified_echo_error,
            "freq_shift": fitted_frequency,
            "zz_coeff": zz_coeff,
            "best_detuning": best_detuning,
            "best_zz_coeff": best_zz_coeff,
        }
    )

    # --- Success mask (finite values for both frequency and T2)
    nan_success = (
        xr.ufuncs.isnan(fitted_frequency) | xr.ufuncs.isnan(T2_modified_echo) | xr.ufuncs.isnan(T2_modified_echo_error)
    )
    success_criteria = ~nan_success
    ds_fit = ds_fit.assign({"success": success_criteria})

    # --- Package into FitParameters (align with your previous keys)
    # Reuse 'freq_shift' to carry the frequency shift (in Hz) for compatibility with Ramsey output
    def _sel_best(da, qp, *, control_state):
        detuning = best_detuning.sel(qubit_pair=qp).item()
        return (
            da.sel(qubit_pair=qp, control_target="t", control_state=control_state)
            .sel(detuning=detuning, method="nearest")
            .squeeze(drop=True)
            .item()
        )

    fit_results: Dict[str, FitParameters] = {
        qp: FitParameters(
            best_detuning=best_detuning.sel(qubit_pair=qp).item(),
            zz_coeff=1e9 * best_zz_coeff.sel(qubit_pair=qp).item(),
            frequency_shift_target_qc0=1e9 * _sel_best(fitted_frequency, qp, control_state=0),
            frequency_shift_target_qc1=1e9 * _sel_best(fitted_frequency, qp, control_state=1),
            # qc0 should use control_state=0
            T2_modified_echo_target_qc0=1e-9 * _sel_best(T2_modified_echo, qp, control_state=0),
            T2_modified_echo_error_target_qc0=1e-9 * _sel_best(T2_modified_echo_error, qp, control_state=0),
            # qc1 should use control_state=1
            T2_modified_echo_target_qc1=1e-9 * _sel_best(T2_modified_echo, qp, control_state=1),
            T2_modified_echo_error_target_qc1=1e-9 * _sel_best(T2_modified_echo_error, qp, control_state=1),
            success=bool(
                success_criteria.sel(qubit_pair=qp, control_target="t")
                .sel(detuning=best_detuning.sel(qubit_pair=qp).item(), method="nearest")
                .all(dim="control_state")
                .item()
            ),
        )
        # iterate over the object that actually has the coord (adjust as needed)
        for qp in fitted_frequency.qubit_pair.values
    }

    return ds_fit, fit_results
