import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.qua_datasets import convert_IQ_to_V
from quam_experiments.analysis.fit import fit_decay_exp


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    T2_echo: float
    T2_echo_error: float
    success: bool
    qubit_name: Optional[str] = ""


def log_fitted_results(fit_results: Dict, logger=None):
    """
    Logs the node-specific fitted results for all qubits from the fit xarray Dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the fitted results for all qubits.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    # for q in fit_results.keys():
    #     s_qubit = f"Results for qubit {q}: "
    #     s_freq = f"\tQubit frequency: {1e-9 * fit_results[q]['frequency']:.3f} GHz | "
    #     s_fwhm = f"FWHM: {1e-3 * fit_results[q]['fwhm']:.1f} kHz | "
    #     s_angle = (
    #         f"The integration weight angle: {fit_results[q]['iw_angle']:.3f} rad\n "
    #     )
    #     s_saturation = f"To get the desired FWHM, the saturation amplitude is updated to: {1e3 * fit_results[q]['saturation_amp']:.1f} mV | "
    #     s_x180 = f"To get the desired x180 gate, the x180 amplitude is updated to: {1e3 * fit_results[q]['x180_amp']:.1f} mV\n "
    #     if fit_results[q]["success"]:
    #         s_qubit += " SUCCESS!\n"
    #     else:
    #         s_qubit += " FAIL!\n"
    #     logger.info(
    #         s_qubit + s_freq + s_fwhm + s_freq + s_angle + s_saturation + s_x180
    #     )
    pass


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the qubit frequency and FWHM for each qubit in the dataset.

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
    # # Fit the exponential decay
    if node.parameters.use_state_discrimination:
        fit_data = fit_decay_exp(ds_fit.state, "idle_time")
    else:
        fit_data = fit_decay_exp(ds_fit.I, "idle_time")
    ds_fit = xr.merge([ds, fit_data.rename("fit_data")])

    ds_fit, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return ds_fit, fit_results


def _extract_relevant_fit_parameters(ds_fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    # Decay rate and its uncertainty
    decay = ds_fit.fit_data.sel(fit_vals="decay")
    decay_res = ds_fit.fit_data.sel(fit_vals="decay_decay")
    # T2 echo and its uncertainty
    ds_fit["T2_echo"] = -1 / ds_fit.fit_data.sel(fit_vals="decay")
    ds_fit["T2_echo"].attrs = {"long_name": "T2 echo", "units": "ns"}
    ds_fit["T2_echo_error"] = -ds_fit["T2_echo"] * (np.sqrt(decay_res) / decay)
    ds_fit["T2_echo_error"].attrs = {"long_name": "T2 echo error", "units": "ns"}
    # Assess whether the fit was successful or not
    nan_success = np.isnan(ds_fit["T2_echo"]) | np.isnan(ds_fit["T2_echo_error"])
    success_criteria = ~nan_success
    ds_fit = ds_fit.assign({"success": success_criteria})
    # Populate the FitParameters class with fitted values
    fit_results = {
        q: FitParameters(
            T2_echo=1e-9 * float(ds_fit["T2_echo"].sel(qubit=q)),
            T2_echo_error=1e-9 * float(ds_fit["T2_echo_error"].sel(qubit=q)),
            success=bool(ds_fit["success"].sel(qubit=q)),
        )
        for q in ds_fit.qubit.values
    }
    return ds_fit, fit_results
