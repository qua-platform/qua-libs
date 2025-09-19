import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis.fitting import fit_oscillation_decay_exp, oscillation_decay_exp
from qualibration_libs.data import convert_IQ_to_V
from scipy.optimize import curve_fit

from quam.components.quantum_components import qubit
from qualibration_libs.analysis import oscillation



@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    success: bool
    flux_coupler_min: float
    flux_qubit_max: float

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
    pass


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """
    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the processed data.
    node : QualibrationNode
        The calibration node containing parameters and qubit pairs.

    Returns:
    --------
    Tuple[xr.Dataset, Dict[str, FitResults]]
        Dataset with fit results and dictionary of fit results for each qubit pair.
    """
    detuning_mode = "quadratic" # "cosine" or "quadratic"
    qubit_pairs = [node.machine.qubit_pairs[pair] for pair in node.parameters.qubit_pairs]
    fluxes_qp = node.namespace["fluxes_qp"]
    fluxes_coupler = ds.coupler_flux.values
    flux_qubit_full = np.array([fluxes_qp[qp.name] for qp in qubit_pairs])
    ds = ds.assign_coords({"flux_qubit_full": (["qubit", "flux_qubit"], flux_qubit_full)})

    flux_coupler_full = np.array([fluxes_coupler + qp.coupler.decouple_offset for qp in qubit_pairs])
    if detuning_mode == "quadratic":
        detuning = np.array([-fluxes_qp[qp.name] ** 2 * qp.qubit_control.freq_vs_flux_01_quad_term  for qp in qubit_pairs])
    elif detuning_mode == "cosine":
        detuning = np.array([oscillation(fluxes_qp, qp.qubit_control.extras['a'], qp.qubit_control.extras['f'], qp.qubit_control.extras['phi'], qp.qubit_control.extras['offset']) for qp in qubit_pairs])
    ds = ds.assign_coords({"flux_coupler_full": (["qubit_pair", "flux_coupler"], flux_coupler_full)})
    ds = ds.assign_coords({"detuning": (["qubit_pair", "flux_qubit"], detuning)})

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

    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""

    # Populate the FitParameters class with fitted values
    fit_results = {}
    for qp in fit.qubit_pair.values:
        if node.parameters.use_state_discrimination:
            res_sum = -fit.state_control + fit.state_target
        else:
            res_sum = -fit.I_control + fit.I_target
        fluxes_qp = node.namespace["fluxes_qp"]
        coupler_min_arg = res_sum.sel(qubit_pair=qp).mean(dim='qubit_flux').argmin()
        flux_coupler_min = fit.flux_coupler_full.sel(qubit_pair=qp)[coupler_min_arg]
        qubit_max_arg = res_sum.sel(qubit_pair=qp).mean(dim="coupler_flux").argmax()
        flux_qubit_max = fluxes_qp[qp][qubit_max_arg]
        #fit_results[qp] = {"flux_coupler_min": float(flux_coupler_min.values), "flux_qubit_max": float(flux_qubit_max)}
        fit_results[qp] = FitParameters(
            success=True,
            flux_coupler_min=float(flux_coupler_min.values),
            flux_qubit_max=float(flux_qubit_max)
        )
    return fit, fit_results
