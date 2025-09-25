import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode

from quam.components.quantum_components import qubit
from qualibration_libs.analysis import oscillation, fit_oscillation
from qualibration_libs.legacy.lib.fit import fix_oscillation_phi_2pi



@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    success: bool
    coupler_flux_Cz: float
    qubit_flux_Cz: float

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
    qubit_flux_full = np.array([fluxes_qp[qp.name] for qp in qubit_pairs])
    ds = ds.assign_coords({"qubit_flux_full": (["qubit_pair", "qubit_flux"], qubit_flux_full)})

    coupler_flux_full = np.array([fluxes_coupler + qp.coupler.decouple_offset for qp in qubit_pairs])
    if detuning_mode == "quadratic":
        detuning = np.array([-fluxes_qp[qp.name] ** 2 * qp.qubit_control.freq_vs_flux_01_quad_term  for qp in qubit_pairs])
    elif detuning_mode == "cosine":
        detuning = np.array([oscillation(fluxes_qp, qp.qubit_control.extras['a'], qp.qubit_control.extras['f'], qp.qubit_control.extras['phi'], qp.qubit_control.extras['offset']) for qp in qubit_pairs])
    ds = ds.assign_coords({"coupler_flux_full": (["qubit_pair", "coupler_flux"], coupler_flux_full)})
    ds = ds.assign_coords({"detuning": (["qubit_pair", "qubit_flux"], detuning)})

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

    fit_data = fit_oscillation(fit.state_target, "frames")
    phase = fix_oscillation_phi_2pi(fit_data)
    phase_diff = phase.diff(dim="control_ax")
    leak = fit.state_control_f.mean(dim = "frames").sel(control_ax = 1)
    # (((phase_diff+0.5 )% 1 -0.5)*360).plot()

    mask = (np.abs((np.abs(phase_diff)-0.5))<0.02)
    leak_mask = leak * mask + (1 - mask)
    min_value = leak_mask.min(dim=["qubit_flux", "coupler_flux","control_ax"])
    min_coords = {}
    for qp in phase_diff.qubit_pair.values:
        min_coords[qp] = leak_mask.sel(qubit_pair=qp).where(leak_mask.sel(qubit_pair=qp) == min_value.sel(qubit_pair=qp), drop=True)[0][0]
        
        fit_results[qp] = FitParameters(
        success=True,
        coupler_flux_Cz=float(min_coords[qp].coupler_flux_full.values),
        qubit_flux_Cz=float(min_coords[qp].qubit_flux_full.values)
        )
    fit = xr.Dataset({'phase_diff':phase_diff,'leak_mask':leak_mask,'mask':mask})
    node.namespace['mini_coords']=min_coords
    return fit, fit_results
