import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis import oscillation

from quam.components.quantum_components import qubit


@dataclass
class FitParameters:
    """Stores the relevant coupler zero point experiment fit parameters for a single qubit pair"""

    success: bool
    optimal_coupler_flux: float
    optimal_qubit_flux: float


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit xarray Dataset.

    Parameters:
    -----------
    fit_results : dict
        A dictionary containing the fitted results for each qubit pair.
    log_callable : callable, optional
        Callable for logging the fitted results. If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for qp_name, fit_result in fit_results.items():
        success = fit_result.get("success", False)
        optimal_coupler_flux = fit_result.get("optimal_coupler_flux", np.nan)
        optimal_qubit_flux = fit_result.get("optimal_qubit_flux", np.nan)

        s_qubit = f"Results for qubit pair {qp_name}: "
        s_qubit += "SUCCESS!\n" if success else "FAIL!\n"

        if not np.isnan(optimal_coupler_flux):
            s_qubit += f"\tOptimal coupler flux: {optimal_coupler_flux:.6f} V\n"
        else:
            s_qubit += "\tOptimal coupler flux: N/A\n"

        if not np.isnan(optimal_qubit_flux):
            s_qubit += f"\tOptimal qubit flux: {optimal_qubit_flux:.6f} V"
        else:
            s_qubit += "\tOptimal qubit flux: N/A"

        log_callable(s_qubit)


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
    xr.Dataset
        Dataset with additional coordinates.
    """
    detuning_mode = "quadratic"  # "cosine" or "quadratic"
    qubit_pairs = [node.machine.qubit_pairs[pair] for pair in node.parameters.qubit_pairs]
    fluxes_qp = node.namespace["fluxes_qp"]
    fluxes_coupler = ds.coupler_flux.values
    qubit_flux_full = np.array([fluxes_qp[qp.name] for qp in qubit_pairs])
    ds = ds.assign_coords({"qubit_flux_full": (["qubit_pair", "qubit_flux"], qubit_flux_full)})

    coupler_flux_full = np.array([fluxes_coupler + qp.coupler.decouple_offset for qp in qubit_pairs])
    if detuning_mode == "quadratic":
        detuning = np.array(
            [-fluxes_qp[qp.name] ** 2 * qp.qubit_control.freq_vs_flux_01_quad_term for qp in qubit_pairs]
        )
    elif detuning_mode == "cosine":
        detuning = np.array(
            [
                oscillation(
                    fluxes_qp,
                    qp.qubit_control.extras["a"],
                    qp.qubit_control.extras["f"],
                    qp.qubit_control.extras["phi"],
                    qp.qubit_control.extras["offset"],
                )
                for qp in qubit_pairs
            ]
        )
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
    node : QualibrationNode
        The calibration node.

    Returns:
    --------
    Tuple[xr.Dataset, dict[str, FitParameters]]
        A tuple containing the dataset with fit results and a dictionary of fit results for each qubit pair.
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
        if node.parameters.cz_or_iswap == "iswap":
            coupler_optimal_flux_point = res_sum.sel(qubit_pair=qp).mean(dim="qubit_flux").argmin()
        else:
            coupler_optimal_flux_point = res_sum.sel(qubit_pair=qp).mean(dim="qubit_flux").argmax()
        optimal_coupler_flux = fit.coupler_flux_full.sel(qubit_pair=qp)[coupler_optimal_flux_point]
        if node.parameters.cz_or_iswap == "iswap":
            qubit_optimal_flux_point = res_sum.sel(qubit_pair=qp).mean(dim="coupler_flux").argmax()
        else:
            qubit_optimal_flux_point = res_sum.sel(qubit_pair=qp).mean(dim="coupler_flux").argmin()
        optimal_qubit_flux = fluxes_qp[qp][qubit_optimal_flux_point]

        fit_results[qp] = FitParameters(
            success=True,
            optimal_coupler_flux=float(optimal_coupler_flux.values),
            optimal_qubit_flux=float(optimal_qubit_flux),
        )
    return fit, fit_results
