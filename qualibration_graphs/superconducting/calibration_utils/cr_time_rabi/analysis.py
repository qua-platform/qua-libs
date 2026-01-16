import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Literal
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from ..cr_utils import *
from calibration_utils.data_process_utils import reshape_control_target_val2dim 


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

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
    pass


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if node.parameters.use_state_discrimination:
        ds = reshape_control_target_val2dim(ds, state_discrimination=node.parameters.use_state_discrimination)
    else:
        ds = reshape_control_target_val2dim(ds, state_discrimination=node.parameters.use_state_discrimination)
        ds = convert_IQ_to_V(ds, qubits=None, qubit_pairs=node.namespace["qubit_pairs"])
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

    ds_fit = ds.assign({
        col.replace("state", "bloch"): -2 * ds[col] + 1
        for col in ds.data_vars
        if "state" in col
    })
    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)

    if node.parameters.use_state_discrimination:
        for qp in node.namespace["qubit_pairs"]:
            # Perform CR Hamiltonian tomography
            print("-" * 40)
            print(f"fitting for {qp.name}")
            fit_data_qp = fit_data.sel(qubit_pair=qp.name, control_target="t")
            crht = CRHamiltonianTomographyAnalysis(
                ts=fit_data_qp.pulse_duration.data,
                data=fit_data_qp["bloch"].data,  # target data: len(cr_drive_phases) x len(t_vec_cycle) x 3 x 2
            )
            try:
                crht.fit_params()
                fig_analysis = crht.plot_fit_result(do_show=False)
                node.results[f"figure_analysis_{qp.name}"] = fig_analysis
            except:
                print(f"-> failed")
                crht.interaction_coeffs_MHz = {p: None for p in PAULI_2Q}
        
            node.results[f"interaction_coefficients_{qp.name}"] = crht.interaction_coeffs_MHz

    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""

    # Populate the FitParameters class with fitted values
    fit_results = {
        qp: FitParameters(
            success=True,
        )
        for qp in fit.qubit_pair.values
    }
    return fit, fit_results
