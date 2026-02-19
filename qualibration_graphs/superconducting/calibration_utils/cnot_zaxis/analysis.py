import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Literal
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from calibration_utils.data_process_utils import reshape_control_target_val2dim


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    fidelity: float
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
        ds = reshape_control_target_val2dim(
            ds, state_discrimination=node.parameters.use_state_discrimination
        )
    else:
        ds = reshape_control_target_val2dim(
            ds, state_discrimination=node.parameters.use_state_discrimination
        )
        ds = convert_IQ_to_V(ds, qubits=None, qubit_pairs=node.namespace["qubit_pairs"])
    return ds


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
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
    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(ds_fit: xr.Dataset, node: QualibrationNode):
    """Add metadata (counts, density matrix, fidelity) and build fit_results."""

    # Map integer 0..3 to basis labels
    basis_labels = ["00", "01", "10", "11"]

    # Compute joint outcomes = 2*c + t
    joint = (
        2 * ds_fit.sel(control_target="c").state + ds_fit.sel(control_target="t").state
    )
    # joint dims: (qubit_pair, n_shots, correction_phases_2pi, initial_state)

    # One-hot encode outcomes [0,1,2,3] along a new "measured_state" dim
    counts = xr.apply_ufunc(
        lambda arr: np.eye(4, dtype=int)[arr],
        joint,
        input_core_dims=[["n_shots"]],
        output_core_dims=[["n_shots", "measured_state"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[int],
    ).sum(dim="n_shots")

    counts = counts.assign_coords(measured_state=("measured_state", basis_labels))
    counts.name = "counts"
    counts.attrs.update(long_name="shot counts per outcome")

    # Total counts per (pair, phase, prepared state)
    total_counts = counts.sum(dim="measured_state")
    total_counts.name = "total_counts"
    prob = (counts / total_counts).astype(float)  # P(y | x), dims: (qubit_pair, prepared_state, measured_state)

    # (optional) readout mitigation using single-qubit confusion matrices M_c, M_t (2x2 each)
    # if you have them, build joint M = kron(M_c, M_t) (shape 4x4) and deconvolve:
    # prob_corr = xr.apply_ufunc(lambda P: np.linalg.pinv(M, rcond=1e-3) @ P,
    #                            prob, input_core_dims=[["measured_state"]],
    #                            output_core_dims=[["measured_state"]],
    #                            vectorize=True)
    # prob_corr = prob_corr.clip(min=0)  # clean negatives
    # prob_corr = prob_corr / prob_corr.sum("measured_state")  # renormalize
    # prob_used = prob_corr

    # ideal mapping indices: 00->00, 01->01, 10->11, 11->10  (0,1,3,2)
    ideal_idx = xr.DataArray([0,1,3,2], dims=["prepared_state"], coords={"prepared_state": basis_labels})

    p_correct = prob.isel(measured_state=ideal_idx)  # dims: (qubit_pair, prepared_state)
    F_TT = p_correct.mean("prepared_state").rename("fidelity_tt")
    F_TT.attrs["long_name"] = "State Fidelity"

    # Attach everything to the dataset
    ds_fit = ds_fit.assign(
        {            
            "prob": prob,
            "counts": counts,
            "total_counts": total_counts,
            "fidelity": F_TT,
        },
    )

    # Build FitParameters for each qubit_pair
    fit_results = {
        qp: FitParameters(
            fidelity=F_TT.sel(qubit_pair=qp).item(),
            success=True,
        )
        for qp in ds_fit.qubit_pair.values
    }

    return ds_fit, fit_results
