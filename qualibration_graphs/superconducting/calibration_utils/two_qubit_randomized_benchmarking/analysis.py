import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from qualibration_libs.analysis import fit_decay_exp


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    error_per_clifford: float
    # error_per_gate: float
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
    for q in fit_results.keys():
        s_qubit = f"Results for qubit pair {q}: "
        # s_fidelity = f"\tSingle qubit gate fidelity: {100 * (1 - fit_results[q]['error_per_gate']):.3f} %\n"
        s_fidelity = f"\tTwo-qubit Clifford gate fidelity: {100 * (1 - fit_results[q]['error_per_clifford']):.3f} %\n"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_fidelity)
    pass


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, qubit_pairs=node.namespace["qubit_pairs"])
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
    if node.parameters.use_state_discrimination:
        ds_fit["averaged_data"] = 1 - ds.state.mean(dim="nb_of_sequences")
    else:
        raise NotImplementedError()
    # Fit the exponential decay
    fit_data = fit_decay_exp(ds_fit["averaged_data"], "depths")

    ds_fit = xr.merge([ds, fit_data.rename("fit_data")])

    # # select
    # use_sd = node.parameters.use_state_discrimination
    # sig_c = ds.state_c if use_sd else ds.I_c
    # sig_t = ds.state_t if use_sd else ds.I_t

    # # average over sequences
    # avg_c = (1 - sig_c.mean(dim="nb_of_sequences")).rename("averaged_data_c")
    # avg_t = (1 - sig_t.mean(dim="nb_of_sequences")).rename("averaged_data_t")

    # # fit per qubit_pair (keeps vectorization; adjust if fit_decay_exp signature differs)
    # fit_c = (
    #     avg_c.groupby("qubit_pair")
    #     .map(lambda g: fit_decay_exp(g, "depths"))
    #     .rename("fit_data_c")
    # )
    # fit_t = (
    #     avg_t.groupby("qubit_pair")
    #     .map(lambda g: fit_decay_exp(g, "depths"))
    #     .rename("fit_data_t")
    # )

    # # final dataset
    # ds_fit = xr.merge([ds, avg_c, avg_t, fit_c, fit_t])

    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)

    return ds_fit, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    # Extract the decay rate
    alpha = np.exp(fit.fit_data.sel(fit_vals="decay"))
    n_qubits = 2
    d = 2**n_qubits
    fit["error_per_clifford"] = (1 - alpha) * (1 - 1 / d)
    # Assess whether the fit was successful or not
    nan_success = np.isnan(fit.error_per_clifford)
    rb_success = (0 < fit.error_per_clifford) & (fit.error_per_clifford < 1)
    success_criteria = ~nan_success & rb_success
    fit = fit.assign({"success": success_criteria})

    # TODO
    # For 2 qubit Clifford, the decomposition is ~8.25 1Q gate and 1.5 2Q gate
    # However, the epg cannot be obtained directly
    # One should instead should use alpha_2q = (a0+a1+3*a0*a1)*a01/5
    # a0 = alpha_0**(N1/2), a1 = alpha_1**(N1/2), a01 = alpha_12**N2
    # N1=8.25, N2=1.5 https://journals.aps.org/prx/abstract/10.1103/PhysRevX.9.021011
    # N1=12.2167, N2=1.5 https://arxiv.org/pdf/1712.06550v2
    # which meant that the relevant alpha_12 is (alpha_2q/((a0+a1+3*a0*a1)/5))**(1/N2)
    # https://arxiv.org/src/1712.06550v2/anc/threeq_supp.pdf
    # https://qiskit-community.github.io/qiskit-experiments/manuals/verification/randomized_benchmarking.html#id10
    # alpha_c = 0
    # alpha_t = 0
    # N1 = 8.25
    # N2 = 1.5
    # a0 = alpha_c ** (N1 / 2)
    # a1 = alpha_t ** (N1 / 2)
    # alpha_12 = (alpha / ((a0 + a1 + 3 * a0 * a1) / 5)) ** (1 / N2)

    # Save fitting results
    fit_results = {
        qp: FitParameters(
            error_per_clifford=float(fit.sel(qubit_pair=qp)["error_per_clifford"]),
            success=bool(fit.sel(qubit_pair=qp).success),
        )
        for qp in fit.qubit_pair.values
    }
    node.outcomes = {
        qp: "successful" if fit_results[qp].success else "fail"
        for qp in fit.qubit_pair.values
    }

    return fit, fit_results
