import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from quam_libs.qua_datasets import add_amplitude_and_phase, convert_IQ_to_V
from quam_experiments.analysis.fit import peaks_dips
from quam_config.instrument_limits import instrument_limits


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

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
    # ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    # ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    # full_freq = np.array(
    #     [ds.detuning + q.xy.RF_frequency for q in node.namespace["qubits"]]
    # )
    # ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    # ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    pass
    # # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    # ds = fetch_results_as_xarray(
    #     job.result_handles,
    #     qubits,
    #     {"amplitude": amps, "N": np.linspace(1, n_runs, n_runs)},
    # )
    # # Add the absolute readout power to the dataset
    # ds = ds.assign_coords(
    #     {
    #         "readout_amp": (
    #             ["qubit", "amplitude"],
    #             np.array(
    #                 [
    #                     amps * q.resonator.operations["readout"].amplitude
    #                     for q in qubits
    #                 ]
    #             ),
    #         )
    #     }
    # )
    # # Rearrange the data to combine I_g and I_e into I, and Q_g and Q_e into Q
    # ds_rearranged = xr.Dataset()
    # # Combine I_g and I_e into I
    # ds_rearranged["I"] = xr.concat([ds.I_g, ds.I_e], dim="state")
    # ds_rearranged["I"] = ds_rearranged["I"].assign_coords(state=[0, 1])
    # # Combine Q_g and Q_e into Q
    # ds_rearranged["Q"] = xr.concat([ds.Q_g, ds.Q_e], dim="state")
    # ds_rearranged["Q"] = ds_rearranged["Q"].assign_coords(state=[0, 1])
    # # Copy other coordinates and data variables
    # for var in ds.coords:
    #     if var not in ds_rearranged.coords:
    #         ds_rearranged[var] = ds[var]
    #
    # for var in ds.data_vars:
    #     if var not in ["I_g", "I_e", "Q_g", "Q_e"]:
    #         ds_rearranged[var] = ds[var]
    #
    # # Replace the original dataset with the rearranged one
    # ds = ds_rearranged
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
    # def apply_fit_gmm(I, Q):
    #     I_mean = np.mean(I, axis=1)
    #     Q_mean = np.mean(Q, axis=1)
    #     means_init = [[I_mean[0], Q_mean[0]], [I_mean[1], Q_mean[1]]]
    #     precisions_init = [
    #         1 / ((np.mean(np.var(I, axis=1)) + np.mean(np.var(Q, axis=1))) / 2)
    #     ] * 2
    #     clf = GaussianMixture(
    #         n_components=2,
    #         covariance_type="spherical",
    #         means_init=means_init,
    #         precisions_init=precisions_init,
    #         tol=1e-5,
    #         reg_covar=1e-12,
    #     )
    #     X = np.array([np.array(I).flatten(), np.array(Q).flatten()]).T
    #     clf.fit(X)
    #     meas_fidelity = (
    #         np.sum(clf.predict(np.array([I[0], Q[0]]).T) == 0) / len(I[0])
    #         + np.sum(clf.predict(np.array([I[1], Q[1]]).T) == 1) / len(I[1])
    #     ) / 2
    #     loglikelihood = clf.score_samples(X)
    #     max_ll = np.max(loglikelihood)
    #     outliers = np.sum(loglikelihood > np.log(0.01) + max_ll) / len(X)
    #     return np.array([meas_fidelity, outliers])
    #
    # fit_res = xr.apply_ufunc(
    #     apply_fit_gmm,
    #     ds.I,
    #     ds.Q,
    #     input_core_dims=[["state", "N"], ["state", "N"]],
    #     output_core_dims=[["result"]],
    #     vectorize=True,
    # )
    #
    # fit_res = fit_res.assign_coords(result=["meas_fidelity", "outliers"])
    #
    # plot_individual = False
    # best_data = {}
    #
    # best_amp = {}
    # for q in qubits:
    #     fit_res_q = fit_res.sel(qubit=q.name)
    #     valid_amps = fit_res_q.amplitude[
    #         (fit_res_q.sel(result="outliers") >= node.parameters.outliers_threshold)
    #     ]
    #     amps_fidelity = fit_res_q.sel(
    #         amplitude=valid_amps.values, result="meas_fidelity"
    #     )
    #     best_amp[q.name] = float(amps_fidelity.readout_amp[amps_fidelity.argmax()])
    #     print(f"amp for {q.name} is {best_amp[q.name]}")
    #     node.results["results"][q.name] = {}
    #     node.results["results"][q.name]["best_amp"] = best_amp[q.name]
    #
    #     # Select data for the best amplitude
    #     best_amp_data = ds.sel(qubit=q.name, amplitude=float(amps_fidelity.idxmax()))
    #     best_data[q.name] = best_amp_data
    #
    #     # Extract I and Q data for ground and excited states
    #     I_g = best_amp_data.I.sel(state=0)
    #     Q_g = best_amp_data.Q.sel(state=0)
    #     I_e = best_amp_data.I.sel(state=1)
    #     Q_e = best_amp_data.Q.sel(state=1)
    #     angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(
    #         I_g, Q_g, I_e, Q_e, True, b_plot=plot_individual
    #     )
    #     I_rot = I_g * np.cos(angle) - Q_g * np.sin(angle)
    #     hist = np.histogram(I_rot, bins=100)
    #     RUS_threshold = hist[1][1:][np.argmax(hist[0])]
    #     if plot_individual:
    #         fig = plt.gcf()
    #         plt.show()
    #         node.results["figs"][q.name] = fig
    #     node.results["results"][q.name]["angle"] = float(angle)
    #     node.results["results"][q.name]["threshold"] = float(threshold)
    #     node.results["results"][q.name]["fidelity"] = float(fidelity)
    #     node.results["results"][q.name]["confusion_matrix"] = np.array(
    #         [[gg, ge], [eg, ee]]
    #     )
    #     node.results["results"][q.name]["rus_threshold"] = float(RUS_threshold)
    # node.outcomes = {q.name: "successful" for q in node.namespace["qubits"]}

    ds_fit = ds
    fit_results = {
        q: FitParameters(
            success=False,
        )
        for q in ds_fit.qubit.values
    }
    return ds_fit, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    pass
    # return fit, fit_results
