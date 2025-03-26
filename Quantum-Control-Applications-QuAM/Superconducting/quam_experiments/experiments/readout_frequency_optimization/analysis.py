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
    # ds = fetch_results_as_xarray(job.result_handles, qubits, {"freq": dfs})
    # # Convert IQ data into volts
    # ds = convert_IQ_to_V(ds, qubits, ["I_g", "Q_g", "I_e", "Q_e"])
    # # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2) for |g> and |e> as well as the distance between the two blobs D
    # ds = ds.assign(
    #     {
    #         "D": np.sqrt((ds.I_g - ds.I_e) ** 2 + (ds.Q_g - ds.Q_e) ** 2),
    #         "IQ_abs_g": np.sqrt(ds.I_g ** 2 + ds.Q_g ** 2),
    #         "IQ_abs_e": np.sqrt(ds.I_e ** 2 + ds.Q_e ** 2),
    #     }
    # )
    # # Add the absolute frequency to the dataset
    # ds = ds.assign_coords(
    #     {
    #         "freq_full": (
    #             ["qubit", "freq"],
    #             np.array([dfs + q.resonator.RF_frequency for q in qubits]),
    #         )
    #     }
    # )
    # ds.freq_full.attrs["long_name"] = "Frequency"
    # ds.freq_full.attrs["units"] = "GHz"
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
    # # Get the readout detuning as the index of the maximum of the cumulative average of D
    # detuning = ds.D.rolling({"freq": 5}).mean("freq").idxmax("freq")
    # # Get the dispersive shift as the distance between the resonator frequency when the qubit is in |g> and |e>
    # chi = (ds.IQ_abs_e.idxmin(dim="freq") - ds.IQ_abs_g.idxmin(dim="freq")) / 2
    #
    # # Save fitting results
    # fit_results = {
    #     q.name: {"detuning": detuning.loc[q.name].values, "chi": chi.loc[q.name].values}
    #     for q in qubits
    # }
    # node.results["fit_results"] = fit_results
    #
    # for q in qubits:
    #     print(
    #         f"{q.name}: Shifting readout frequency by {fit_results[q.name]['detuning'] / 1e3:.0f} kHz"
    #     )
    #     print(f"{q.name}: Chi = {fit_results[q.name]['chi']:.2f} \n")

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
