from typing import List, Tuple

from scipy.signal import savgol_filter
import numpy as np
import xarray as xr

from qm import QmJob
from qualang_tools.config import ReadoutResonator

from quam_builder.architecture.superconducting.qubit import AnyTransmon
from qualibration_libs.save_utils import fetch_results_as_xarray
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.qua_datasets import add_amplitude_and_phase, convert_IQ_to_V
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
    # print(f"Time Of Flight to add: {delays} ns")
    # print(f"Offsets to add for I: {ds.offsets_I.values * 1000} mV")
    # print(f"Offsets to add for Q: {ds.offsets_Q.values * 1000} mV")

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
    # Convert raw ADC traces into volts
    # ds = ds.assign({key: -ds[key] / 2 ** 12 for key in ("adcI", "adcQ", "adc_single_runI", "adc_single_runQ")})
    # ds = ds.assign({"IQ_abs": np.sqrt(ds["adcI"] ** 2 + ds["adcQ"] ** 2)})
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
    # # Filter the data to get the pulse arrival time
    # filtered_adc = savgol_filter(np.abs(ds.IQ_abs), 11, 3)
    # # Detect the arrival of the readout signal
    # delays = []
    # fit_results = {}
    # for q in qubits:
    #     fit_results[q.name] = {}
    #     # Filter the time trace to easily find the rising edge of the readout pulse
    #     filtered_adc = savgol_filter(np.abs(ds.sel(qubit=q.name).IQ_abs), 11, 3)
    #     # Get a threshold to detect the rising edge
    #     th = (np.mean(filtered_adc[:100]) + np.mean(filtered_adc[:-100])) / 2
    #     delay = np.where(filtered_adc > th)[0][0]
    #     # Find the closest multiple integer of 4ns
    #     delay = np.round(delay / 4) * 4
    #     delays.append(delay)
    #     fit_results[q.name]["delay_to_add"] = delay
    #     fit_results[q.name]["fit_successful"] = True
    # node.results["fit_results"] = fit_results
    # # Add the delays to the dataset
    # ds = ds.assign_coords({"delays": (["qubit"], delays)})
    # ds = ds.assign_coords({"offsets_I": ds.adcI.mean(dim="time")})
    # ds = ds.assign_coords({"offsets_Q": ds.adcQ.mean(dim="time")})
    # ds = ds.assign_coords(
    #     {
    #         "con": (
    #             ["qubit"],
    #             [node.machine.qubits[q.name].resonator.opx_input_I.controller_id for q in qubits],
    #         )
    #     }
    # )
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


def analyze_pulse_arrival_times(ds: xr.Dataset, qubits: List[AnyTransmon]) -> Tuple[xr.Dataset, dict]:
    """
    Processes the dataset by filtering signals, detecting pulse arrival times, and adding coordinates.

    Adds the following coordinates to the dataset:
    - delays: Delay values (in nanoseconds) for each qubit.
    - offsets_I: Mean of the in-phase ADC values across the time axis.
    - offsets_Q: Mean of the quadrature ADC values across the time axis.
    - con: Controller IDs of the resonators associated with each qubit.

    Returns:
    - ds: Updated xarray.Dataset with the new coordinates.
    - fit_results: A dictionary of relevant fit results for each qubit.
    """
    delays, fit_results = detect_pulse_arrival(ds, qubits)
    ds = add_coordinates_to_dataset(ds, delays, qubits)

    return ds, fit_results


def detect_pulse_arrival(
    ds: xr.Dataset, qubits: List[AnyTransmon], window_length=11, polyorder=3
) -> Tuple[List[int], dict]:
    """
    Detects the pulse arrival times for each qubit by analyzing the filtered IQ_abs signal.

    Returns:
    - delays: List of delays (in nanoseconds) for each qubit, rounded to the nearest 4ns.
    - fit_results: Dictionary with fit results for each qubit, including delay and success status.
    """
    delays = []
    fit_results = {}

    for q in qubits:
        fit_results[q.name] = {}
        # Filter the time trace for the qubit's IQ_abs
        filtered_adc = savgol_filter(np.abs(ds.sel(qubit=q.name).IQ_abs), window_length, polyorder)
        # Calculate a threshold to detect the rising edge
        threshold = (np.mean(filtered_adc[:100]) + np.mean(filtered_adc[:-100])) / 2
        delay = np.where(filtered_adc > threshold)[0][0]
        # Round to the nearest multiple of 4ns
        delay = np.round(delay / 4) * 4
        delays.append(delay)
        fit_results[q.name]["delay_to_add"] = delay
        fit_results[q.name]["fit_successful"] = True

    return delays, fit_results


def filter_adc_signal(ds, window_length=11, polyorder=3):
    """
    Applies a Savitzky-Golay filter to smooth the absolute IQ signal in the dataset.
    """
    return savgol_filter(np.abs(ds.IQ_abs), window_length, polyorder)


def add_coordinates_to_dataset(ds, delays, qubits):
    ds = ds.assign_coords({"delays": (["qubit"], delays)})
    ds = ds.assign_coords({"offsets_I": ds.adcI.mean(dim="time")})
    ds = ds.assign_coords({"offsets_Q": ds.adcQ.mean(dim="time")})
    ds = ds.assign_coords(
        {
            "con": (
                ["qubit"],
                [qubit.resonator.opx_input_I.controller_id for qubit in qubits],
            )
        }
    )
    return ds
