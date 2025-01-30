from typing import List, Tuple

from scipy.signal import savgol_filter
import numpy as np
import xarray as xr

from qm import QmJob
from qualang_tools.config import ReadoutResonator

from quam_libs.components import Transmon
from quam_libs.lib.save_utils import fetch_results_as_xarray


def fetch_dataset(job: QmJob, qubits: List[Transmon], resonators: List[ReadoutResonator]):
    """
    Fetches raw ADC data from the OPX and processes it into an xarray dataset with labeled coordinates and attributes.

    Attributes:
    - The dataset will contain the following variables:
        - adcI: Normalized ADC values for the in-phase component.
        - adcQ: Normalized ADC values for the quadrature component.
        - adc_single_runI: Normalized single-run ADC values for the in-phase component.
        - adc_single_runQ: Normalized single-run ADC values for the quadrature component.
        - IQ_abs: Absolute value of the IQ signal, calculated as sqrt(adcI^2 + adcQ^2).

    Coordinates:
    - time: 1D array representing the time axis, with values ranging from 0 to the readout length of the first resonator.

    """
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    time_axis = np.linspace(0, resonators[0].operations["readout"].length, resonators[0].operations["readout"].length)
    ds = fetch_results_as_xarray(job.result_handles, qubits, {"time": time_axis})

    # Convert raw ADC traces into volts
    ds = ds.assign({key: -ds[key] / 2 ** 12 for key in ("adcI", "adcQ", "adc_single_runI", "adc_single_runQ")})
    ds = ds.assign({"IQ_abs": np.sqrt(ds["adcI"] ** 2 + ds["adcQ"] ** 2)})

    return ds


def analyze_pulse_arrival_times(ds: xr.Dataset, qubits: List[Transmon]) -> Tuple[xr.Dataset, dict]:
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


def detect_pulse_arrival(ds: xr.Dataset, qubits: List[Transmon], window_length=11, polyorder=3) -> Tuple[List[int], dict]:
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
        {"con": (["qubit"], [qubit.resonator.opx_input_I.controller_id for qubit in qubits])}
    )
    return ds
