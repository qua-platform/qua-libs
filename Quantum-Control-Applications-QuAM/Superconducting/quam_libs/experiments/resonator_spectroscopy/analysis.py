from typing import List

import numpy as np
import xarray as xr

from qm import QmJob
from quam_libs.lib.qua_datasets import apply_angle, subtract_slope, convert_IQ_to_V

from quam_libs.components import Transmon
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit_utils import fit_resonator


def fetch_dataset(job: QmJob, qubits: List[Transmon], frequencies: List[float]):
    """
    Fetches IQ measurement results from the OPX as a function of resonator frequency detuning and processes them into an xarray dataset.

    Args:
        job: QmJob object containing the measurement results
        qubits: List of Transmon qubits being measured
        frequencies: List of frequency detunings to sweep through relative to each resonator's RF frequency

    Returns:
        xarray.Dataset containing the following variables:
            - I: In-phase component in volts
            - Q: Quadrature component in volts 
            - IQ_abs: Magnitude of the IQ signal in volts
            - phase: Phase of the IQ signal in radians, with linear frequency dependence subtracted

        The dataset has coordinates:
            - qubit: Names of the measured qubits
            - freq: Frequency detunings from the resonator frequencies
            - freq_full: Absolute frequencies (detuning + resonator RF frequency)
    """
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    ds = fetch_results_as_xarray(job.result_handles, qubits, {"freq": frequencies})
    # Convert raw ADC traces into volts
    ds = convert_IQ_to_V(ds, qubits)
    ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
    # Derive the phase IQ_abs = angle(I + j*Q)
    ds = ds.assign({"phase": subtract_slope(apply_angle(ds.I + 1j * ds.Q, dim="freq"), dim="freq")})
    # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
    ds = ds.assign_coords(
        {
            "freq_full": (
                ["qubit", "freq"],
                np.array([frequencies + q.resonator.RF_frequency for q in qubits]),
            )
        }
    )
    return ds

def fit_resonators(ds: xr.Dataset, qubits: List[Transmon]):
    """
    Fits the resonator frequency and quality factors for each qubit in the dataset.
    """
    fits = {}
    fit_evals = {}
    fit_results = {}

    for index, q in enumerate(qubits):
        fit, fit_eval = fit_resonator(ds.sel(qubit=q.name), q.resonator.RF_frequency)
        fits[q.name] = fit
        fit_evals[q.name] = fit_eval
        Qe = np.abs(fit.params["Qe_real"].value + 1j * fit.params["Qe_imag"].value)
        Qi = 1 / (1 / fit.params["Q"].value - 1 / Qe)
        fit_results[q.name] = {}
        fit_results[q.name]["resonator_freq_detuning"] = fit.params["omega_r"].value
        fit_results[q.name]["resonator_freq_full"] = fit.params["omega_r"].value + q.resonator.RF_frequency
        fit_results[q.name]["Quality_external"] = Qe
        fit_results[q.name]["Quality_internal"] = Qi
    
    ds = ds.assign({"fit_evals": (("qubit","freq"),np.array([fit_evals[q.name] for q in qubits]))})
    return ds, fit_results
