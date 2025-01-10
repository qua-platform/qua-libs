from typing import List, Dict
import numpy as np
import xarray as xr
from qm import QmJob
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.components import Transmon
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit import peaks_dips


def fetch_dataset(job: QmJob, qubits: List[Transmon], frequencies: List[float]):
    """
    Fetches IQ measurement results from the OPX as a function of resonator frequency detuning and processes them into an xarray dataset.

    Args:
        job: QmJob object containing the measurement results
        qubits: List of Transmon qubits being measured
        frequencies: List of frequency detuning to sweep through relative to each resonator's RF frequency
        detuning:

    Returns:
        xarray.Dataset containing the following variables:
            - I: In-phase component in volts
            - Q: Quadrature component in volts
            - IQ_abs: Magnitude of the IQ signal in volts
            - phase: Phase of the IQ signal in radians, with linear frequency dependence subtracted

        The dataset has coordinates:
            - qubit: Names of the measured qubits
            - freq: Frequency detuning from the resonator frequencies
            - freq_full: Absolute frequencies (detuning + resonator RF frequency)
    """
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    ds = fetch_results_as_xarray(job.result_handles, qubits, {"freq": frequencies})
    # Convert raw ADC traces into volts
    ds = convert_IQ_to_V(ds, qubits)
    ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
    # Derive the phase IQ_abs = angle(I + j*Q)
    ds = ds.assign({"phase": np.arctan2(ds.Q, ds.I)})
    # Add the qubit RF frequency axis of each qubit to the dataset coordinates for plotting
    ds = ds.assign_coords(
        {
            "freq_full": (
                ["qubit", "freq"],
                np.array([frequencies + q.xy.RF_frequency for q in qubits]),
            )
        }
    )
    ds.freq_full.attrs["long_name"] = "Frequency"
    ds.freq_full.attrs["units"] = "GHz"
    return ds


def fit_qubits(ds: xr.Dataset, qubits: List[Transmon], node_parameters):
    """
    Fits the qubit frequency and quality factors for each qubit in the dataset.
    """
    fit_results = {}
    # search for frequency for which the amplitude the farthest from the mean to indicate the approximate location of the peak
    shifts = np.abs((ds.IQ_abs - ds.IQ_abs.mean(dim="freq"))).idxmax(dim="freq")
    # Find the rotation angle to align the separation along the 'I' axis
    angle = np.arctan2(
        ds.sel(freq=shifts).Q - ds.Q.mean(dim="freq"),
        ds.sel(freq=shifts).I - ds.I.mean(dim="freq"),
    )
    # rotate the data to the new I axis
    ds = ds.assign({"I_rot": ds.I * np.cos(angle) + ds.Q * np.sin(angle)})
    # Find the peak with minimal prominence as defined, if no such peak found, returns nan
    result = peaks_dips(ds.I_rot, dim="freq", prominence_factor=5)
    fit_results["fit_ds"] = result

    # Save fitting results
    for q in qubits:
        fit_results[q.name] = {}
        if not np.isnan(result.sel(qubit=q.name).position.values):
            fit_results[q.name]["fit_successful"] = True
            print(
                f"Drive frequency for {q.name} is "
                f"{(result.sel(qubit=q.name).position.values + q.xy.RF_frequency) / 1e9:.6f} GHz"
            )
            fit_results[q.name]["drive_freq"] = result.sel(qubit=q.name).position.values + q.xy.RF_frequency
            # Get optimum iw angle
            prev_angle = q.resonator.operations["readout"].integration_weights_angle
            if not prev_angle:
                prev_angle = 0.0
            fit_results[q.name]["angle"] = (prev_angle + angle.sel(qubit=q.name).values) % (2 * np.pi)

            # Get saturation amplitude
            Pi_length = q.xy.operations["x180"].length
            used_amp = q.xy.operations["saturation"].amplitude * node_parameters.operation_amplitude_factor
            factor_cw = float(node_parameters.target_peak_width / result.sel(qubit=q.name).width.values)
            fit_results[q.name]["saturation_amplitude"] = (
                factor_cw * used_amp / node_parameters.operation_amplitude_factor
            )

            # get expected x180 amplitude
            factor_pi = np.pi / (result.sel(qubit=q.name).width.values * Pi_length * 1e-9)
            fit_results[q.name]["x180_amplitude"] = factor_pi * used_amp

            print(f"(shift of {result.sel(qubit=q.name).position.values / 1e6:.3f} MHz)")
            print(f"Found a peak width of {result.sel(qubit=q.name).width.values / 1e6:.2f} MHz")
            print(
                f"To obtain a peak width of {node_parameters.target_peak_width / 1e6:.1f} MHz the cw amplitude is modified "
                f"by {factor_cw:.2f} to {factor_cw * used_amp / node_parameters.operation_amplitude_factor * 1e3:.0f} mV"
            )
            print(
                f"To obtain a Pi pulse at {Pi_length} ns the Rabi amplitude is modified by {factor_pi:.2f} "
                f"to {factor_pi * used_amp * 1e3:.0f} mV"
            )
            print(f"readout angle for qubit {q.name}: {angle.sel(qubit=q.name).values:.4}\n")

        else:
            fit_results[q.name]["fit_successful"] = False
            print(f"Failed to find a peak for {q.name}\n")
    return ds, fit_results
