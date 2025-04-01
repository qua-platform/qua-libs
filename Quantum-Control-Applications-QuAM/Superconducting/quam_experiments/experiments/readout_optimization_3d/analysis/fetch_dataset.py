import numpy as np

from quam_experiments.experiments.readout_optimization_3d.parameters import (
    get_frequency_detunings_in_hz,
    ReadoutOptimization3dParameters,
    get_amplitude_factors,
    get_durations,
)
from qualibration_libs.qua_datasets import convert_IQ_to_V
from qualibration_libs.save_utils import fetch_results_as_xarray


def fetch_dataset(job, qubits, run_axis: np.ndarray, node_parameters: ReadoutOptimization3dParameters):
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    dfs = get_frequency_detunings_in_hz(node_parameters)
    amps = get_amplitude_factors(node_parameters)
    durations = get_durations(node_parameters)

    ds = fetch_results_as_xarray(
        handles=job.result_handles,
        qubits=qubits,
        measurement_axis={
            "duration": durations,
            "amp": amps,
            "freq": dfs,
            "run": run_axis,
        },
    )
    ds = convert_IQ_to_V(ds, qubits, ["I_g", "Q_g", "I_e", "Q_e"])

    # since we use accumulated demod, the readout length differs over the duration axis.
    # the previous step assumed each segment had the maximum readout length, so we have
    # to "undo" or compensate for the overcorrection.
    for duration in ds.duration:
        ds.loc[dict(duration=duration)] *= duration / node_parameters.max_duration_in_ns

    # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2) for |g> and |e> as well as the distance between the two blobs D
    ds = ds.assign(
        {
            "D": np.sqrt((ds.I_g - ds.I_e) ** 2 + (ds.Q_g - ds.Q_e) ** 2),
            "IQ_abs_g": np.sqrt(ds.I_g**2 + ds.Q_g**2),
            "IQ_abs_e": np.sqrt(ds.I_e**2 + ds.Q_e**2),
        }
    )
    # Add the absolute frequency to the dataset
    dfs = get_frequency_detunings_in_hz(node_parameters)
    ds = ds.assign_coords(
        {
            "freq_full": (
                ["qubit", "freq"],
                np.array([dfs + q.resonator.RF_frequency for q in qubits]),
            )
        }
    )
    ds.freq_full.attrs["long_name"] = "Frequency"
    ds.freq_full.attrs["units"] = "GHz"

    ds = ds.assign_coords(freq_mhz=ds.freq / 1e6)
    ds.freq_mhz.attrs["long_name"] = "Frequency"
    ds.freq_mhz.attrs["units"] = "MHz"

    return ds
