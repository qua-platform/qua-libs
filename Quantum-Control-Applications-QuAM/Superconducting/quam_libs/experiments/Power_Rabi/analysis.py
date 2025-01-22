from typing import List, Dict, Tuple
import numpy as np
import xarray as xr
from qm import QmJob
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.components import Transmon
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit import fit_oscillation, oscillation
from quam_libs.lib.instrument_limits import instrument_limits


def fetch_dataset(job: QmJob, qubits: List[Transmon], N_pi: int, state_discrimination: bool, operation: str, amps: np.ndarray, N_pi_vec: np.ndarray) -> xr.Dataset:
    """
    Fetches raw ADC data from the OPX and processes it into an xarray dataset with labeled coordinates and attributes.

    arguments:
    - job (QmJob): the job object containing the results.
    - qubits (List[Transmon]): the qubits on which the data was acquired.
    - N_pi (int): the max number of pi pulses applied.
    - state_discrimination (bool): whether the data was acquired with state discrimination.
    - operation (str): the operation performed on the qubit (e.g., 'x180').
    - amps (np.ndarray): the amplitude factors of the drive pulse.
    - N_pi_vec (np.ndarray): array of the number of pulses applied.
    
    
    Attributes:
    - The dataset will contain the following variables:
        - I: I signal.
        - Q: Q signal.

    Coordinates:
    - qubit: the qubit on which the data was acquired.
    - N: the number of pi pulses applied.
    - amp: the amplitude factors of the drive pulse.
    - abs_amp: the absolute values of the amplitudes of the drive pulse in volts. 

    
    returns:
    - ds (xr.Dataset): the dataset containing the fetched measurement results.
    """
    
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    if N_pi == 1:
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"amp": amps})
    else:
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"amp": amps, "N": N_pi_vec})
    if not state_discrimination:
        ds = convert_IQ_to_V(ds, qubits)
    # Add the qubit pulse absolute amplitude to the dataset
    ds = ds.assign_coords(
    {
        "abs_amp": (
            ["qubit", "amp"],
            np.array([q.xy.operations[operation].amplitude * amps for q in qubits]),
        )
        }
    )
    
    return ds

def fit_pi_amplitude(ds: xr.Dataset, N_pi: int, state_discrimination: bool, qubits: List[Transmon], operation: str, N_pi_vec: np.ndarray) -> Dict:
    
    """
    Fits the Pi pulse amplitude for a given dataset.
    Parameters:
    ds (xr.Dataset): The dataset containing the experimental data.
    N_pi (int): The max number of Pi pulses.
    state_discrimination (bool): Whether state discrimination is used.
    qubits (List[Transmon]): List of qubits to fit.
    operation (str): The operation being performed (e.g., 'x180').
    N_pi_vec (np.ndarray): Array of Pi pulse numbers.
    Returns:
    Dict: A dictionary containing the fit results.
    """    
    
    fit_results = {}
    pi_amp_fit = {}
    
    if N_pi == 1:
        # Fit the power Rabi oscillations
        if state_discrimination:
            fit = fit_oscillation(ds.state, "amp")
        else:
            fit = fit_oscillation(ds.I, "amp")
        fit_evals = oscillation(
            ds.amp,
            fit.sel(fit_vals="a"),
            fit.sel(fit_vals="f"),
            fit.sel(fit_vals="phi"),
            fit.sel(fit_vals="offset"),
        )
        
        fit_results['fit_evals'] = fit_evals

        # Save fitting results
        for q in qubits:
            pi_amp_fit[q.name] = {}
            f_fit = fit.loc[q.name].sel(fit_vals="f")
            phi_fit = fit.loc[q.name].sel(fit_vals="phi")
            phi_fit = phi_fit - np.pi * (phi_fit > np.pi / 2)
            factor = float(1.0 * (np.pi - phi_fit) / (2 * np.pi * f_fit))
            new_pi_amp = q.xy.operations[operation].amplitude * factor
            limits = instrument_limits(q.xy)
            if new_pi_amp < limits.max_x180_wf_amplitude:
                print(f"amplitude for Pi pulse is modified by a factor of {factor:.2f}")
                print(f"new amplitude is {1e3 * new_pi_amp:.2f} {limits.units} \n")
                pi_amp_fit[q.name]["Pi_amplitude"] = new_pi_amp
            else:
                print(f"Fitted amplitude too high, new amplitude is {limits.max_x180_wf_amplitude} \n")
                pi_amp_fit[q.name]["Pi_amplitude"] = limits.max_x180_wf_amplitude

    elif N_pi > 1:
        # Get the average along the number of pulses axis to identify the best pulse amplitude
        if state_discrimination:
            I_n = ds.state.mean(dim="N")
        else:
            I_n = ds.I.mean(dim="N")
        if (N_pi_vec[0] % 2 == 0 and operation == "x180") or (N_pi_vec[0] % 2 != 0 and operation != "x180"):
            data_max_idx = I_n.argmin(dim="amp")
        else:
            data_max_idx = I_n.argmax(dim="amp")
        
        fit_results['data_max_idx'] = data_max_idx

        # Save fitting results
        for q in qubits:
            new_pi_amp = float(ds.abs_amp.sel(qubit=q.name)[data_max_idx.sel(qubit=q.name)].data)
            pi_amp_fit[q.name] = {}
            limits = instrument_limits(q.xy)
            if new_pi_amp < limits.max_x180_wf_amplitude:
                pi_amp_fit[q.name]["Pi_amplitude"] = new_pi_amp
                print(
                    f"amplitude for Pi pulse is modified by a factor of {I_n.idxmax(dim='amp').sel(qubit = q.name):.2f}"
                )
                print(f"new amplitude is {1e3 * new_pi_amp:.2f} {limits.units} \n")
            else:
                print(f"Fitted amplitude too high, new amplitude is {limits.max_x180_wf_amplitude} \n")
                pi_amp_fit[q.name]["Pi_amplitude"] = limits.max_x180_wf_amplitude
    
    fit_results['pi_amp_fit'] = pi_amp_fit
    
    return fit_results