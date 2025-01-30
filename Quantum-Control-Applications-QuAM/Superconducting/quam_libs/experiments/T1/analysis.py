from typing import List
import numpy as np
import xarray as xr
from qm import QmJob
from qualang_tools.units import unit
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.components import Transmon
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit import fit_decay_exp, decay_exp

def fetch_dataset(job: QmJob, qubits: List[Transmon], idle_times: np.ndarray) -> xr.Dataset:
    
    """
    Fetches raw ADC data from the OPX and processes it into an xarray dataset with labeled coordinates and attributes.
    
    arguments:
    - job (QmJob): the job object containing the results.
    - qubits (List[Transmon]): the qubits on which the data was acquired.
    - idle_times (np.ndarray): the idle times array.
    - unit (unit): the unit object.
    
    Attributes:
    - The dataset will contain the following variables:
        - adcI: Normalized ADC values for the in-phase component.
        - adcQ: Normalized ADC values for the quadrature component.
       
    Coordinates:
    - idle_time: 1D array representing the idle time axis.
    
    returns:
    - ds (xr.Dataset): the dataset containing the fetched measurement results.

    """
    
    u = unit(coerce_to_integer=True)
    
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    ds = fetch_results_as_xarray(job.result_handles, qubits, {"idle_time": idle_times})
    # Convert IQ data into volts
    ds = convert_IQ_to_V(ds, qubits)
    # Convert time into µs
    ds = ds.assign_coords(idle_time=4 * ds.idle_time / u.us)  # convert to µs
    ds.idle_time.attrs = {"long_name": "idle time", "units": "µs"}
    
    return ds

def fit_exponential_decay(ds: xr.Dataset, node_parameters) -> dict:
    
    """
    Fits the T1 associated with the exponential decay.
    Parameters:
    ds (xr.Dataset): The dataset containing the experimental data.
    note_parameters: The node parameters.

    Returns:
    Dict: A dictionary containing the fit results.
    """    
    
    # Fit the exponential decay
    if node_parameters.use_state_discrimination:
        fit_data = fit_decay_exp(ds.state, "idle_time")
    else:
        fit_data = fit_decay_exp(ds.I, "idle_time")
    fit_data.attrs = {"long_name": "time", "units": "µs"}
    # Fitted decay
    fitted = decay_exp(
        ds.idle_time,
        fit_data.sel(fit_vals="a"),
        fit_data.sel(fit_vals="offset"),
        fit_data.sel(fit_vals="decay"),
    )
    # Decay rate and its uncertainty
    decay = fit_data.sel(fit_vals="decay")
    decay.attrs = {"long_name": "decay", "units": "ns"}
    decay_res = fit_data.sel(fit_vals="decay_decay")
    decay_res.attrs = {"long_name": "decay", "units": "ns"}
    # T1 and its uncertainty
    tau = -1 / fit_data.sel(fit_vals="decay")
    tau.attrs = {"long_name": "T1", "units": "µs"}
    tau_error = -tau * (np.sqrt(decay_res) / decay)
    tau_error.attrs = {"long_name": "T1 error", "units": "µs"}
    
    fit_results = {"fitted": fitted, "tau": tau, "tau_error": tau_error}
    
    return fit_results