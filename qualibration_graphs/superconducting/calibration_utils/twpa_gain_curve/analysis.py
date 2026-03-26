import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr
from qualibrate.core import QualibrationNode


@dataclass
class FitParameters:
    """Stores the relevant resonator spectroscopy experiment fit parameters for a single qubit"""

    gain: float
    max_gain: float
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
        s_qubit = f"Results for qubit {q}: "
        s_gain = f"\tGain: {fit_results[q]['gain']:.2f} dB | "
        s_maxgain = f"\tMax TWPA Gain: {fit_results[q]['max_gain']:.2f} dB | "
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_gain + s_maxgain)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    # Extract the raw datasets
    ds = node.results["ds_raw"]
    # Convert data into V
    qubits = [node.machine.qubits[t.qubits[0]] for t in node.namespace["twpas"]]
    readout_lengths = xr.DataArray(
        [q.resonator.operations["readout"].length for q in qubits],
        coords=[("twpa", [t.name for t in node.namespace["twpas"]])],
    )
    ds = ds.assign({key: ds[key] * 2**12 / readout_lengths for key in ["Ion", "Qon", "Ioff", "Qoff"]})
    # Get the power from I and Q for pump on and off
    ds = ds.assign({"IQ_abs_off": np.sqrt(ds.Ioff**2 + ds.Qoff**2)})
    ds = ds.assign({"IQ_abs_on": np.sqrt(ds.Ion**2 + ds.Qon**2)})
    # Extract the gain
    ds = ds.assign({"gain": 20 * np.log(ds["IQ_abs_on"] / ds["IQ_abs_off"])})
    ds.gain.attrs = {"long_name": "Amplification gain", "units": "dB"}
    # Add the RF frequency to the dataset
    full_freq = np.array([(ds.detuning + node.parameters.frequency_center_in_mhz * 1e6) * 1e-9 for q in qubits])
    ds = ds.assign_coords(full_freq=(["twpa", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "GHz"}
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the T1 relaxation time for each qubit according to ``a * np.exp(t * decay) + offset``.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node : QualibrationNode
        The QUAlibrate node.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    """

    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""

    # for twpa in fit.twpa.values:
    #     max_gain = fit.sel(twpa=twpa).gain.max()
    #     at_max = fit.sel(twpa=twpa).gain.where(fit.sel(twpa=twpa).gain == max_gain, drop=True).squeeze()
    #     coords_at_max = {dim: at_max.coords[dim].item() for dim in at_max.coords}
    #     print(f"{twpa}: Maximum gain of {max_gain.values:.2f} dB obtained at {coords_at_max['full_freq']:.5f} GHz")

    qubit_to_twpa = node.namespace["qubit_to_twpa"]
    # Assess whether the fit was successful or not
    gain_success =  ~np.isnan(fit.gain.mean(dim="detuning").data)
    success_criteria = gain_success
    fit = fit.assign_coords(success=("twpa", success_criteria))

    fit_results = {
        q: FitParameters(
            gain=fit.sel(twpa=qubit_to_twpa[q]).sel(full_freq=node.machine.qubits[q].resonator.RF_frequency * 1e-9, method="nearest").gain.values.__float__(),
            max_gain=fit.sel(twpa=qubit_to_twpa[q]).gain.max().values.__float__(),
            success=fit.sel(twpa=qubit_to_twpa[q]).success.values.__bool__(),
        )
        for q in qubit_to_twpa.keys()
    }
    return fit, fit_results
