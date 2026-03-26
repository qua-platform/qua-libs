import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate.core import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V


@dataclass
class FitParameters:
    """Stores the relevant resonator spectroscopy experiment fit parameters for a single qubit"""

    gain: float
    snr: float
    method: str
    twpa_power_p: float
    twpa_frequency_p: float
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
        s_snr = f"\tSNR: {fit_results[q]['snr']:.2f} dB"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_gain + s_snr)
    s_opt_params = (
        f"Optimal TWPA parameters:\n"
        f"\t TWPA pump power: {fit_results[q]['twpa_power_p']:.2f} dBm\n"
        f"\t TWPA pump frequency: {fit_results[q]['twpa_frequency_p'] / 1e9:.3f} Ghz\n"
    )
    log_callable(s_opt_params)


def process_raw_dataset(node: QualibrationNode):
    # Extract the raw datasets
    ds_on = node.results["ds_raw_on"]
    ds_off = node.results["ds_raw_off"]
    # Convert data into V
    ds_off = convert_IQ_to_V(ds_off, node.namespace["qubits"], IQ_list=["Ioff", "Qoff"])
    ds_on = convert_IQ_to_V(ds_on, node.namespace["qubits"], IQ_list=["Ion", "Qon"])
    # Get the power from I and Q for pump on and off
    ds_off = ds_off.assign({"IQ_abs_off": np.sqrt(ds_off.Ioff**2 + ds_off.Qoff**2)})
    ds_on = ds_on.assign({"IQ_abs_on": np.sqrt(ds_on.Ion**2 + ds_on.Qon**2)})

    # Merge the on and off datasets
    ds_off_broadcast = ds_off.broadcast_like(ds_on)
    ds = xr.merge([ds_on, ds_off_broadcast])

    # Get the SNR for pump on and off
    std_off = ds.IQ_abs_off.std(dim="shots")
    mean_off = ds.IQ_abs_off.mean(dim="shots")
    ds = ds.assign({"snr_off": mean_off / std_off})
    std_on = ds.IQ_abs_on.std(dim="shots")
    mean_on = ds.IQ_abs_on.mean(dim="shots")
    ds = ds.assign({"snr_on": mean_on / std_on})
    # Extract the gain and SNR
    ds = ds.assign({"gain": 20 * np.log(mean_on / mean_off)})
    ds = ds.assign({"snr": 20 * np.log(ds["snr_on"] / ds["snr_off"])})
    # Add the RF frequency to the dataset
    full_freq_p = np.array(
        [
            (ds.detuning_p + node.machine.twpas[node.namespace["qubit_to_twpa"][q.name]].pump.RF_frequency) * 1e-9
            for q in node.namespace["qubits"]
        ]
    )
    ds = ds.assign_coords(full_freq_p=(["qubit", "detuning_p"], full_freq_p))
    ds.full_freq_p.attrs = {"long_name": "RF pump frequency", "units": "GHz"}
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

    ds_fit = ds
    # Only keep points where all qubits have gain >= min_gain
    gain_min_over_qubits = ds_fit.gain.min(dim="qubit")
    gain_all_ok = gain_min_over_qubits >= node.parameters.min_gain

    # Aggregate SNR per point (choose one):
    snr_min_over_qubits = ds_fit.snr.min(dim="qubit")
    if node.parameters.optimizer_method == "average":
        snr_agg = ds_fit.snr.mean(dim="qubit")  # average SNR across qubits
    elif node.parameters.optimizer_method == "worst-qubit":
        snr_agg = ds_fit.snr.min(dim="qubit")  # or worst-qubit SNR (good for multiplexed readout)
    else:
        raise NotImplementedError("Method not implemented")
    # Find the point for which the SNR and the gain are above the thresholds
    snr_constrained = snr_agg.where(gain_all_ok)
    # Get the highest SNR among all the possibilities
    max_snr = snr_constrained.max()
    if np.isnan(max_snr):
        raise ValueError(f"There is no pumping point which satisfies gain >= {node.parameters.min_gain}")
    # Extract the corresponding TWPAI parameters
    at_best = snr_agg.where(
        (snr_agg == max_snr) & gain_all_ok,
        drop=True,
    )
    coords_best = {dim: at_best.coords[dim].values.flat[0] for dim in at_best.dims}

    # Gain and SNR per qubit at this point (coords_best has no "qubit", so this keeps qubit dim)
    gain_at_best = ds_fit.gain.sel(**coords_best)  # DataArray with qubit dim
    snr_at_best = ds_fit.snr.sel(**coords_best)  # DataArray with qubit dim

    ds_fit.attrs["coords_best"] = coords_best
    ds_fit.attrs["method"] = node.parameters.optimizer_method
    ds_fit = ds_fit.assign(gain_at_best=gain_at_best, snr_at_best=snr_at_best)
    print(
        f"Best SNR ({node.parameters.optimizer_method}) satisfying (gain > {node.parameters.min_gain} dB):"
        f"\n\tgain{gain_at_best.qubit.values}: {gain_at_best.values})"
        f"\n\tsnr{snr_at_best.qubit.values}: {snr_at_best.values} "
        f"\nobtained for {coords_best}:"
    )
    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""

    qubit_to_twpa = node.namespace["qubit_to_twpa"]
    # Assess whether the fit was successful or not
    gain_success = ~np.isnan(fit.gain_at_best.data)
    snr_success = ~np.isnan(fit.snr_at_best.data)
    success_criteria = gain_success & snr_success
    fit = fit.assign_coords(success=("qubit", success_criteria))
    fit_results = {
        q: FitParameters(
            gain=fit.sel(qubit=q).gain_at_best.values.__float__(),
            snr=fit.sel(qubit=q).snr_at_best.values.__float__(),
            method=fit.method,
            twpa_power_p=fit.coords_best["twpa_power_p"],
            twpa_frequency_p=fit.sel(qubit=q)
            .full_freq_p.sel(detuning_p=fit.coords_best["detuning_p"])
            .values.__float__()
            * 1e9,
            success=fit.sel(qubit=q).success.values.__bool__(),
        )
        for q in qubit_to_twpa.keys()
    }
    return fit, fit_results
