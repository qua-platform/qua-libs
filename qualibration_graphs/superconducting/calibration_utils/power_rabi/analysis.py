import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibration_libs.analysis import fit_oscillation
from qualibration_libs.data import convert_IQ_to_V
from quam_config.instrument_limits import instrument_limits
from scipy.signal import correlate

from qualibrate import QualibrationNode


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    opt_amp_prefactor: float
    opt_amp: float
    operation: str
    outcome: str
    fit_quality: float = None


def compute_snr(signal: np.ndarray) -> float:
    """
    Compute the signal-to-noise ratio (SNR) for a 1D array.
    SNR is defined as (max(signal) - min(signal)) / std(noise),
    where noise is estimated as the residual after subtracting a smoothed version of the signal.
    """
    if signal.size < 2:
        return 0.0
    from scipy.ndimage import uniform_filter1d
    smooth = uniform_filter1d(signal, size=max(3, signal.size // 10))
    noise = signal - smooth
    noise_std = np.std(noise)
    if noise_std == 0:
        return 0.0
    return (np.max(signal) - np.min(signal)) / noise_std


def should_fit_qubits(ds: xr.Dataset, signal_key: str = "state") -> dict[str, bool]:
    """
    Evaluate fit-worthiness of each qubit based on autocorrelation of its signal.
    Smooth structured signals (like Rabi) show slow autocorrelation decay.
    Spiky/noisy signals decay immediately.

    Returns:
    --------
    dict[str, bool]: {qubit: True/False}
    """
    result = {}

    for q in ds.qubit.values:
        try:
            s = ds[signal_key].sel(qubit=q).squeeze().values
            if s.ndim == 2:
                s = s.mean(axis=0)

            ptp = np.ptp(s)
            if ptp == 0 or np.isnan(ptp):
                result[q] = False
                continue

            s_norm = (s - np.mean(s)) / (np.std(s) + 1e-9)
            ac = correlate(s_norm, s_norm, mode='full')
            ac = ac[ac.size // 2:]  # take only positive lags
            ac /= ac[0]  # normalize

            # Check lag-1 and lag-2 autocorrelation
            r1, r2 = ac[1], ac[2]

            result[q] = (r1 > 0.8 and r2 > 0.5)

        except Exception:
            result[q] = False

    return result


def has_chevron_modulation(sig_2d: np.ndarray, threshold: float = 0.4) -> bool:
    """
    Determines whether the signal exhibits chevron-like modulation.
    This is detected via variation in peak-to-peak amplitude across the amplitude axis.
    
    Parameters:
    -----------
    sig_2d : np.ndarray
        2D array of signal values
    threshold : float
        Minimum ratio of modulation range to average peak-to-peak amplitude to consider as valid modulation
        
    Returns:
    --------
    bool
        True if significant modulation is detected, False otherwise
    """
    if sig_2d.ndim != 2:
        return False
    amp_axis = 1
    ptps = np.ptp(sig_2d, axis=0)
    modulation_range = np.ptp(ptps)
    avg_ptp = np.mean(ptps)
    return (modulation_range / (avg_ptp + 1e-12)) > threshold


def _determine_detuning_direction(fit: xr.Dataset, qubit: str, node=None) -> str:
    """
    Determines whether the detuning is positive or negative based on the drive and qubit frequencies if available,
    otherwise falls back to fit-based logic.
    Returns 'positive' or 'negative' detuning direction.
    """
    try:
        drive_freq = None
        qubit_freq = None
        # Try to use drive and qubit frequencies from node if available
        if node is not None and hasattr(node, "namespace") and "qubits" in node.namespace:
            qubits_dict = node.namespace["qubits"]
            q_obj = None
            # Always search by id
            for v in (qubits_dict.values() if isinstance(qubits_dict, dict) else qubits_dict):
                if hasattr(v, 'id') and getattr(v, 'id', None) == qubit:
                    q_obj = v
                    break
            if q_obj is not None:
                drive_freq = getattr(q_obj.xy, "RF_frequency", None)
                qubit_freq = getattr(q_obj, "f_01", None)
        if drive_freq is not None and qubit_freq is not None:
            detuning = drive_freq - qubit_freq
            direction = "positive" if detuning > 0 else "negative"
            return direction
        # Fallback: use fit-based logic
        fit_params = fit.fit.sel(qubit=qubit).values
        fit_vals = fit.fit.sel(qubit=qubit, fit_vals=slice(None)).coords['fit_vals'].values
        fit_dict = {k: v for k, v in zip(fit_vals, fit_params)}
        phase = float(fit_dict.get('phi', float('nan')))
        freq = float(fit_dict.get('f', float('nan')))
        amp = float(fit_dict.get('a', float('nan')))
        phase_norm = (phase + np.pi) % (2 * np.pi) - np.pi
        if abs(phase_norm) < (np.pi / 2):
            direction = "positive" if freq > 0 else "negative"
        else:
            direction = "positive" if phase_norm < 0 else "negative"
        return direction
    except (AttributeError, KeyError, ValueError):
        return ""


def get_fit_outcome(
    nan_success: bool,
    amp_success: bool,
    opt_amp: float,
    max_amplitude: float,
    fit_quality: float = None,
    min_fit_quality: float = 0.5,
    min_amplitude: float = 0.01,
    max_amp_prefactor: float = 10.0,
    amp_prefactor: float = None,
    snr: float = None,
    snr_min: float = 2.5,
    should_fit: bool = True,
    is_1d_dataset: bool = True,
    has_structure: bool = True,
    fit: xr.Dataset = None,
    qubit: str = None,
    node: object = None,
) -> str:

    if not nan_success:
        return "Fit parameters are invalid (NaN values detected)"

    # Only apply noise check for 1D datasets
    if is_1d_dataset and opt_amp < min_amplitude:
        if fit_quality is not None and fit_quality < min_fit_quality:
            return f"Poor fit quality (R² = {fit_quality:.3f} < {min_fit_quality}). There is too much noise in the data, consider increasing averaging or shot count"
        else:
            return f"There is too much noise in the data, consider increasing averaging or shot count"
    
    if not has_structure and not is_1d_dataset:
        return "No chevron modulation detected. Please check the drive frequency and amplitude sweep range"
        
    if snr is not None and snr < snr_min:
        return f"SNR too low (SNR = {snr:.2f} < {snr_min})"
    
    if not amp_success:
        detuning_direction = _determine_detuning_direction(fit, qubit, node=node) if fit is not None and qubit is not None else ""
        direction_str = f"{detuning_direction} detuning" if detuning_direction else ""
        return f"The drive frequency is off-resonant with high {direction_str}, please adjust the drive frequency"
    
    if amp_prefactor is not None and abs(amp_prefactor) > max_amp_prefactor:
        return f"Amplitude prefactor too large (|{amp_prefactor:.2f}| > {max_amp_prefactor})"

    if not should_fit and is_1d_dataset:
        return "There is too much noise in the data, consider increasing averaging or shot count"

    # Check fit quality first
    if fit_quality is not None and fit_quality < min_fit_quality:
        # For 1D datasets, check if the poor fit quality is due to too many oscillations
        if is_1d_dataset and fit is not None and qubit is not None and 'fit' in fit and hasattr(fit.fit, 'sel'):
            try:
                freq = float(fit.fit.sel(qubit=qubit, fit_vals='f').values)
                if freq > 1.0:
                    return "Fit quality poor due to large pulse amplitude range, consider a smaller range"
            except (ValueError, KeyError, AttributeError):
                pass
        return f"Poor fit quality (R² = {fit_quality:.3f} < {min_fit_quality})"
    
    return "successful"


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
        s_amp = f"The calibrated {fit_results[q]['operation']} amplitude: {1e3 * fit_results[q]['opt_amp']:.2f} mV (x{fit_results[q]['opt_amp_prefactor']:.2f})\n "
        if fit_results[q]["outcome"] == "successful":
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += f" FAIL! Reason: {fit_results[q]['outcome']}\n"
        # Add fit quality information if available
        if fit_results[q].get('fit_quality') is not None:
            s_qubit += f"Fit quality (R²): {fit_results[q]['fit_quality']:.3f}\n"
        log_callable(s_qubit + s_amp)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    full_amp = np.array(
        [ds.amp_prefactor * q.xy.operations[node.parameters.operation].amplitude for q in node.namespace["qubits"]]
    )
    ds = ds.assign_coords(full_amp=(["qubit", "amp_prefactor"], full_amp))
    ds.full_amp.attrs = {"long_name": "pulse amplitude", "units": "V"}
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
    if node.parameters.max_number_pulses_per_sweep == 1:
        ds_fit = ds.sel(nb_of_pulses=1)
        # Fit the power Rabi oscillations
        if node.parameters.use_state_discrimination:
            fit_vals = fit_oscillation(ds_fit.state, "amp_prefactor")
        else:
            fit_vals = fit_oscillation(ds_fit.I, "amp_prefactor")

        ds_fit = xr.merge([ds, fit_vals.rename("fit")])
    else:
        ds_fit = ds
        # Get the average along the number of pulses axis to identify the best pulse amplitude
        if node.parameters.use_state_discrimination:
            ds_fit["data_mean"] = ds.state.mean(dim="nb_of_pulses")
        else:
            ds_fit["data_mean"] = ds.I.mean(dim="nb_of_pulses")
        if (ds.nb_of_pulses.data[0] % 2 == 0 and node.parameters.operation == "x180") or (
            ds.nb_of_pulses.data[0] % 2 != 0 and node.parameters.operation != "x180"
        ):
            ds_fit["opt_amp_prefactor"] = ds_fit["data_mean"].idxmin(dim="amp_prefactor")
        else:
            ds_fit["opt_amp_prefactor"] = ds_fit["data_mean"].idxmax(dim="amp_prefactor")

    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    limits = [instrument_limits(q.xy) for q in node.namespace["qubits"]]
    if node.parameters.max_number_pulses_per_sweep == 1:
        # Process the fit parameters to get the right amplitude
        phase = fit.fit.sel(fit_vals="phi") - np.pi * (fit.fit.sel(fit_vals="phi") > np.pi / 2)
        factor = (np.pi - phase) / (2 * np.pi * fit.fit.sel(fit_vals="f"))
        fit = fit.assign({"opt_amp_prefactor": factor})
        fit.opt_amp_prefactor.attrs = {
            "long_name": "factor to get a pi pulse",
            "units": "Hz",
        }
        current_amps = xr.DataArray(
            [q.xy.operations[node.parameters.operation].amplitude for q in node.namespace["qubits"]],
            coords=dict(qubit=fit.qubit.data),
        )
        opt_amp = factor * current_amps
        fit = fit.assign({"opt_amp": opt_amp})
        fit.opt_amp.attrs = {"long_name": "x180 pulse amplitude", "units": "V"}
    else:
        current_amps = xr.DataArray(
            [q.xy.operations[node.parameters.operation].amplitude for q in node.namespace["qubits"]],
            coords=dict(qubit=fit.qubit.data),
        )
        fit = fit.assign({"opt_amp": fit.opt_amp_prefactor * current_amps})
        fit.opt_amp.attrs = {
            "long_name": f"{node.parameters.operation} pulse amplitude",
            "units": "V",
        }

    # Assess whether the fit was successful or not
    nan_success = ~(np.isnan(fit.opt_amp_prefactor) | np.isnan(fit.opt_amp))
    amp_success = fit.opt_amp < limits[0].max_x180_wf_amplitude
    
    # Compute SNR for each qubit
    snrs = []
    for q in fit.qubit.values:
        if hasattr(fit, "I") and hasattr(fit.I, "sel"):
            signal = fit.I.sel(qubit=q).values
        elif hasattr(fit, "state") and hasattr(fit.state, "sel"):
            signal = fit.state.sel(qubit=q).values
        else:
            signal = np.zeros(2)
        snrs.append(compute_snr(signal))
    fit = fit.assign_coords(snr=("qubit", snrs))
    fit.snr.attrs = {"long_name": "signal-to-noise ratio", "units": ""}

    # Check if signals are worth fitting
    signal_key = "state" if hasattr(fit, "state") else "I"
    fit_flags = should_fit_qubits(fit, signal_key=signal_key)

    # Get outcomes for each qubit
    outcomes = []
    for i, q in enumerate(fit.qubit.values):
        fit_quality = None
        if 'fit' in fit and hasattr(fit.fit, 'attrs'):
            if 'r_squared' in fit.fit.attrs:
                fit_quality = float(fit.fit.attrs['r_squared'])
            
        # Check for chevron modulation in 2D case
        has_structure = True
        if node.parameters.max_number_pulses_per_sweep > 1:
            if hasattr(fit, "I") and hasattr(fit.I, "sel"):
                signal = fit.I.sel(qubit=q).values
            elif hasattr(fit, "state") and hasattr(fit.state, "sel"):
                signal = fit.state.sel(qubit=q).values
            else:
                signal = np.zeros((2, 2))
            has_structure = has_chevron_modulation(signal)
            
        outcome = get_fit_outcome(
            nan_success=bool(nan_success.sel(qubit=q).values),
            amp_success=bool(amp_success.sel(qubit=q).values),
            opt_amp=float(fit.sel(qubit=q).opt_amp.values),
            max_amplitude=float(limits[i].max_x180_wf_amplitude),
            fit_quality=fit_quality,
            amp_prefactor=float(fit.sel(qubit=q).opt_amp_prefactor.values),
            snr=float(fit.sel(qubit=q).snr.values),
            should_fit=fit_flags[q],
            is_1d_dataset=node.parameters.max_number_pulses_per_sweep == 1,
            has_structure=has_structure,
            fit=fit,
            qubit=q,
            node=node,
        )
        outcomes.append(outcome)
    fit = fit.assign_coords(outcome=("qubit", outcomes))
    fit.outcome.attrs = {"long_name": "fit outcome", "units": ""}

    fit_results = {
        q: FitParameters(
            opt_amp_prefactor=fit.sel(qubit=q).opt_amp_prefactor.values.__float__(),
            opt_amp=fit.sel(qubit=q).opt_amp.values.__float__(),
            operation=node.parameters.operation,
            outcome=fit.sel(qubit=q).outcome.values.__str__(),
            fit_quality=fit_quality if fit_quality is not None else None,
        )
        for q in fit.qubit.values
    }
    return fit, fit_results