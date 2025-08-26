from scipy.signal import savgol_filter
import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V

from qualibrate import QualibrationNode

SNR_MIN = 3  # dB
GAIN_MIN = 10  # dB


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    success: bool
    max_gain: float
    optimal_pump_frequency: float
    optimal_pump_power: float
    snr_at_optimal_point: float


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


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Processes the raw dataset by converting the 'I' and 'Q' quadratures to V, or adding the RF_frequency as a coordinate for instance."""

    # Convert the 'I' and 'Q' quadratures from demodulation units to V.
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    # Add the amplitude and phase to the raw dataset
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    # Add the RF frequency as a coordinate of the raw dataset
    full_freq = np.array([ds.detuning + q.resonator.RF_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    # Normalize the IQ_abs with respect to the amplitude axis
    ds = ds.assign({"IQ_abs_norm": ds["IQ_abs"] / ds.IQ_abs.mean(dim=["detuning"])})

    # Calculate the SNR over the detuning axis
    snr = xr.apply_ufunc(
        _get_snr,
        ds["I"],
        input_core_dims=[["detuning"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        kwargs={"mask_halfwidth": 2},
    )
    # Average the SNR over the multiplexed qubits
    snr_avg = snr.mean("qubit")
    # Convert the SNR to dB scale
    snr_db = 20 * np.log10(snr_avg)
    # Pump off SNR is the SNR at pump_amp=0
    snr_pump_off_db = snr_db.sel(pump_amp=0.0, method="nearest", tolerance=None)
    # Delta SNR is the difference between the SNR at pump_amp and the SNR at pump_amp=0
    snr_delta_db = snr_db - snr_pump_off_db

    # Calculate average gain over the mutiplexed qubits and detuning axis
    signal_lin = ds["I"].mean("detuning").mean("qubit")
    # Convert the signal to dBm
    signal_db = xr.apply_ufunc(_volt_to_dbm, signal_lin, vectorize=True, dask="parallelized", output_dtypes=[float])
    # Signal at pump_amp=0 is the average signal at pump_amp=0
    signal_pump_off_db = signal_db.sel(pump_amp=0, method="nearest")
    # Signal wnen pump is on
    signal_pump_on_db = signal_db.where(ds["pump_amp"] != 0, drop=True)
    # Gain is the difference between the signal at pump_amp and the signal at pump_amp=0
    gain_db = signal_db - signal_pump_off_db

    ds = ds.assign(
        snr_db=snr_db,
        snr_pump_off_db=snr_pump_off_db,
        snr_delta_db=snr_delta_db,
        signal_pump_off_db=signal_pump_off_db,
        signal_pump_on_db=signal_pump_on_db,
        gain_db=gain_db,
    )

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

    ds_fit = ds
    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset):
    """Add metadata to the dataset and fit results."""
    # Calculate max gain, and pump frequency and pump power corresponding to it
    gain_max = fit["gain_db"].max().item()
    gain_max_idx = fit["gain_db"].argmax()
    gain_max_coords = {dim: fit[dim].values[gain_max_idx[dim]] for dim in gain_max_idx.dims}

    # Calculate max SNR, and pump frequency and pump power corresponding to it
    snr_max = fit["snr_delta_db"].max()
    snr_max_idx = fit["snr_delta_db"].argmax()
    snr_max_coords = {dim: fit[dim].values[snr_max_idx[dim]] for dim in snr_max_idx.dims}
    snr_at_optimal_point = fit["snr_delta_db"].values1[gain_max_idx.values]
    fit.assign_coords(snr_max=snr_max.data)
    fit.assign_coords(snr_max_coords)

    fit = fit.assign_coords(
        snr_max_db=((), snr_max.item()),
        snr_max_freq=((), snr_max_coords.get("pump_frequency")),
        snr_max_amp=((), snr_max_coords.get("pump_amp")),
        gain_max_db=((), gain_max.item()),
        gain_max_freq=((), gain_max_coords.get("pump_frequency")),
        gain_max_amp=((), gain_max_coords.get("pump_amp")),
    )
    fit.snr_max_db.attrs = {"long_name": "Max SNR Δ", "units": "dB"}
    fit.gain_max_db.attrs = {"long_name": "Max Gain", "units": "dB"}

    nan_success = np.isnan(fit.gain_max_freq.data) | np.isnan(fit.gain_max_amp.data)

    # Populate the FitParameters class with fitted values
    fit_results = {
        FitParameters(
            success=False,
            max_gain=gain_max.item(),
            optimal_pump_frequency=gain_max_coords.get("pump_frequency"),
            optimal_pump_power=gain_max_coords.get("pump_amp"),
            snr_at_optimal_point=snr_at_optimal_point,
        )
    }
    return fit, fit_results


def _volt_to_dbm(v, R=50.0):
    """
    Convert a voltage amplitude into power in dBm assuming a resistive load.

    Parameters
    ----------
    v : array_like
        Voltage amplitude(s) in volts. Can be a scalar or NumPy array.
    R : float, optional
        Load resistance in ohms (default is 50 Ω for the OPX).

    Returns
    -------
    p_dbm : numpy.ndarray or float
        Power in dBm corresponding to the input voltage(s).

    Notes
    -----
    - Power is computed as P = V^2 / R (watts).
    - The result is converted to milliwatts and expressed in decibels (dBm):
        dBm = 10 * log10(P / 1 mW).
    - A floor value (1e-20 W) is enforced to avoid taking log(0).
    """

    v = np.asarray(v)
    p_w = (v**2) / R
    p_w = np.maximum(p_w, 1e-20)  # avoid log of zero
    p_dbm = 10 * np.log10(p_w * 1e3)
    return p_dbm


def _get_snr(spec, mask_halfwidth=2):
    """
    Very Crude function to estimate SNR. Ideally should be done by IQ blobs
    and state discrimination.
    Estimate SNR (dB) assuming the feature is a DIP.
    Excludes a small window around the minimum to form the baseline.

    spec: 1D array in linear units (e.g., volts), NOT dBm.
    mask_halfwidth: how many points on each side of the dip to exclude.
    """
    s = np.asarray(spec, dtype=float)
    if s.size == 0:
        return np.nan

    # Find dip and exclude a neighborhood around it from the baseline
    idx = np.argmin(s)
    lo = max(0, idx - mask_halfwidth)
    hi = min(s.size, idx + mask_halfwidth + 1)
    if s.size > (hi - lo):
        base = np.concatenate([s[:lo], s[hi:]])
    else:
        # fallback if the window covers the whole array
        base = s[np.arange(s.size) != idx]

    if base.size == 0:
        return np.nan

    # Signal = dip depth; Noise = baseline std
    signal = base.mean() - s[idx]
    noise = base.std(ddof=1)

    if signal <= 0 or noise <= 0:
        return -np.inf

    return signal / noise
