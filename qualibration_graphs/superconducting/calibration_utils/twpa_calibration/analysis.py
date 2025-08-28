import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from qualang_tools.units import unit
from collections import defaultdict
from qualibrate import QualibrationNode
u = unit(coerce_to_integer=True)


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
    full_freq = np.array(
        [ds.detuning + q.resonator.RF_frequency for q in node.namespace["qubits"]]
    )  # HARD coded, need to do this properly
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    # Normalize the IQ_abs with respect to the amplitude axis
    ds = ds.assign({"IQ_abs_norm": ds["IQ_abs"] / ds.IQ_abs.mean(dim=["detuning"])})
    ds = process_raw_dataset_per_twpa(ds, node.namespace["twpa_group"], node)
    return ds

def process_raw_dataset_per_twpa(raw_data: xr.Dataset, twpa_group: dict, node):
    """
    Run SNR / gain processing per TWPA, average over qubits,
    and return dataset that contains both the original raw data
    and new twpa-dim variables and coords.
    """
    pump_freq_axis = raw_data["pump_frequency"].data
    pump_amp_axis = raw_data["pump_amp"].data

    twpa_results = []
    twpa_ids = []

    for twpa_id, qubits in twpa_group.items():
        ds = raw_data.sel(qubit=qubits)

        # SNR and gain calculations per TWPA
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

        snr_avg = snr.mean("qubit")
        snr_db = 20 * np.log10(snr_avg)
        snr_pump_off_db = snr_db.sel(pump_amp=0.0, method="nearest")
        snr_delta_db = snr_db - snr_pump_off_db

        signal_lin = ds["I"].mean("detuning").mean("qubit")
        signal_db = xr.apply_ufunc(
            u.volts2dBm, signal_lin, vectorize=True, dask="parallelized", output_dtypes=[float]
        )
        signal_pump_off_db = signal_db.sel(pump_amp=0, method="nearest")
        gain_db = signal_db - signal_pump_off_db

        ds_result = xr.Dataset(
            dict(
                snr_db=snr_db,
                snr_pump_off_db=snr_pump_off_db,
                snr_delta_db=snr_delta_db,
                signal_db=signal_db,
                signal_pump_off_db=signal_pump_off_db,
                gain_db=gain_db,
            )
        )

        twpa_results.append(ds_result)
        twpa_ids.append(twpa_id)

    #stack all TWPA results along new 'twpa' dim
    twpa_data = xr.concat(twpa_results, dim="twpa").assign_coords(twpa=twpa_ids)

    # attach per-TWPA pump coordinates
    full_pump_freqs = []
    pump_powers = []
    for twpa_id in twpa_ids:
        pump_rf = node.machine.twpas[twpa_id].pump.RF_frequency
        full_scale_power_dbm = node.machine.twpas[twpa_id].pump.opx_output.full_scale_power_dbm
        full_pump_freqs.append(pump_freq_axis + pump_rf)
        pump_powers.append(_amp_to_dbm(full_scale_power_dbm, amp=pump_amp_axis, offset_db=0.0))

    twpa_data = twpa_data.assign_coords(
        full_pump_freq=(("twpa", "pump_frequency"), np.vstack(full_pump_freqs)),
        pump_power_dBm=(("twpa", "pump_amp"), np.vstack(pump_powers)),
    )
    twpa_data["full_pump_freq"].attrs = {"long_name": "TWPA pump RF frequency", "units": "Hz"}
    twpa_data["pump_power_dBm"].attrs = {"long_name": "TWPA pump power", "units": "dBm"}

    # return original + new results together
    ds_out = raw_data.copy()
    for var in twpa_data.data_vars:
        ds_out[var] = twpa_data[var]

    # also bring in coords
    ds_out = ds_out.assign_coords(
        full_pump_freq=twpa_data["full_pump_freq"],
        pump_power_dBm=twpa_data["pump_power_dBm"],
        twpa=twpa_data["twpa"],
    )

    return ds_out


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


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """
    Extract relevant fit parameters (max gain and max ΔSNR) per TWPA.
    Report maxima at full_pump_freq and pump_power_dBm coordinates.
    """
    twpa_ids = fit.coords["twpa"].values
    fit_results = {}

    snr_max_db = []
    snr_max_freq = []
    snr_max_power = []
    gain_max_db = []
    gain_max_freq = []
    gain_max_power = []
    success_flags = []

    for twpa in twpa_ids:
        fit_twpa = fit.sel(twpa=twpa)

        #Max gain
        gain_stacked = fit_twpa["gain_db"].stack(z=fit_twpa["gain_db"].dims)
        gain_imax = gain_stacked.argmax().item()
        gain_coords = {dim: gain_stacked.isel(z=gain_imax)[dim].item()
                       for dim in fit_twpa["gain_db"].dims}
        gain_max_val = fit_twpa["gain_db"].sel(gain_coords).item()

        # convert to physical coords
        gain_max_freq_val = fit_twpa["full_pump_freq"].sel(pump_frequency=gain_coords["pump_frequency"]).item()
        gain_max_power_val = fit_twpa["pump_power_dBm"].sel(pump_amp=gain_coords["pump_amp"]).item()

        # Max ΔSNR
        snr_stacked = fit_twpa["snr_delta_db"].stack(z=fit_twpa["snr_delta_db"].dims)
        snr_imax = snr_stacked.argmax().item()
        snr_coords = {dim: snr_stacked.isel(z=snr_imax)[dim].item()
                      for dim in fit_twpa["snr_delta_db"].dims}
        snr_max_val = fit_twpa["snr_delta_db"].sel(snr_coords).item()

        # convert to physical coords
        snr_max_freq_val = fit_twpa["full_pump_freq"].sel(pump_frequency=snr_coords["pump_frequency"]).item()
        snr_max_power_val = fit_twpa["pump_power_dBm"].sel(pump_amp=snr_coords["pump_amp"]).item()

        # SNR at optimal gain point 
        snr_at_optimal_point = fit_twpa["snr_delta_db"].sel(gain_coords).item()

        # store results (arrays)
        snr_max_db.append(snr_max_val)
        snr_max_freq.append(snr_max_freq_val)
        snr_max_power.append(snr_max_power_val)
        gain_max_db.append(gain_max_val)
        gain_max_freq.append(gain_max_freq_val)
        gain_max_power.append(gain_max_power_val)
        success = not (np.isnan(gain_max_freq_val) or np.isnan(gain_max_power_val))
        success_flags.append(success)

        # Build FitParameters per TWPA 
        fit_results[twpa] = FitParameters(
            success=success,
            max_gain=float(gain_max_val),
            optimal_pump_frequency=float(gain_max_freq_val),
            optimal_pump_power=float(gain_max_power_val),
            snr_at_optimal_point=float(snr_at_optimal_point),
        )

    # attach coords back to dataset
    fit = fit.assign_coords(
        snr_max_db=(("twpa",), snr_max_db),
        snr_max_freq=(("twpa",), snr_max_freq),
        snr_max_power=(("twpa",), snr_max_power),
        gain_max_db=(("twpa",), gain_max_db),
        gain_max_freq=(("twpa",), gain_max_freq),
        gain_max_power=(("twpa",), gain_max_power),
        success=(("twpa",), success_flags),
    )

    # add metadata
    fit["snr_max_db"].attrs = {"long_name": "Max ΔSNR", "units": "dB"}
    fit["snr_max_freq"].attrs = {"long_name": "Pump frequency at max ΔSNR", "units": "Hz"}
    fit["snr_max_power"].attrs = {"long_name": "Pump power at max ΔSNR", "units": "dBm"}
    fit["gain_max_db"].attrs = {"long_name": "Max Gain", "units": "dB"}
    fit["gain_max_freq"].attrs = {"long_name": "Pump frequency at max Gain", "units": "Hz"}
    fit["gain_max_power"].attrs = {"long_name": "Pump power at max Gain", "units": "dBm"}

    return fit, fit_results




def _amp_to_dbm(
    full_scale_power_dbm: float,
    amp: float | np.ndarray,
    R: float = 50.0,
    offset_db: float = 0.0,
) -> float | np.ndarray:
    """
    Convert a normalized amplitude 'amp' into output power in dBm,
    given the instrument full-scale output power (in dBm) into R.

    Parameters
    ----------
    full_scale_power_dbm : float
        Full-scale average output power (into R) in dBm.
    amp : float or array-like
        Normalized amplitude (0..1 typically).
    R : float
        Load resistance (ohms).
    offset_db : float
        Additive calibration/attenuation offset in dB.

    Returns
    -------
    dBm : float or ndarray
        Output power in dBm at the given amplitude.
    """
    v_peak = u.dBm2volts(full_scale_power_dbm)

    # Convert 'amp' to V_rms_out according to how the instrument defines amplitude
    amp = np.asarray(amp)
    v = amp * v_peak
    return u.volts2dBm(v) + offset_db


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
