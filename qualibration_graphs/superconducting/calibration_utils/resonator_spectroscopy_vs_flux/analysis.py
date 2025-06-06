import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibration_libs.analysis import fit_oscillation
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from scipy.ndimage import gaussian_filter1d

from qualibrate import QualibrationNode


@dataclass
class FitParameters:
    """Stores the relevant node-specific fitted parameters used to update the state at the end of the node."""

    resonator_frequency: float
    frequency_shift: float
    min_offset: float
    idle_offset: float
    dv_phi0: float
    phi0_current: float
    m_pH: float
    outcome: str


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
        s_idle_offset = f"\tidle offset: {fit_results[q]['idle_offset'] * 1e3:.0f} mV | "
        s_min_offset = f"min offset: {fit_results[q]['min_offset'] * 1e3:.0f} mV | "
        s_freq = f"Resonator frequency: {1e-9 * fit_results[q]['resonator_frequency']:.3f} GHz | "
        s_shift = f"(shift of {1e-6 * fit_results[q]['frequency_shift']:.0f} MHz)\n"
        if fit_results[q]["outcome"] == "successful":
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += f" FAIL! Reason: {fit_results[q]['outcome']}\n"
        log_callable(s_qubit + s_idle_offset + s_min_offset + s_freq + s_shift)


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
    # Add the current axis of each qubit to the dataset coordinates for plotting
    current = ds.flux_bias / node.parameters.input_line_impedance_in_ohm
    ds = ds.assign_coords({"current": (["flux_bias"], current.data)})
    ds.current.attrs["long_name"] = "Current"
    ds.current.attrs["units"] = "A"
    # Add attenuated current to dataset
    attenuation_factor = 10 ** (-node.parameters.line_attenuation_in_db / 20)
    attenuated_current = ds.current * attenuation_factor
    ds = ds.assign_coords({"attenuated_current": (["flux_bias"], attenuated_current.values)})
    ds.attenuated_current.attrs["long_name"] = "Attenuated Current"
    ds.attenuated_current.attrs["units"] = "A"
    return ds


def has_resonator_trace(
    ds: xr.Dataset,
    qubit: str,
    var_name: str = "IQ_abs",
    freq_dim: str = "detuning",
    flux_dim: str = "flux_bias",
    smooth_sigma: float = 1.5,
    dip_threshold: float = 0.01,
    gradient_threshold: float = 0.001
) -> bool:
    """
    Improved detector for whether a resonator-like frequency trace exists.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing spectroscopy data.
    qubit : str
        Qubit to test.
    var_name : str
        Data variable to use (e.g. IQ_abs).
    freq_dim : str
        Frequency axis name.
    flux_dim : str
        Flux bias axis name.
    smooth_sigma : float
        Gaussian smoothing factor (in index units).
    dip_threshold : float
        Minimum depth of the min trace across flux to qualify as a resonance.
    gradient_threshold : float
        Minimum slope variation to confirm feature isn't flat.

    Returns
    -------
    bool
        True if resonator trace is detected, else False.
    """
    # Extract and normalize
    da = ds[var_name].sel(qubit=qubit)
    da_norm = da / da.mean(dim=freq_dim)

    # Get min trace across frequency sweep
    min_trace = da_norm.min(dim=freq_dim).values

    # Smooth to suppress noise
    min_trace_smooth = gaussian_filter1d(min_trace, sigma=smooth_sigma)

    # Calculate peak-to-peak dip depth and variation
    dip_depth = np.max(min_trace_smooth) - np.min(min_trace_smooth)
    gradient = np.max(np.abs(np.gradient(min_trace_smooth)))

    return (dip_depth > dip_threshold) and (gradient > gradient_threshold)


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Robustly fit the resonance for each qubit as a function of flux.
    Outcome logic:
      - If peak_freq is mostly NaN or nearly constant, outcome = 'No peaks were detected...'
      - If fit amplitude is below threshold, outcome = 'No oscillations were detected...'
      - Otherwise, outcome = 'successful' or other error.
    """
    # Find the minimum of each frequency line to follow the resonance vs flux
    peak_freq = ds.IQ_abs.idxmin(dim="detuning")
    outcomes = {}
    fit_results_da = None
    # Device-agnostic thresholds
    nan_frac_thresh = 0.8  # If >80% NaN, call it 'no peaks'
    flat_std_rel_thresh = 1e-6  # If std < 1e-6 * mean, call it 'no peaks'
    amp_rel_thresh = 0.01  # If fit amplitude < 1% of median freq, call it 'no oscillations'
    # For each qubit, check peak_freq quality
    qubit_outcomes = {}
    for q in ds.qubit.values:
        # First check if there's a resonator trace
        if not has_resonator_trace(ds, q):
            qubit_outcomes[q] = "no_peaks"
            continue
            
        pf = peak_freq.sel(qubit=q)
        pf_vals = pf.values
        n_nan = np.sum(np.isnan(pf_vals))
        frac_nan = n_nan / pf_vals.size
        pf_valid = pf_vals[~np.isnan(pf_vals)]
        if pf_valid.size == 0 or frac_nan > nan_frac_thresh:
            qubit_outcomes[q] = "no_peaks"
        else:
            pf_std = np.nanstd(pf_valid)
            pf_mean = np.nanmean(np.abs(pf_valid))
            if pf_mean == 0 or pf_std < flat_std_rel_thresh * pf_mean:
                qubit_outcomes[q] = "no_peaks"
            else:
                qubit_outcomes[q] = "fit"

    # Only fit oscillation for qubits with valid peaks
    fit_results_da = fit_oscillation(peak_freq.dropna(dim="flux_bias"), "flux_bias")
    fit_results_ds = xr.merge([fit_results_da.rename("fit_results"), peak_freq.rename("peak_freq")])
    # Pass outcomes to _extract_relevant_fit_parameters
    fit_dataset, fit_results = _extract_relevant_fit_parameters(fit_results_ds, node, qubit_outcomes, amp_rel_thresh)
    return fit_dataset, fit_results


def get_fit_outcome(
    freq_shift: float,
    flux_min: float,
    flux_idle: float,
    frequency_span_in_mhz: float,
    snr: float = None,
    snr_min: float = 2,
    has_oscillations: bool = True,
    has_anticrossings: bool = False,
) -> str:
    """
    Returns the outcome string for a given fit result.

    Parameters
    ----------
    freq_shift : float
        Frequency shift in Hz
    flux_min : float
        Minimum flux offset in V
    flux_idle : float
        Idle flux offset in V
    frequency_span_in_mhz : float
        Maximum allowed frequency span in MHz
    snr : float, optional
        Signal to noise ratio
    snr_min : float, default=2
        Minimum required SNR
    has_oscillations : bool, default=True
        Whether oscillations were detected
    has_anticrossings : bool, default=False
        Whether anti-crossings were detected

    Returns
    -------
    str
        Outcome string describing the fit result
    """
    if not has_oscillations:
        return "No oscillations were detected, consider checking that the flux line is connected or increase the flux range"
    
    if has_anticrossings:
        return "Anti-crossings were detected, consider adjusting the flux range or checking the device setup"
    
    snr_low = snr is not None and snr < snr_min
    if snr_low:
        return "The SNR isn't large enough, consider increasing the number of shots"
    
    if np.isnan(freq_shift) or np.isnan(flux_min) or np.isnan(flux_idle):
        return "No peaks were detected, consider changing the frequency range"
    
    if np.abs(freq_shift) >= frequency_span_in_mhz * 1e6:
        return f"Frequency shift {1e-6 * freq_shift:.0f} MHz exceeds span {frequency_span_in_mhz} MHz"
    
    return "successful"


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode, qubit_outcomes: dict, amp_rel_thresh: float):
    """Add metadata to the fit dataset and fit result dictionary, using robust outcome logic."""
    # Ensure that the phase is between -pi and pi
    flux_idle = -fit.sel(fit_vals="phi")
    flux_idle = np.mod(flux_idle + np.pi, 2 * np.pi) - np.pi
    # converting the phase phi from radians to voltage
    flux_idle = flux_idle / fit.sel(fit_vals="f") / 2 / np.pi
    fit = fit.assign_coords(idle_offset=("qubit", flux_idle.fit_results.data))
    fit.idle_offset.attrs = {"long_name": "idle flux bias", "units": "V"}
    # finding the location of the minimum frequency flux point
    flux_min = flux_idle + ((flux_idle < 0) - 0.5) / fit.sel(fit_vals="f")
    flux_min = flux_min * (np.abs(flux_min) < 0.5) + 0.5 * (flux_min > 0.5) - 0.5 * (flux_min < -0.5)
    fit = fit.assign_coords(flux_min=("qubit", flux_min.fit_results.data))
    fit.flux_min.attrs = {"long_name": "minimum frequency flux bias", "units": "V"}
    # finding the frequency as the sweet spot flux
    full_freq = np.array([q.resonator.RF_frequency for q in node.namespace["qubits"]])
    freq_shift = fit.peak_freq.sel(flux_bias=flux_idle.fit_results, method="nearest")
    fit = fit.assign_coords(freq_shift=("qubit", freq_shift.data))
    fit.freq_shift.attrs = {"long_name": "frequency shift", "units": "Hz"}
    fit = fit.assign_coords(sweet_spot_frequency=("qubit", freq_shift.data + full_freq))
    fit.sweet_spot_frequency.attrs = {
        "long_name": "sweet spot frequency",
        "units": "Hz",
    }
    # m_pH
    attenuation_factor = 10 ** (-node.parameters.line_attenuation_in_db / 20)
    m_pH = (
        1e12
        * 2.068e-15
        / (1 / fit.sel(fit_vals="f"))
        / node.parameters.input_line_impedance_in_ohm
        * attenuation_factor
    )
    # Calculate outcomes for each qubit
    outcomes = []
    fit_results = {}
    for q in fit.qubit.values:
        # Default outcome
        outcome = "successful"
        if qubit_outcomes.get(q) == "no_peaks":
            outcome = "No peaks were detected, consider changing the frequency range"
            amp = np.nan
        else:
            # Extract fit amplitude
            amp = float(fit.sel(fit_vals="a", qubit=q).fit_results.data)
            pf_valid = fit.peak_freq.sel(qubit=q).values
            pf_valid = pf_valid[~np.isnan(pf_valid)]
            pf_median = np.median(np.abs(pf_valid)) if pf_valid.size > 0 else 1.0
            if np.abs(amp) < amp_rel_thresh * pf_median:
                outcome = "No oscillations were detected, consider checking that the flux line is connected or increase the flux range"
        freq_shift_val = float(freq_shift.sel(qubit=q).values) if outcome == "successful" else np.nan
        flux_min_val = float(flux_min.sel(qubit=q).fit_results.data) if outcome == "successful" else np.nan
        flux_idle_val = float(flux_idle.sel(qubit=q).fit_results.data) if outcome == "successful" else np.nan
        fit_results[q] = FitParameters(
            resonator_frequency=float(fit.sweet_spot_frequency.sel(qubit=q).values) if outcome == "successful" else np.nan,
            frequency_shift=freq_shift_val,
            min_offset=flux_min_val,
            idle_offset=flux_idle_val,
            dv_phi0=1 / fit.sel(fit_vals="f", qubit=q).fit_results.data if outcome == "successful" else np.nan,
            phi0_current=1 / fit.sel(fit_vals="f", qubit=q).fit_results.data * node.parameters.input_line_impedance_in_ohm * attenuation_factor if outcome == "successful" else np.nan,
            m_pH=m_pH.sel(qubit=q).fit_results.data if outcome == "successful" else np.nan,
            outcome=outcome,
        )
        outcomes.append(outcome)
    fit = fit.assign_coords(outcome=("qubit", outcomes))
    return fit, fit_results
