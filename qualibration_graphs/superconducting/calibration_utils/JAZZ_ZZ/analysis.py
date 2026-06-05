"""Analysis functions for JAZZ_ZZ calibration: fitting and extracting residual ZZ coupling.

Reference: arXiv:2402.18926 — Li et al., "Realization of High-Fidelity CZ Gate based on a
Double-Transmon Coupler", Sec. III.1.

JAZZ pulse sequence (as implemented in 19_zz_off_jazz.py)
---------------------------------------------------------
The ZZ Hamiltonian is H_ZZ = (ζ/4) σz¹σz², so the phase rate seen by Q1's superposition
depends on Q2's state:
    Q2 = |0⟩  →  phase rate = −ζ/2  (σz eigenvalue = +1 for |0⟩)
    Q2 = |1⟩  →  phase rate = +ζ/2  (σz eigenvalue = −1 for |1⟩)
ZZ is ALWAYS active; only its sign changes with Q2's state. The purpose of the echo
is NOT to switch ZZ on — it is to cancel single-qubit phase offsets while keeping
both ZZ contributions accumulating in the same direction.

Step 1 — x90 on Q1 (measured qubit):
    Q1 is prepared in superposition |+⟩ = (|0⟩ + |1⟩) / √2.
    Q2 is in |0⟩ after reset.

Step 2 — First coupler pulse for duration t_single:
    Q2 is in |0⟩. ZZ is active with rate −ζ/2.
    Phase accumulated on Q1:  φ₁ = −(ζ/2) · t_single.

Step 3 — x180 echo on both Q1 and Q2:
    Xπ on Q1: inverts the sign of the already-accumulated phase (φ₁ → +ζ/2 · t_single).
    Xπ on Q2: flips Q2 from |0⟩ → |1⟩, switching the ZZ rate from −ζ/2 to +ζ/2.
    Combined effect: single-qubit phase offsets (bare qubit frequencies etc.) cancel,
    while the ZZ phase from step 2 and step 5 accumulate additively.

Step 4 — Frame rotation on Q1 by ωb · t_single  (artificial detuning):
    An intentional phase ωb · t_single is added to Q1's rotating frame, where
    ωb = artificial_detuning_in_mhz [MHz].
    This shifts the oscillation away from DC so the fitting is more reliable.
    Necessary because ζ near the idle point is tiny (~kHz), giving a very slow
    oscillation that is hard to resolve within the qubit coherence time.

Step 5 — Second coupler pulse for duration t_single:
    Q2 is now in |1⟩. ZZ is active with rate +ζ/2.
    Phase accumulated on Q1:  φ₂ = +(ζ/2) · t_single.

Step 6 — x90 on Q1 → measure:
    Population oscillates as  P = (1 − cos φ) / 2.

Phase and frequency summary
---------------------------
Both intervals contribute ZZ phase in the same direction after the echo:
    φ_ZZ = (ζ/2)·t_single  +  (ζ/2)·t_single  =  ζ · t_single

Total phase including artificial detuning:
    φ = ζ · t_single + ωb · t_single = (ζ + ωb) · t_single

Fitting  cos(2π · ωm · t_single)  to the signal gives:
    ωm = ζ + ωb                          ← stored as ``jeff_raw`` [MHz]

True residual ZZ coupling:
    ζ = ωm − ωb = jeff_raw − artificial_detuning

The optimal coupler amplitude is where |ζ| is minimised (ζ → 0),
i.e. where  jeff_raw ≈ artificial_detuning.

Variable reference
------------------
``jeff_raw``   = ωm = ζ + ωb  [MHz]  (measured oscillation frequency, NOT ζ itself)
``jeff_smooth``                        (Savitzky–Golay smoothed version of jeff_raw)
``gamma_raw``  = γ  [1/µs]    (decay rate of the damped cosine envelope)
``tau_raw``    = τ = 1/γ [µs] (decay time constant)
residual       = |ζ| = |jeff_raw − artificial_detuning|  [MHz]
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


@dataclass
class FitResults:
    """Stores the relevant JAZZ_ZZ experiment fit parameters for a single qubit pair"""

    optimal_amplitude: float
    max_decay_time: float  # maximum decay time constant (1/gamma) in µs
    max_decay_time_amplitude: float  # amplitude where max decay time occurs
    success: bool


def damped_cosine(t, A, gamma, f, phi, C):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Damped cosine fitting function for JAZZ_ZZ oscillations.

    Args:
        t     : t_single [µs]  — single coupler pulse duration (sweep axis)
        f     : ωm [MHz]       — total oscillation frequency; ωm = ζ + ωb  (stored as jeff_raw)
        gamma : γ [1/µs]       — decay rate of the oscillation envelope
        A, phi, C              — amplitude, phase offset, vertical offset
    """
    return A * np.exp(-gamma * t) * np.cos(2 * np.pi * f * t + phi) + C


def log_fitted_results(fit_results: Dict[str, FitResults], log_callable=None):
    """
    Logs the JAZZ_ZZ fitted results for all qubit pairs.

    Parameters:
    -----------
    fit_results : Dict[str, FitResults]
        Dictionary containing FitResults for each qubit pair.
    log_callable : callable, optional
        Logger for logging the fitted results. If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for qp_name, fit_result in fit_results.items():
        s_qubit = f"Results for qubit pair {qp_name}: "
        s_amp = f"\tOptimal JAZZ_ZZ amplitude for minimum coupling: {fit_result.optimal_amplitude:.6f} a.u."
        s_tau = (
            f"\tMaximum decay time constant: {fit_result.max_decay_time:.6f} µs "
            f"at amplitude {fit_result.max_decay_time_amplitude:.6f} a.u."
        )

        if fit_result.success:
            s_qubit += "SUCCESS!\n"
        else:
            s_qubit += "FAIL!\n"

        log_message = s_qubit + s_amp + "\n" + s_tau

        log_callable(log_message)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """
    Process the raw dataset for JAZZ_ZZ analysis.

    Parameters:
    -----------
    ds : xr.Dataset
        Raw dataset from the experiment
    node : QualibrationNode
        The calibration node containing qubit pairs information

    Returns:
    --------
    xr.Dataset
        Processed dataset with additional coordinates
    """
    # Convert time from ns to µs for fitting
    time_us = ds.time.data * 1e-3
    ds = ds.assign_coords(time_us=("time", time_us))

    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitResults]]:
    """
    Fit the JAZZ_ZZ data by extracting effective coupling J_eff from oscillations.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the processed data.
    node : QualibrationNode
        The calibration node containing parameters and qubit pairs.

    Returns:
    --------
    Tuple[xr.Dataset, Dict[str, FitResults]]
        Dataset with fit results and dictionary of fit results for each qubit pair.
    """
    ds_fit = ds.groupby("qubit_pair").apply(lambda da: fit_jazz_zz_routine(da, node))

    # Extract the relevant fitted parameters
    ds_fit, fit_results = _extract_relevant_parameters(ds_fit, node)

    return ds_fit, fit_results


def fit_jazz_zz_routine(da, node):  # pylint: disable=too-many-statements
    """
    Extract effective coupling J_eff from JAZZ_ZZ oscillations for each amplitude.

    Parameters:
    -----------
    da : xr.DataArray
        Data array containing the oscillation data
    node : QualibrationNode
        The calibration node containing parameters

    Returns:
    --------
    xr.DataArray
        Data array with added fit results
    """
    if hasattr(da, "state_measured"):
        data = "state_measured"
    else:
        data = "I_measured"

    # Extract the data matrix with time along axis 0, amplitude along axis 1
    signal_da = da[data].squeeze()
    data_matrix = signal_da.transpose("time", "amp").values  # shape = (n_time, n_amp)
    flux_bias = da.amp.data  # amplitude values
    time_us = da.time_us.data  # t_single [µs]: single coupler pulse duration (sweep axis)

    # Fit a damped cosine to the oscillation vs t_single for each coupler amplitude.
    # Extracted frequency ωm = ζ + ωb  →  stored as jeff_raw  (see module docstring).
    jeff_raw = []  # ωm = ζ + ωb [MHz]
    gamma_raw = []  # γ [1/µs]
    fit_mask = []
    fitted_matrix = np.full(data_matrix.shape, np.nan)

    for i in range(data_matrix.shape[1]):
        ydata = data_matrix[:, i] - np.mean(data_matrix[:, i])

        try:
            popt, _ = curve_fit(
                damped_cosine,
                time_us,
                ydata,
                p0=[0.3, 1.0, 5.0, 0.0, 0.0],
                # bounds=([-np.inf, -np.inf, -np.inf, -np.pi, -np.inf], [np.inf, np.inf, np.inf, np.pi, np.inf]),
                maxfev=5000,
            )
            freq_mhz = popt[2]  # ωm [MHz] = ζ + ωb  (stored as jeff_raw)
            gamma_mhz = popt[1]  # γ [1/µs]
            jeff_raw.append(freq_mhz)
            gamma_raw.append(gamma_mhz)
            fit_mask.append(True)
            fitted_matrix[:, i] = damped_cosine(time_us, *popt)
        except RuntimeError:
            jeff_raw.append(0.0)
            gamma_raw.append(0.0)
            fit_mask.append(False)

    jeff_raw = np.array(jeff_raw)
    gamma_raw = np.array(gamma_raw)
    fit_mask = np.array(fit_mask)

    # Calculate decay time constants (τ = 1/γ) in microseconds
    # Convert gamma from MHz to 1/µs: τ = 1/γ [µs] = 1/(γ [MHz])
    tau_raw = np.zeros_like(gamma_raw)
    tau_raw[fit_mask & (gamma_raw > 0)] = 1.0 / gamma_raw[fit_mask & (gamma_raw > 0)]

    # Smooth only the valid (nonzero) portion if there are enough valid points
    if np.sum(fit_mask) >= 5:  # Need at least 5 points for window_length=5
        jeff_smooth = np.zeros_like(jeff_raw)
        gamma_smooth = np.zeros_like(gamma_raw)
        tau_smooth = np.zeros_like(tau_raw)
        jeff_smooth[fit_mask] = savgol_filter(jeff_raw[fit_mask], window_length=5, polyorder=3)
        gamma_smooth[fit_mask] = savgol_filter(gamma_raw[fit_mask], window_length=5, polyorder=3)
        # For tau, only smooth valid positive values
        valid_tau_mask = fit_mask & (tau_raw > 0)
        if np.sum(valid_tau_mask) >= 5:
            tau_smooth[valid_tau_mask] = savgol_filter(tau_raw[valid_tau_mask], window_length=5, polyorder=3)
        else:
            tau_smooth = tau_raw.copy()
    else:
        jeff_smooth = jeff_raw.copy()
        gamma_smooth = gamma_raw.copy()
        tau_smooth = tau_raw.copy()

    # Find the coupler amplitude where |ζ| = |ωm − ωb| is minimised (ζ → 0).
    valid_indices = np.where(fit_mask)[0]
    if len(valid_indices) > 0:
        coupling_deviation = np.abs(jeff_smooth[fit_mask] - node.parameters.artificial_detuning_in_mhz)
        min_coupling_idx = valid_indices[np.argmin(coupling_deviation)]
        optimal_amplitude = flux_bias[min_coupling_idx]

        # Find maximum decay time constant
        valid_tau_indices = np.where(fit_mask & (tau_smooth > 0))[0]
        if len(valid_tau_indices) > 0:
            max_tau_idx = valid_tau_indices[np.argmax(tau_smooth[valid_tau_indices])]
            max_decay_time = tau_smooth[max_tau_idx]
            max_decay_time_amplitude = flux_bias[max_tau_idx]
        else:
            max_decay_time = np.nan
            max_decay_time_amplitude = np.nan

        success = True
    else:
        optimal_amplitude = np.nan
        max_decay_time = np.nan
        max_decay_time_amplitude = np.nan
        success = False

    # Match fitted_state dim order to the measured signal (typically amp, time)
    fitted_state = xr.DataArray(
        fitted_matrix,
        dims=("time", "amp"),
        coords={"time": da.time, "amp": da.amp},
    ).transpose(*signal_da.dims)

    # Add results to data array
    da = da.assign(
        fitted_state=fitted_state,
        jeff_raw=("amp", jeff_raw),
        jeff_smooth=("amp", jeff_smooth),
        gamma_raw=("amp", gamma_raw),
        gamma_smooth=("amp", gamma_smooth),
        tau_raw=("amp", tau_raw),
        tau_smooth=("amp", tau_smooth),
        fit_mask=("amp", fit_mask),
        optimal_amplitude=optimal_amplitude,
        max_decay_time=max_decay_time,
        max_decay_time_amplitude=max_decay_time_amplitude,
        success=success,
    )

    return da


def _extract_relevant_parameters(
    ds_fit: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitResults]]:
    """
    Extract relevant fit parameters and create FitResults for each qubit pair.

    Parameters:
    -----------
    ds_fit : xr.Dataset
        Dataset containing the fit results from fit_jazz_zz_routine.
    node : QualibrationNode
        The calibration node containing parameters and qubit pairs.

    Returns:
    --------
    Tuple[xr.Dataset, Dict[str, FitResults]]
        Dataset with additional metadata and dictionary of FitResults for each qubit pair.
    """
    qubit_pairs = node.namespace["qubit_pairs"]

    # Add metadata attributes to the dataset
    if "optimal_amplitude" in ds_fit.data_vars:
        ds_fit.optimal_amplitude.attrs = {
            "long_name": "optimal JAZZ_ZZ amplitude for minimum coupling",
            "units": "a.u.",
        }
    if "max_decay_time" in ds_fit.data_vars:
        ds_fit.max_decay_time.attrs = {
            "long_name": "maximum decay time constant",
            "units": "µs",
        }
    if "max_decay_time_amplitude" in ds_fit.data_vars:
        ds_fit.max_decay_time_amplitude.attrs = {
            "long_name": "amplitude at maximum decay time",
            "units": "a.u.",
        }
    if "fitted_state" in ds_fit.data_vars:
        ds_fit.fitted_state.attrs = {"long_name": "fitted oscillation", "units": "a.u."}
    ds_fit["artificial_detuning"] = node.parameters.artificial_detuning_in_mhz
    if "jeff_raw" in ds_fit.data_vars:
        ds_fit.jeff_raw.attrs = {
            "long_name": "measured oscillation frequency",
            "units": "MHz",
        }
    if "jeff_smooth" in ds_fit.data_vars:
        ds_fit.jeff_smooth.attrs = {
            "long_name": "smoothed oscillation frequency",
            "units": "MHz",
        }
    if "gamma_raw" in ds_fit.data_vars:
        ds_fit.gamma_raw.attrs = {"long_name": "raw extracted decay rate", "units": "MHz"}
    if "gamma_smooth" in ds_fit.data_vars:
        ds_fit.gamma_smooth.attrs = {"long_name": "smoothed decay rate", "units": "MHz"}
    if "tau_raw" in ds_fit.data_vars:
        ds_fit.tau_raw.attrs = {"long_name": "raw decay time constant", "units": "µs"}
    if "tau_smooth" in ds_fit.data_vars:
        ds_fit.tau_smooth.attrs = {"long_name": "smoothed decay time constant", "units": "µs"}
    if "fit_mask" in ds_fit.data_vars:
        ds_fit.fit_mask.attrs = {"long_name": "successful fit mask", "units": "bool"}

    # Create FitResults for each qubit pair
    fit_results = {}
    for qp in qubit_pairs:
        qp_name = qp.name
        qp_data = ds_fit.sel(qubit_pair=qp_name)

        fit_results[qp_name] = FitResults(
            optimal_amplitude=float(qp_data.optimal_amplitude.values),
            max_decay_time=float(qp_data.max_decay_time.values),
            max_decay_time_amplitude=float(qp_data.max_decay_time_amplitude.values),
            success=bool(qp_data.success.values),
        )

    return ds_fit, fit_results
