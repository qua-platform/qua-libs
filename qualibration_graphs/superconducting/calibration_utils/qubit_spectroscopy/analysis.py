"""Analysis utilities for qubit spectroscopy calibration (v2).

Replaces the legacy single-pass `peaks_dips` path with an explicit Lorentzian
peak + linear-background `curve_fit` mirroring the resonator-single fitter.
Adds smart initial guess (Savgol smoothing + `find_peaks` with a noise-floor
prominence threshold), narrow-window refit, and R²/FWHM/contrast quality gates
so weak fits stop silently overwriting QUAM state.

The IQ rotation step (per-qubit `arctan2`-derived `iw_angle` aligning the
signal onto the rotated I-quadrature) is preserved from the legacy pipeline —
that's the qubit-spec-specific piece, not part of the resonator analysis.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from quam_config.instrument_limits import instrument_limits


# ---------------------------------------------------------------------------
# Fit model
# ---------------------------------------------------------------------------

def lorentzian_peak_linbg(f, f0, fwhm, amp, bg0, bg1):
    """Lorentzian peak (positive amp) with linear background.

    R(f) = [bg0 + bg1*(f - fc)] + amp / [1 + ((f - f0)/(fwhm/2))^2]

    `fc` is the mean of `f` so `bg0` is the baseline level near the peak,
    not a value that depends on the choice of absolute frequency offset.
    """
    fc = float(np.mean(f))
    return (bg0 + bg1 * (f - fc)) + amp / (1.0 + ((f - f0) / (fwhm / 2.0)) ** 2)


# ---------------------------------------------------------------------------
# Peak finder
# ---------------------------------------------------------------------------

def find_best_peak(smoothed, raw_residual_std, edge_fraction=0.04, prominence_factor=3.0):
    """Return (peak_idx, is_edge_only) for the highest inner local peak.

    `prominence_factor * raw_residual_std` sets the minimum prominence so we
    ignore wiggles that are not statistically above the per-point noise. If
    no inner peak is found we fall back to the global argmax with the edge
    flag set, so the caller can still reject the fit.
    """
    N = len(smoothed)
    edge = int(N * edge_fraction)
    prominence = max(prominence_factor * raw_residual_std, 1e-12)
    peaks, _ = find_peaks(smoothed, prominence=prominence)
    inner = [p for p in peaks if edge <= p <= N - 1 - edge]
    if inner:
        return int(inner[np.argmax(smoothed[inner])]), False
    return int(np.argmax(smoothed)), True


# ---------------------------------------------------------------------------
# Main fitter
# ---------------------------------------------------------------------------

def fit_qubit_peak(
    freqs,
    signal,
    *,
    window_fwhm_factor: float = 4.0,
    min_window_mhz: float = 5.0,
    refit_window_fwhm_factor: float = 8.0,
    min_refit_window_mhz: float = 20.0,
    detrend_window_mhz: float = 10.0,
    max_fwhm_mhz: float = 30.0,
    fwhm_floor_hz: float = 5e4,
    r2_threshold: float = 0.75,
    min_contrast: float = 0.05,
    edge_fraction: float = 0.04,
    smooth_window: int = 11,
):
    """Fit a Lorentzian peak with linear background to `signal(freqs)`.

    Pipeline: Savgol smooth → noise-floor-scaled `find_peaks` → smart f0/FWHM
    init via local detrend half-max → first `curve_fit` over ±max(2×FWHM, 2.5 MHz)
    → **wider refit** over ±max(4×FWHM, 10 MHz) using the first-pass values →
    quality gates (R² inside refit window, FWHM bound, contrast = amp / |baseline|).
    Frequencies are in Hz; signal is the rotated I-quadrature in V.

    The refit window is intentionally wider than the first-pass window: it
    constrains the linear-bg slope on a representative slice of flat baseline
    (a handful of FWHMs on each wing) and makes R² a meaningful number.

    `fwhm_floor_hz` is a hard lower bound for the fitted FWHM. Without it,
    `curve_fit` can collapse the Lorentzian onto a single noise spike when the
    data has no real peak, producing fake-narrow fits with absurdly high
    contrast. Default 50 kHz is well below any real qubit linewidth.

    Returns dict with: f0, fwhm, amp, bg0, bg1, r2, contrast, popt(5), success,
    edge_peak, peak_idx, fit_window_half_hz (the refit half-width — useful for
    plotting so the overlay only spans the region the fit was actually done on).
    """
    nan5 = np.full(5, np.nan)
    result = dict(
        f0=np.nan, fwhm=np.nan, amp=np.nan, bg0=np.nan, bg1=np.nan,
        r2=np.nan, contrast=np.nan, popt=nan5.copy(),
        success=False, edge_peak=False, peak_idx=-1,
        fit_window_half_hz=np.nan,
    )

    freqs = np.asarray(freqs, dtype=float)
    signal = np.asarray(signal, dtype=float)
    N = len(freqs)
    if N < 8:
        return result

    span_hz = freqs[-1] - freqs[0]
    edge_s = int(N * edge_fraction)
    step_hz = float(freqs[1] - freqs[0]) if N > 1 else 1e5
    fwhm_min = max(2.0 * step_hz, fwhm_floor_hz)

    # 1. Smooth
    win = min(smooth_window, N // 3 * 2 - 1)
    win = win if win % 2 == 1 else win - 1
    win = max(win, 5)
    try:
        smoothed = savgol_filter(signal, win, 3)
    except Exception:
        smoothed = signal.copy()
    residual_std = float(np.std(signal - smoothed))

    # 2. Find best inner peak above the noise floor
    peak_idx, is_edge = find_best_peak(
        smoothed, residual_std, edge_fraction=edge_fraction,
    )
    result["peak_idx"] = peak_idx
    result["edge_peak"] = is_edge
    if is_edge or peak_idx < edge_s or peak_idx > N - 1 - edge_s:
        return result

    f0_init = freqs[peak_idx]

    # 3. Smart FWHM init via local linear detrend + half-max crossings
    detr_half = detrend_window_mhz * 1e6 / 2.0
    detr_mask = (freqs >= f0_init - detr_half) & (freqs <= f0_init + detr_half)
    if detr_mask.sum() < 8:
        detr_mask = np.ones(N, dtype=bool)
    f_d = freqs[detr_mask]
    s_d = signal[detr_mask]
    b0d = (s_d[0] + s_d[-1]) / 2.0
    b1d = (s_d[-1] - s_d[0]) / (f_d[-1] - f_d[0]) if len(f_d) > 1 else 0.0
    s_detr = s_d - (b0d + b1d * (f_d - f_d.mean()))
    pi2 = int(np.argmax(s_detr))
    half_d = s_detr[pi2] / 2.0
    lc = np.where(s_detr[: pi2 + 1] <= half_d)[0]
    rc = np.where(s_detr[pi2:] <= half_d)[0]
    if len(lc) and len(rc):
        fwhm_init = f_d[pi2 + rc[0]] - f_d[lc[-1]]
    else:
        # Fallback: scan-width tenth
        fwhm_init = max(span_hz * 0.1, 1e5)
    fwhm_init = max(fwhm_init, 1e5)  # floor at 100 kHz

    # 4. First fit window (±N × fwhm_init, clamped)
    half_win = max(fwhm_init * window_fwhm_factor / 2.0, min_window_mhz * 1e6 / 2.0)
    fit_mask = (freqs >= f0_init - half_win) & (freqs <= f0_init + half_win)
    if fit_mask.sum() < 8:
        fit_mask = np.ones(N, dtype=bool)
    f_win = freqs[fit_mask]
    s_win = signal[fit_mask]

    bg0_init = float(np.median(s_win))
    bg1_init = 0.0
    amp_init = float(s_win.max() - bg0_init)
    if amp_init <= 0:
        amp_init = float(np.std(s_win))
    p0 = [f0_init, max(fwhm_init, fwhm_min), amp_init, bg0_init, bg1_init]
    bounds = (
        [f_win.min(), fwhm_min, 0.0, -np.inf, -np.inf],
        [f_win.max(), span_hz, max(amp_init * 5, 1e-6) * 10, np.inf, np.inf],
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt1, _ = curve_fit(
                lorentzian_peak_linbg, f_win, s_win,
                p0=p0, bounds=bounds, maxfev=10000,
            )
    except Exception:
        return result

    # 5. Wider refit: ±max(refit_window_fwhm_factor/2 × FWHM, min_refit_window/2).
    #    Wider than the first pass so the linear-bg slope is constrained on a
    #    representative slice of flat baseline and R² is a meaningful number.
    f0_a, fwhm_a, amp_a, bg0_a, bg1_a = popt1
    refit_half = max(
        fwhm_a * refit_window_fwhm_factor / 2.0,
        min_refit_window_mhz * 1e6 / 2.0,
    )
    refit_mask = (freqs >= f0_a - refit_half) & (freqs <= f0_a + refit_half)
    if refit_mask.sum() < 8:
        refit_mask = fit_mask
    f_win = freqs[refit_mask]
    s_win = signal[refit_mask]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                lorentzian_peak_linbg, f_win, s_win,
                p0=list(popt1),
                bounds=(
                    [f_win.min(), fwhm_min, 0.0, -np.inf, -np.inf],
                    [f_win.max(), span_hz, max(amp_a * 5, 1e-6) * 10, np.inf, np.inf],
                ),
                maxfev=10000,
            )
    except Exception:
        popt = popt1  # accept first pass if refit fails
    # Record the half-width actually used so the plot can clip its overlay
    # to the region the fit covered.
    result["fit_window_half_hz"] = float(refit_half)

    f0_fit, fwhm_fit, amp_fit, bg0_fit, bg1_fit = popt

    # 6. R² inside the refit window (matches what the plot shows)
    y_pred = lorentzian_peak_linbg(f_win, *popt)
    ss_res = float(np.sum((s_win - y_pred) ** 2))
    ss_tot = float(np.sum((s_win - s_win.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # 7. Contrast: fitted peak height / fitted baseline level at f0
    bg_at_f0 = bg0_fit + bg1_fit * (f0_fit - f_win.mean())
    contrast = amp_fit / abs(bg_at_f0) if abs(bg_at_f0) > 1e-12 else float("inf")

    # 8. Quality gates
    success = (
        r2 >= r2_threshold
        and 0 < fwhm_fit <= max_fwhm_mhz * 1e6
        and fwhm_fit / span_hz <= 0.5
        and contrast >= min_contrast
        and f_win.min() <= f0_fit <= f_win.max()
    )

    result.update(
        f0=float(f0_fit), fwhm=float(fwhm_fit), amp=float(amp_fit),
        bg0=float(bg0_fit), bg1=float(bg1_fit),
        r2=float(r2), contrast=float(contrast),
        popt=np.array(popt, dtype=float), success=bool(success),
    )
    return result


# ---------------------------------------------------------------------------
# FitParameters dataclass
# ---------------------------------------------------------------------------

@dataclass
class FitParameters:
    """Stores the fitted qubit-spectroscopy parameters for a single qubit."""

    frequency: float       # absolute RF frequency of the qubit
    relative_freq: float   # peak position in the detuning frame
    fwhm: float
    r2: float
    contrast: float
    iw_angle: float        # final integration weight angle = prev + delta (rad)
    saturation_amp: float
    x180_amp: float
    success: bool


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log the fitted results for all qubits."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        status = "SUCCESS!" if res["success"] else "FAIL!"
        log_callable(
            f"Results for qubit {q}: {status}\n"
            f"\tQubit frequency: {1e-9 * res['frequency']:.4f} GHz | "
            f"FWHM: {1e-3 * res['fwhm']:.1f} kHz | "
            f"R²: {res['r2']:.3f} | "
            f"contrast: {res['contrast']:.3f}\n"
            f"\tIntegration-weight angle: {res['iw_angle']:.3f} rad | "
            f"saturation amp: {1e3 * res['saturation_amp']:.2f} mV | "
            f"x180 amp: {1e3 * res['x180_amp']:.2f} mV"
        )


# ---------------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------------

def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Convert IQ to V, add amplitude/phase, attach full-frequency coordinate."""
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    full_freq = np.array(
        [ds.detuning.values + q.xy.RF_frequency for q in node.namespace["qubits"]]
    )
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


# ---------------------------------------------------------------------------
# fit_raw_data
# ---------------------------------------------------------------------------

def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit a Lorentzian peak per qubit on the rotated I-quadrature.

    Per qubit:
      1. arctan2-rotate IQ onto the new I axis (re-used from the legacy node).
      2. Smart-init + first `curve_fit` + narrow-window refit on `I_rot(detuning)`.
      3. Compute R² inside the refit window, plus contrast = amp / |baseline|.
      4. `success = R² > threshold && FWHM in (0, max] && contrast > min`.
      5. Derive the absolute integration-weight angle, saturation amplitude
         and x180 amplitude using the existing legacy formulas.
    """
    qubits = node.namespace["qubits"]
    limits = [instrument_limits(q.xy) for q in qubits]

    # --- Per-qubit IQ rotation (per legacy analysis.py:91-99) ---
    shifts = np.abs(
        (ds.IQ_abs - ds.IQ_abs.mean(dim="detuning"))
    ).idxmax(dim="detuning")
    rotation_angle = np.arctan2(
        ds.sel(detuning=shifts).Q - ds.Q.mean(dim="detuning"),
        ds.sel(detuning=shifts).I - ds.I.mean(dim="detuning"),
    )
    ds_fit = ds.assign({"iw_angle_delta": rotation_angle})
    ds_fit = ds_fit.assign(
        {"I_rot": ds_fit.I * np.cos(rotation_angle) + ds_fit.Q * np.sin(rotation_angle)}
    )

    # --- Per-qubit Lorentzian peak fit on I_rot vs detuning ---
    qubit_names = [q.name for q in qubits]
    f0_list, fwhm_list, amp_list, bg0_list, bg1_list = [], [], [], [], []
    r2_list, contrast_list, success_list, popt_list, fit_half_list = (
        [], [], [], [], [],
    )
    for q in qubits:
        det_q = ds_fit.detuning.values
        sig_q = ds_fit.sel(qubit=q.name).I_rot.values
        res = fit_qubit_peak(
            det_q, sig_q,
            max_fwhm_mhz=node.parameters.max_fwhm_mhz,
            r2_threshold=node.parameters.r2_threshold,
            min_contrast=node.parameters.min_contrast,
        )
        f0_list.append(res["f0"])
        fwhm_list.append(res["fwhm"])
        amp_list.append(res["amp"])
        bg0_list.append(res["bg0"])
        bg1_list.append(res["bg1"])
        r2_list.append(0.0 if np.isnan(res["r2"]) else res["r2"])
        contrast_list.append(0.0 if np.isnan(res["contrast"]) else res["contrast"])
        success_list.append(res["success"])
        popt_list.append(res["popt"])
        fit_half_list.append(res["fit_window_half_hz"])

    popt_arr = np.stack(popt_list, axis=0)  # (n_qubits, 5)
    f0_arr = np.array(f0_list)
    fwhm_arr = np.array(fwhm_list)

    # --- Derive QUAM-state values (mirrors legacy _extract_relevant_fit_parameters) ---
    full_freq = np.array([q.xy.RF_frequency for q in qubits])
    res_freq = f0_arr + full_freq

    prev_angles = np.array(
        [q.resonator.operations["readout"].integration_weights_angle for q in qubits]
    )
    iw_angle_final = (prev_angles + rotation_angle.values) % (2 * np.pi)

    used_amp = np.array(
        [
            q.xy.operations["saturation"].amplitude * node.parameters.operation_amplitude_factor
            for q in qubits
        ]
    )
    # Treat target_peak_width as the desired FWHM (matches the new fwhm semantics).
    with np.errstate(divide="ignore", invalid="ignore"):
        factor_cw = node.parameters.target_peak_width / fwhm_arr
        sat_amp = factor_cw * used_amp / node.parameters.operation_amplitude_factor
        x180_length = np.array(
            [q.xy.operations["x180"].length * 1e-9 for q in qubits]
        )
        factor_x180 = np.pi / (fwhm_arr * x180_length)
        x180_amp = factor_x180 * used_amp

    # Tighten success further by the bound checks the legacy node used.
    sat_amp_in_bounds = np.array(
        [
            (not np.isnan(sat_amp[i])) and abs(sat_amp[i]) < limits[i].max_wf_amplitude
            for i in range(len(qubits))
        ]
    )
    success_arr = np.array(success_list) & sat_amp_in_bounds

    # --- Pack into ds_fit and FitParameters dict ---
    ds_fit = ds_fit.assign(
        {
            "f0": xr.DataArray(
                f0_arr, coords={"qubit": qubit_names}, dims="qubit",
                attrs={"long_name": "qubit detuning at peak", "units": "Hz"},
            ),
            "res_freq": xr.DataArray(
                res_freq, coords={"qubit": qubit_names}, dims="qubit",
                attrs={"long_name": "absolute qubit RF frequency", "units": "Hz"},
            ),
            "fwhm": xr.DataArray(
                fwhm_arr, coords={"qubit": qubit_names}, dims="qubit",
                attrs={"long_name": "peak FWHM", "units": "Hz"},
            ),
            "amplitude": xr.DataArray(
                np.array(amp_list), coords={"qubit": qubit_names}, dims="qubit",
            ),
            "bg0": xr.DataArray(
                np.array(bg0_list), coords={"qubit": qubit_names}, dims="qubit",
            ),
            "bg1": xr.DataArray(
                np.array(bg1_list), coords={"qubit": qubit_names}, dims="qubit",
            ),
            "r2": xr.DataArray(
                np.array(r2_list), coords={"qubit": qubit_names}, dims="qubit",
                attrs={"long_name": "R²"},
            ),
            "contrast": xr.DataArray(
                np.array(contrast_list), coords={"qubit": qubit_names}, dims="qubit",
                attrs={"long_name": "contrast = amp / |baseline|"},
            ),
            "popt": xr.DataArray(
                popt_arr,
                coords={"qubit": qubit_names, "param": np.arange(5)},
                dims=["qubit", "param"],
                attrs={"long_name": "fit parameters [f0, fwhm, amp, bg0, bg1]"},
            ),
            "fit_window_half_hz": xr.DataArray(
                np.array(fit_half_list), coords={"qubit": qubit_names}, dims="qubit",
                attrs={"long_name": "half-width of the refit window", "units": "Hz"},
            ),
            "iw_angle": xr.DataArray(
                iw_angle_final, coords={"qubit": qubit_names}, dims="qubit",
                attrs={"long_name": "integration-weight angle", "units": "rad"},
            ),
            "saturation_amplitude": xr.DataArray(
                sat_amp, coords={"qubit": qubit_names}, dims="qubit",
                attrs={"units": "V"},
            ),
            "x180_amplitude": xr.DataArray(
                x180_amp, coords={"qubit": qubit_names}, dims="qubit",
                attrs={"units": "V"},
            ),
            "success": xr.DataArray(
                success_arr, coords={"qubit": qubit_names}, dims="qubit",
            ),
        }
    )

    fit_results = {
        q: FitParameters(
            frequency=float(ds_fit.sel(qubit=q).res_freq.values),
            relative_freq=float(ds_fit.sel(qubit=q).f0.values),
            fwhm=float(ds_fit.sel(qubit=q).fwhm.values),
            r2=float(ds_fit.sel(qubit=q).r2.values),
            contrast=float(ds_fit.sel(qubit=q).contrast.values),
            iw_angle=float(ds_fit.sel(qubit=q).iw_angle.values),
            saturation_amp=float(ds_fit.sel(qubit=q).saturation_amplitude.values),
            x180_amp=float(ds_fit.sel(qubit=q).x180_amplitude.values),
            success=bool(ds_fit.sel(qubit=q).success.values),
        )
        for q in qubit_names
    }
    return ds_fit, fit_results
