import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths, savgol_filter

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V


def lorentzian_dip_linbg(
    f: NDArray[np.float64], f0: float, fwhm: float, amp: float, bg0: float, bg1: float
) -> NDArray[np.float64]:
    """Inverted Lorentzian with linear background.

    R(f) = [bg0 + bg1*(f - fc)] - amp / [1 + ((f - f0)/(fwhm/2))^2]

    fc is the centre of the frequency array so that bg1 has units of V/Hz
    without absorbing a large offset into bg0.
    """
    fc = f.mean()
    return (bg0 + bg1 * (f - fc)) - amp / (1 + ((f - f0) / (fwhm / 2)) ** 2)


def lorentzian_dip_quadbg(
    f: NDArray[np.float64], f0: float, fwhm: float, amp: float, bg0: float, bg1: float, bg2: float
) -> NDArray[np.float64]:
    """Inverted Lorentzian with quadratic background (curved / bowl baselines)."""
    fc = f.mean()
    x = f - fc
    return (bg0 + bg1 * x + bg2 * x * x) - amp / (1 + ((f - f0) / (fwhm / 2)) ** 2)


def find_dip_candidates(
    freqs: NDArray[np.float64],
    smoothed: NDArray[np.float64],
    noise_sigma: float,
    *,
    min_dip_snr: float = 6.0,
    max_dip_width_hz: float = 22.5e6,
    edge_fraction: float = 0.02,
) -> list[dict[str, int | float]]:
    """Noise-relative, background-free dip candidates on an arbitrary-width scan.

    1. Remove broad structure (cable ripple, bowl) with a rolling-median
       baseline (~8 MHz window): resonator dips (<= a few MHz) pass through,
       ripple minima (>= tens of MHz) are levelled — so their fake prominence
       disappears BEFORE significance is measured.
    2. ``find_peaks`` on the inverted baseline-subtracted trace with prominence
       >= ``min_dip_snr`` x per-point noise sigma (statistical existence, not a
       range fraction).
    3. Keep only candidates whose half-prominence width looks like a resonator
       (<= ``max_dip_width_hz``).

    Returns a list of dicts ``{idx, f, prom_snr, width_hz}`` sorted by
    descending prominence (may be empty).
    """
    freqs = np.asarray(freqs, dtype=float)
    smoothed = np.asarray(smoothed, dtype=float)
    N = len(smoothed)
    step = float(np.median(np.abs(np.diff(freqs)))) if N > 1 else 1e5
    edge = max(1, int(N * edge_fraction))
    sigma = max(float(noise_sigma), 1e-15)

    bl_win = max(int(round(8e6 / step)) | 1, 9)
    bl_win = min(bl_win, max(N // 3 * 2 - 1, 9))
    baseline = median_filter(smoothed, size=bl_win, mode="nearest")
    detr = smoothed - baseline

    pk, props = find_peaks(-detr, prominence=min_dip_snr * sigma)
    if len(pk) == 0:
        return []
    widths = peak_widths(-detr, pk, rel_height=0.5)[0] * step
    out = []
    for p, prom, w in zip(pk, props["prominences"], widths):
        if p < edge or p > N - 1 - edge:
            continue
        if w > max_dip_width_hz:
            continue
        out.append(dict(idx=int(p), f=float(freqs[p]), prom_snr=float(prom / sigma), width_hz=float(w)))
    out.sort(key=lambda c: -c["prom_snr"])
    return out


def _smooth_and_estimate_noise(amplitude: NDArray[np.float64], smooth_window: int) -> tuple[NDArray[np.float64], float]:
    """Savgol-smooth the trace and estimate the per-point noise sigma from the residual (robust MAD)."""
    N = len(amplitude)
    win = min(smooth_window, N // 3 * 2 - 1)
    win = win if win % 2 == 1 else win - 1
    win = max(win, 5)
    try:
        smoothed = savgol_filter(amplitude, win, 3)
    except Exception:
        smoothed = amplitude.copy()
    resid = amplitude - smoothed
    noise_sigma = 1.4826 * float(np.median(np.abs(resid - np.median(resid)))) + 1e-15
    return smoothed, noise_sigma


def _estimate_initial_fwhm(
    freqs: NDArray[np.float64],
    smoothed: NDArray[np.float64],
    f0_init: float,
    fallback_width_hz: float,
    detrend_window_mhz: float,
    step: float,
) -> tuple[float, float]:
    """Linearly detrend a fixed window around f0_init and measure the half-max width there.

    A fixed-size window (rather than one scaled to a not-yet-known FWHM) keeps
    this estimate robust on curved/bowl baselines. Falls back to the
    candidate's width_hz if the half-max crossings run off either edge of the
    window.

    Returns (fwhm_init, depth_d) where depth_d is the detrended dip depth,
    kept for the contrast fallback in case the full fit later turns out
    unusable.
    """
    N = len(freqs)
    detr_half = detrend_window_mhz * 1e6 / 2.0
    detr_mask = (freqs >= f0_init - detr_half) & (freqs <= f0_init + detr_half)
    if detr_mask.sum() < 8:
        detr_mask = np.ones(N, dtype=bool)
    f_d = freqs[detr_mask]
    a_d = smoothed[detr_mask]
    b0d = (a_d[0] + a_d[-1]) / 2.0
    b1d = (a_d[-1] - a_d[0]) / (f_d[-1] - f_d[0]) if len(f_d) > 1 else 0.0
    a_detr = a_d - (b0d + b1d * (f_d - f_d.mean()))
    di2 = int(np.argmin(a_detr))
    depth_d = -float(a_detr[di2])
    hd2 = a_detr[di2] + depth_d / 2.0
    lc2 = np.where(a_detr[: di2 + 1] >= hd2)[0]
    rc2 = np.where(a_detr[di2:] >= hd2)[0]
    if len(lc2) and len(rc2):
        fwhm_init = f_d[di2 + rc2[0]] - f_d[lc2[-1]]
    else:
        fwhm_init = max(fallback_width_hz, 4 * step)
    return max(float(fwhm_init), 2 * step), depth_d


def _fit_lorentzian_dip_ladder(
    freqs: NDArray[np.float64],
    amplitude: NDArray[np.float64],
    f0_init: float,
    fwhm_init: float,
    A_raw: float,
    step: float,
    window_fwhm_factor: float,
    min_window_mhz: float,
) -> dict[str, Any] | None:
    """Try a small ladder of fit windows/models, keep whichever gives the best R².

    Tries, in order: a linear background at the standard window, the same
    model at a NARROWER window (kills the leverage of curved baselines), and
    a quadratic background at the standard window (fits the curvature
    instead). Returns None if every attempt lacked enough points or failed to
    converge.
    """
    tries = (
        (max(fwhm_init * window_fwhm_factor / 2.0, min_window_mhz * 1e6 / 2.0), lorentzian_dip_linbg, 5),
        (max(fwhm_init * window_fwhm_factor / 4.0, min_window_mhz * 1e6 / 4.0), lorentzian_dip_linbg, 5),
        (max(fwhm_init * window_fwhm_factor / 2.0, min_window_mhz * 1e6 / 2.0), lorentzian_dip_quadbg, 6),
    )
    best = None
    for half_win, model, npar in tries:
        fit_mask = (freqs >= f0_init - half_win) & (freqs <= f0_init + half_win)
        if fit_mask.sum() < npar + 3:
            continue
        f_win = freqs[fit_mask]
        a_win = amplitude[fit_mask]
        b0 = (a_win[0] + a_win[-1]) / 2.0
        b1 = (a_win[-1] - a_win[0]) / (f_win[-1] - f_win[0]) if len(f_win) > 1 else 0.0
        Ar_win = a_win.max() - a_win.min()
        p0 = [f0_init, fwhm_init, Ar_win, b0, b1] + ([0.0] if npar == 6 else [])
        lb = [f_win.min(), 2 * step, 0, -np.inf, -np.inf] + ([-np.inf] if npar == 6 else [])
        ub = [f_win.max(), f_win.max() - f_win.min(), max(A_raw * 5, Ar_win * 5), np.inf, np.inf] + (
            [np.inf] if npar == 6 else []
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt_t, _ = curve_fit(model, f_win, a_win, p0=p0, bounds=(lb, ub), maxfev=10000)
        except Exception:
            continue
        y_pred = model(f_win, *popt_t)
        ss_res = float(np.sum((a_win - y_pred) ** 2))
        ss_tot = float(np.sum((a_win - a_win.mean()) ** 2)) + 1e-30
        r2_t = 1.0 - ss_res / ss_tot
        if best is None or r2_t > best["r2"]:
            bg_at_f0 = popt_t[3] + popt_t[4] * (popt_t[0] - f_win.mean())
            best = dict(
                popt=np.array(popt_t[:5], dtype=float),
                r2=float(r2_t),
                f0=float(popt_t[0]),
                fwhm=float(popt_t[1]),
                amp=float(popt_t[2]),
                contrast=float(popt_t[2] / bg_at_f0) if bg_at_f0 > 0 else 0.0,
                f_win=(float(f_win.min()), float(f_win.max())),
            )
    return best


def _select_fit_outputs(
    best: dict[str, Any] | None,
    f0_init: float,
    fwhm_init: float,
    depth_d: float,
    smoothed: NDArray[np.float64],
    span_hz: float,
    r2_threshold: float,
    max_fwhm_mhz: float,
    min_contrast: float,
) -> dict[str, Any]:
    """Pick between the fit-ladder result and the raw dip estimate, then apply the strict shape gates.

    A fit is only trusted if its center landed close to the detected dip
    (``best["f0"]`` within ``2*fwhm_init`` or 1 MHz of ``f0_init``) — otherwise
    it ran away and the frequency/FWHM fall back to the pre-fit estimates.
    ``success_shape`` gates on R², FWHM range, and contrast; frequency success
    never depends on it, since the dip's existence and position were already
    established before any fit was attempted.
    """

    nan5: NDArray[np.float64] = np.full(5, np.nan)
    max_allowed_drift_hz: float = max(2 * fwhm_init, 1e6)

    fit_trusted: bool = False
    if best is not None:
        fit_centre_drift_hz: float = abs(best["f0"] - f0_init)

        if fit_centre_drift_hz <= max_allowed_drift_hz:
            fit_trusted = True

    f0_out: float
    fwhm_out: float
    r2: float
    fitted_contrast: float
    popt: NDArray[np.float64]
    fit_win_lo: float
    fit_win_hi: float

    if fit_trusted:
        f0_out, fwhm_out = best["f0"], best["fwhm"]
        r2 = best["r2"]
        fitted_contrast = best["contrast"]
        popt = best["popt"]
        fit_win_lo, fit_win_hi = best["f_win"]
    else:
        # Fit unusable/ran away: the dip existence and position stand on their own.
        f0_out, fwhm_out = f0_init, fwhm_init
        r2 = best["r2"] if best is not None else 0.0
        popt = nan5.copy()
        fit_win_lo, fit_win_hi = np.nan, np.nan

        fitted_contrast = np.nan
        if np.median(smoothed) > 0:
            fitted_contrast = depth_d / np.median(smoothed)

    # success_shape is gated on fit_trusted (not just "best is not None") so a
    # discarded/ran-away fit's R² and window can never grade a report that
    # carries an all-NaN popt.
    good_r2: bool = r2 >= r2_threshold
    fwhm_in_range: bool = 0 < fwhm_out <= max_fwhm_mhz * 1e6
    fwhm_not_too_broad: bool = fwhm_out / span_hz <= 0.50
    contrast_ok: bool = fitted_contrast >= min_contrast
    centre_inside_fit_window: bool = fit_win_lo <= f0_out <= fit_win_hi

    success_shape: bool = bool(
        fit_trusted
        and good_r2
        and fwhm_in_range
        and fwhm_not_too_broad
        and contrast_ok
        and centre_inside_fit_window
    )

    return dict(
        f0=f0_out,
        fwhm=fwhm_out,
        r2=r2,
        contrast=fitted_contrast,
        popt=popt,
        fit_win_lo=fit_win_lo,
        fit_win_hi=fit_win_hi,
        success_shape=success_shape,
    )


def fit_resonator(
    freqs: NDArray[np.float64],
    amplitude: NDArray[np.float64],
    *,
    override_center_hz: float | None = None,
    override_span_hz: float | None = None,
    window_fwhm_factor: float = 4.0,
    min_window_mhz: float = 5.0,
    detrend_window_mhz: float = 10.0,
    max_fwhm_mhz: float = 15.0,
    r2_threshold: float = 0.85,
    min_contrast: float = 0.05,
    edge_fraction: float = 0.02,
    smooth_window: int = 11,
    min_dip_snr: float = 6.0,
    dominance: float = 2.0,
) -> dict[str, Any]:
    """Bring-up-grade resonator-dip fitter.

    When override_center_hz and override_span_hz are provided the data is
    pre-sliced to [center - span/2, center + span/2] before fitting, allowing
    the user to manually constrain the fit window for problematic qubits.

    Two-tier success:

    * ``success`` — FREQUENCY: a statistically significant, resonator-shaped
      dip exists (``min_dip_snr`` x noise sigma after broad-background removal)
      and its position is delivered. Does NOT depend on R².
    * ``success_shape`` — the fitted Lorentzian lineshape is trustworthy
      (legacy strict gates, after a fit ladder that also tries a narrower
      window and a quadratic background for curved baselines).

    Multi-dip transparency: ``candidates`` lists every significant dip
    (descending prominence); ``ambiguous`` is set when the second candidate is
    within ``dominance``x of the top one — downstream must disambiguate
    (expected-frequency prior / punch-out check), the fitter never silently
    guesses between comparable dips.

    Returns
    -------
    dict with keys:
        f0, fwhm, r2, success, success_shape, ambiguous, dip_snr,
        candidates (list of {idx, f, prom_snr, width_hz}),
        popt (array[5] or all-NaN), edge_dip, contrast, dip_idx, reason
    """
    _nan5 = np.full(5, np.nan)
    result = dict(
        f0=np.nan,
        fwhm=np.nan,
        r2=np.nan,
        success=False,
        success_shape=False,
        ambiguous=False,
        dip_snr=0.0,
        candidates=[],
        popt=_nan5.copy(),
        edge_dip=False,
        contrast=np.nan,
        dip_idx=-1,
        reason="",
        fit_win_lo=np.nan,
        fit_win_hi=np.nan,
    )

    freqs = np.asarray(freqs, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)

    # --- Apply manual window override if given ---
    if override_center_hz is not None and override_span_hz is not None:
        half = override_span_hz / 2.0
        mask_ov = (freqs >= override_center_hz - half) & (freqs <= override_center_hz + half)
        if mask_ov.sum() >= 8:
            freqs = freqs[mask_ov]
            amplitude = amplitude[mask_ov]

    span_hz = freqs[-1] - freqs[0]
    N = len(freqs)
    if N < 16:
        result["reason"] = "trace too short"
        return result
    step = float(np.median(np.abs(np.diff(freqs)))) if N > 1 else 1e5

    smoothed, noise_sigma = _smooth_and_estimate_noise(amplitude, smooth_window)

    # Significant dip candidates (noise-relative, background-free, width-capped)
    candidates = find_dip_candidates(
        freqs,
        smoothed,
        noise_sigma,
        min_dip_snr=min_dip_snr,
        max_dip_width_hz=1.5 * max_fwhm_mhz * 1e6,
        edge_fraction=edge_fraction,
    )
    result["candidates"] = candidates
    if not candidates:
        result["reason"] = "no significant dip"
        return result
    top = candidates[0]
    dip_idx = top["idx"]
    result["dip_idx"] = dip_idx
    result["dip_snr"] = top["prom_snr"]
    result["ambiguous"] = len(candidates) > 1 and candidates[1]["prom_snr"] >= top["prom_snr"] / dominance
    f0_init = float(freqs[dip_idx])
    A_raw = smoothed.max() - smoothed.min()

    fwhm_init, depth_d = _estimate_initial_fwhm(freqs, smoothed, f0_init, top["width_hz"], detrend_window_mhz, step)

    best = _fit_lorentzian_dip_ladder(
        freqs, amplitude, f0_init, fwhm_init, A_raw, step, window_fwhm_factor, min_window_mhz
    )

    outputs = _select_fit_outputs(
        best, f0_init, fwhm_init, depth_d, smoothed, span_hz, r2_threshold, max_fwhm_mhz, min_contrast
    )

    # Refine the reported centre to the true dip minimum (asymmetric-tail
    # bias correction; see _refine_dip_minimum).
    f0_out = _refine_dip_minimum(freqs, smoothed, outputs["f0"], max(outputs["fwhm"], 2 * step))
    popt = outputs["popt"]
    if not np.any(np.isnan(popt)):
        popt = np.array(popt, dtype=float)
        popt[0] = f0_out

    result.update(
        f0=float(f0_out),
        fwhm=float(outputs["fwhm"]),
        r2=float(outputs["r2"]),
        success=True,  # a significant, resonator-shaped dip exists (candidates non-empty)
        success_shape=outputs["success_shape"],
        popt=np.array(popt),
        contrast=float(outputs["contrast"]),
        fit_win_lo=float(outputs["fit_win_lo"]),
        fit_win_hi=float(outputs["fit_win_hi"]),
    )
    return result


def _refine_dip_minimum(
    freqs: NDArray[np.float64], smoothed: NDArray[np.float64], f0_guess: float, fwhm_guess: float
) -> float:
    """Locate the true dip minimum near *f0_guess* (the visible bottom).

    1. Find the smoothed-amplitude minimum within ±1.5·FWHM of f0_guess (a local
       window so far-off edge artefacts on poor traces are ignored).
    2. Refine to sub-step resolution with a 3-point parabolic interpolation
       around that minimum (uses only the immediate neighbours, so it captures
       the local bottom curvature and is NOT biased by the asymmetric tails the
       symmetric Lorentzian was pulled toward).

    Falls back to the local argmin, then to f0_guess, if the data are too sparse.
    """
    freqs = np.asarray(freqs, dtype=float)
    smoothed = np.asarray(smoothed, dtype=float)
    n = freqs.size
    if n < 3:
        return float(f0_guess)
    step = abs(freqs[1] - freqs[0])

    # 1. Local smoothed argmin within ±1.5*FWHM of the Lorentzian centre
    search_half = max(1.5 * fwhm_guess, 5 * step)
    win = (freqs >= f0_guess - search_half) & (freqs <= f0_guess + search_half)
    if win.sum() < 3:
        return float(f0_guess)
    masked = np.where(win, smoothed, np.inf)
    j = int(np.argmin(masked))

    # 2. 3-point parabolic interpolation around the local minimum
    if 0 < j < n - 1:
        y0, y1, y2 = smoothed[j - 1], smoothed[j], smoothed[j + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom > 0:  # concave up
            delta = 0.5 * (y0 - y2) / denom  # offset in units of the freq step
            if -1.0 <= delta <= 1.0:
                return float(freqs[j] + delta * (freqs[j + 1] - freqs[j]))
    return float(freqs[j])


@dataclass
class FitParameters:
    """Stores the fitted resonator parameters for a single qubit."""

    frequency: float
    fwhm: float
    r2: float
    success: bool  # FREQUENCY success: a significant dip exists, f0 delivered
    success_shape: bool = False  # Lorentzian lineshape trustworthy (strict gates)
    ambiguous: bool = False  # >1 comparable dip in the window — needs disambiguation
    dip_snr: float = 0.0  # top dip prominence / noise sigma
    candidates: list[dict[str, int | float]] = field(default_factory=list)  # all significant dips (desc prominence)
    fit_win_lo: float = float("nan")  # lower edge (Hz) of the window popt was actually fit on
    fit_win_hi: float = float("nan")  # upper edge (Hz) of the window popt was actually fit on


def log_fitted_results(
    fit_results: dict[str, dict[str, Any]], log_callable: Callable[[str], None] | None = None
) -> None:
    """Log the fitted results for all qubits (three-state + ambiguity in v2)."""

    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for q, res in fit_results.items():
        if res["success"] and res.get("success_shape", True):
            status = "SUCCESS!"
        elif res["success"]:
            status = "FREQUENCY OK (lineshape poor)"
        else:
            status = "FAIL!"

        if res.get("ambiguous"):
            n = len(res.get("candidates") or [])
            status += f"  [AMBIGUOUS: {n} comparable dips — verify vs expected freq / punch-out]"
            
        log_callable(
            f"Results for qubit {q}:  {status}\n"
            f"\tResonator frequency: {1e-9 * res['frequency']:.4f} GHz | "
            f"FWHM: {1e-3 * res['fwhm']:.1f} kHz | "
            f"R²: {res['r2']:.3f} | dip SNR: {res.get('dip_snr', 0.0):.1f}"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert IQ to voltage, add amplitude/phase, and attach full-frequency coordinates."""
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    full_freq = np.array([ds.detuning.values + q.resonator.RF_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> tuple[xr.Dataset, dict[str, FitParameters]]:
    """Fit a Lorentzian dip to each qubit's amplitude trace.

    Reads re_fit_resonators / re_fit_centers_ghz / re_fit_span_mhz from
    node.parameters to apply manual fit-window overrides for selected qubits.

    Returns
    -------
    ds_fit : xr.Dataset
        Per-qubit fit results with variables: f0, fwhm, r2, success, popt.
    fit_results : dict[str, FitParameters]
    """
    qubits = node.namespace["qubits"]
    params = node.parameters

    # Build per-qubit override lookup from the three parallel lists
    overrides: dict[str, dict[str, float]] = {}
    re_fit_names = params.re_fit_resonators or []
    re_fit_centers = params.re_fit_centers_ghz or []
    re_fit_spans = params.re_fit_span_mhz or []
    for name, center_ghz, span_mhz in zip(re_fit_names, re_fit_centers, re_fit_spans):
        overrides[name] = {
            "center_hz": center_ghz * 1e9,
            "span_hz": span_mhz * 1e6,
        }

    qubit_names = [q.name for q in qubits]
    f0_vals, fwhm_vals, r2_vals, success_vals = [], [], [], []
    shape_vals, amb_vals, snr_vals, cand_vals = [], [], [], []
    popt_vals = []
    fit_win_lo_vals, fit_win_hi_vals = [], []

    for i, q in enumerate(qubits):
        full_freq_q = ds.sel(qubit=q.name).full_freq.values
        amplitude_q = ds.sel(qubit=q.name).IQ_abs.values

        ov = overrides.get(q.name, {})
        res = fit_resonator(
            full_freq_q,
            amplitude_q,
            override_center_hz=ov.get("center_hz"),
            override_span_hz=ov.get("span_hz"),
            min_dip_snr=getattr(params, "min_dip_snr", 6.0),
            dominance=getattr(params, "dip_dominance", 2.0),
        )

        f0_vals.append(res["f0"])
        fwhm_vals.append(res["fwhm"])
        r2_vals.append(res["r2"] if not np.isnan(res["r2"]) else 0.0)
        success_vals.append(res["success"])
        shape_vals.append(res["success_shape"])
        amb_vals.append(res["ambiguous"])
        snr_vals.append(res["dip_snr"])
        cand_vals.append(res["candidates"])
        popt_vals.append(res["popt"])  # shape (5,) or all-NaN
        fit_win_lo_vals.append(res["fit_win_lo"])
        fit_win_hi_vals.append(res["fit_win_hi"])

    popt_array = np.stack(popt_vals, axis=0)  # (n_qubits, 5)

    ds_fit = xr.Dataset(
        {
            "f0": xr.DataArray(
                f0_vals,
                coords={"qubit": qubit_names},
                dims="qubit",
                attrs={"long_name": "resonator frequency", "units": "Hz"},
            ),
            "fwhm": xr.DataArray(
                fwhm_vals, coords={"qubit": qubit_names}, dims="qubit", attrs={"long_name": "FWHM", "units": "Hz"}
            ),
            "r2": xr.DataArray(r2_vals, coords={"qubit": qubit_names}, dims="qubit", attrs={"long_name": "R²"}),
            "success": xr.DataArray(
                success_vals,
                coords={"qubit": qubit_names},
                dims="qubit",
                attrs={"long_name": "frequency success (significant dip found)"},
            ),
            "success_shape": xr.DataArray(
                shape_vals,
                coords={"qubit": qubit_names},
                dims="qubit",
                attrs={"long_name": "lineshape trustworthy (strict gates)"},
            ),
            "ambiguous": xr.DataArray(
                amb_vals,
                coords={"qubit": qubit_names},
                dims="qubit",
                attrs={"long_name": "multiple comparable dips in window"},
            ),
            "dip_snr": xr.DataArray(
                snr_vals,
                coords={"qubit": qubit_names},
                dims="qubit",
                attrs={"long_name": "top dip prominence / noise sigma"},
            ),
            "popt": xr.DataArray(
                popt_array,
                coords={"qubit": qubit_names, "param": np.arange(5)},
                dims=["qubit", "param"],
                attrs={"long_name": "fit parameters [f0, fwhm, amp, bg0, bg1]"},
            ),
            "fit_win_lo": xr.DataArray(
                fit_win_lo_vals,
                coords={"qubit": qubit_names},
                dims="qubit",
                attrs={"long_name": "lower edge of the window popt was fit on", "units": "Hz"},
            ),
            "fit_win_hi": xr.DataArray(
                fit_win_hi_vals,
                coords={"qubit": qubit_names},
                dims="qubit",
                attrs={"long_name": "upper edge of the window popt was fit on", "units": "Hz"},
            ),
        }
    )

    fit_results = {
        q: FitParameters(
            frequency=float(ds_fit.sel(qubit=q).f0.values),
            fwhm=float(ds_fit.sel(qubit=q).fwhm.values),
            r2=float(ds_fit.sel(qubit=q).r2.values),
            success=bool(ds_fit.sel(qubit=q).success.values),
            success_shape=bool(ds_fit.sel(qubit=q).success_shape.values),
            ambiguous=bool(ds_fit.sel(qubit=q).ambiguous.values),
            dip_snr=float(ds_fit.sel(qubit=q).dip_snr.values),
            candidates=cand_vals[qubit_names.index(q)],
            fit_win_lo=float(ds_fit.sel(qubit=q).fit_win_lo.values),
            fit_win_hi=float(ds_fit.sel(qubit=q).fit_win_hi.values),
        )
        for q in qubit_names
    }
    return ds_fit, fit_results
