"""CROT spectroscopy parity-diff analysis.

The CROT spectroscopy measures the target qubit's resonance frequency as a
function of the exchange voltage, with and without an x180 pulse on the
control qubit.  When the control is in |↓⟩ the target resonates at f_↓;
when flipped to |↑⟩ it resonates at f_↑.  The exchange coupling is
J = |f_↑ − f_↓|.

**Fitting strategy**

Up to four Lorentzian peaks can appear in a single frequency slice:
  - f_target_↓ and f_target_↑ due to exchange splitting of the target qubit.
  - f_control_↓ and f_control_↑ if both qubit resonances fall inside the
    frequency sweep window.
  - Mixed-state initialisation can cause both target frequencies to appear
    simultaneously in the same subplot.

For each exchange-voltage slice, ``_fit_multi_lorentzian`` detects peaks with
``scipy.signal.find_peaks``, then fits an N-Lorentzian model (N ≤ max_peaks)
with explicit bounds on every parameter.

``_build_tracks`` stitches per-slice peak positions into continuous tracks
across exchange voltage using greedy nearest-neighbour assignment.  This
handles peaks appearing/disappearing without misidentifying a jump as the
same resonance.

The primary track for each control state is selected by median amplitude:
the target qubit resonance should dominate because the XY drive is tuned to
the target qubit.  MAD-based outlier rejection cleans residual bad fits before
the optimal exchange point is chosen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

import xarray as xr

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class LorentzianFitResult:
    """Fitted parameters for a single Lorentzian peak."""

    f0: float = 0.0
    amplitude: float = 0.0
    gamma: float = 0.0
    offset: float = 0.0
    success: bool = False
    n_peaks: int = 0
    """Number of Lorentzian components detected in this slice."""


@dataclass
class ExchangeModelFit:
    """Fitted parameters for J(V) = J_0 * exp((V - V_ref) / lever_arm)."""

    J_0: float = 0.0
    V_ref: float = 0.0
    lever_arm: float = 0.0
    success: bool = False


@dataclass
class CROTFitResult:
    """Aggregated fit result for one qubit pair."""

    exchange_coupling_J: float = 0.0
    """Exchange coupling |f_↑ − f_↓| in Hz at the optimal exchange voltage."""
    crot_frequency_down: float = 0.0
    """Target qubit resonance when control is |↓⟩ (Hz)."""
    crot_frequency_up: float = 0.0
    """Target qubit resonance when control is |↑⟩ (Hz)."""
    optimal_exchange_idx: int = 0
    """Index into the exchange array where J is best resolved."""
    exchange_model: ExchangeModelFit | None = None
    """Fitted exponential exchange model J(V)."""
    success: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _scan_width(
    freqs: np.ndarray,
    smoothed: np.ndarray,
    peak_idx: int,
    half_level: float,
    is_peak: bool,
) -> float:
    """Estimate FWHM by scanning outward from peak_idx.

    Unlike a global threshold mask this cannot accidentally bridge across a
    second peak, which would inflate gamma and break curve_fit convergence.
    """
    above = smoothed > half_level if is_peak else smoothed < half_level
    left = peak_idx
    while left > 0 and above[left - 1]:
        left -= 1
    right = peak_idx
    while right < len(smoothed) - 1 and above[right + 1]:
        right += 1
    min_step = float(np.min(np.abs(np.diff(freqs))))
    return max(abs(float(freqs[right] - freqs[left])), min_step)


def _mad_outlier_mask(arr: np.ndarray, threshold: float = 4.0) -> np.ndarray:
    """Boolean mask of inlier elements using MAD-based outlier detection."""
    finite = np.isfinite(arr)
    if np.sum(finite) < 4:
        return finite
    median = np.nanmedian(arr[finite])
    mad = np.nanmedian(np.abs(arr[finite] - median))
    if mad < 1e-30:
        return finite
    return finite & (np.abs(arr - median) / (1.4826 * mad) < threshold)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-peak fitter
# ──────────────────────────────────────────────────────────────────────────────


def _fit_multi_lorentzian(
    freqs: np.ndarray,
    signal: np.ndarray,
    max_peaks: int = 4,
) -> list[tuple[float, float, float]]:
    """Detect and fit up to *max_peaks* Lorentzians in a 1-D frequency trace.

    Returns a list of ``(f0, amplitude, gamma)`` tuples sorted by frequency.
    Returns an empty list if the fit fails or no valid peaks are found.

    Strategy
    --------
    1.  Lightly smooth the trace for robust peak detection and initial-
        parameter estimation (smoothed signal is NOT passed to curve_fit).
    2.  Use ``find_peaks`` to count peaks; keep up to *max_peaks* most
        prominent.
    3.  Fit a shared N-Lorentzian + offset model with bounds:
        ``freq_min ≤ f0 ≤ freq_max``, ``gamma ∈ [freq_step, freq_range]``,
        amplitude constrained to the correct sign (peak vs dip).
    4.  Reject components whose fitted f0 falls outside the sweep range or
        whose gamma is unphysical.
    """
    if len(freqs) < 5 or not np.any(np.isfinite(signal)):
        return []

    freq_min = float(freqs.min())
    freq_max = float(freqs.max())
    freq_range = freq_max - freq_min
    freq_step = float(np.min(np.abs(np.diff(freqs))))

    smooth_len = max(3, len(signal) // 10)
    smoothed = uniform_filter1d(signal.astype(float), size=smooth_len)

    sig_mean = float(np.nanmean(smoothed))
    is_dip = (sig_mean - float(np.nanmin(smoothed))) > (float(np.nanmax(smoothed)) - sig_mean)
    detection = -smoothed if is_dip else smoothed
    sig_range = float(np.ptp(detection))

    min_dist = max(1, len(freqs) // (2 * max_peaks + 1))
    peaks_idx, _ = find_peaks(
        detection,
        distance=min_dist,
        prominence=max(0.08 * sig_range, 1e-9),
    )

    if len(peaks_idx) == 0:
        peaks_idx = np.array([int(np.nanargmax(detection))])

    # Keep the most prominent peaks up to max_peaks
    if len(peaks_idx) > max_peaks:
        order = np.argsort(detection[peaks_idx])[::-1]
        peaks_idx = np.sort(peaks_idx[order[:max_peaks]])

    n = len(peaks_idx)
    amp_sign = -1.0 if is_dip else 1.0
    amp_per_peak = amp_sign * float(np.ptp(signal)) / n
    offset_guess = float(np.nanmax(signal)) if is_dip else float(np.nanmin(signal))

    p0, lo, hi = [], [], []
    for idx in peaks_idx:
        half_level = (smoothed[idx] + sig_mean) / 2.0
        gamma_g = _scan_width(freqs, smoothed, int(idx), half_level, is_peak=not is_dip)
        p0 += [float(freqs[idx]), amp_per_peak, gamma_g]
        lo += [freq_min, -np.inf if is_dip else 0.0, freq_step]
        hi += [freq_max, 0.0 if is_dip else np.inf, freq_range]
    p0.append(offset_guess)
    lo.append(-np.inf)
    hi.append(np.inf)

    def _model(f, *params):
        out = np.full(len(f), params[-1], dtype=float)
        for i in range(n):
            f0, amp, gamma = params[3 * i], params[3 * i + 1], params[3 * i + 2]
            out += amp / (1.0 + ((f - f0) / (gamma / 2.0)) ** 2)
        return out

    try:
        popt, _ = curve_fit(
            _model, freqs, signal, p0=p0, bounds=(lo, hi), maxfev=10000
        )
    except (RuntimeError, ValueError):
        return []

    results = []
    for i in range(n):
        f0, amp, gamma = popt[3 * i], popt[3 * i + 1], popt[3 * i + 2]
        if freq_min <= f0 <= freq_max and freq_step < abs(gamma) < freq_range:
            results.append((float(f0), float(amp), float(abs(gamma))))

    return sorted(results, key=lambda x: x[0])


# ──────────────────────────────────────────────────────────────────────────────
# Peak tracker
# ──────────────────────────────────────────────────────────────────────────────


def _build_tracks(
    peaks_per_slice: list[list[tuple[float, float, float]]],
    n_exchange: int,
    freq_range: float,
    max_peaks: int = 4,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Stitch per-slice peak lists into continuous tracks.

    Uses greedy nearest-neighbour assignment: for each exchange slice all
    (track, peak) pairs are ranked by frequency distance and assigned in
    order, so the closest match wins.  Unmatched peaks start new tracks.

    Parameters
    ----------
    peaks_per_slice : list of length n_exchange.
        Each element is a list of ``(f0, amplitude, gamma)`` tuples (may be
        empty if no peaks were found in that slice).
    n_exchange, freq_range, max_peaks : see caller.

    Returns
    -------
    f0_tracks, amp_tracks : each a list of 1-D arrays of shape (n_exchange,)
        with NaN where the track has no peak assignment.
    """
    threshold = freq_range / max(2 * max_peaks, 4)
    f0_tracks: list[np.ndarray] = []
    amp_tracks: list[np.ndarray] = []

    for slice_idx, peaks in enumerate(peaks_per_slice):
        if not peaks:
            continue

        n_tracks = len(f0_tracks)
        n_new = len(peaks)

        if n_tracks == 0:
            for f0, amp, _ in peaks[:max_peaks]:
                tf0 = np.full(n_exchange, np.nan)
                ta = np.full(n_exchange, np.nan)
                tf0[slice_idx] = f0
                ta[slice_idx] = amp
                f0_tracks.append(tf0)
                amp_tracks.append(ta)
            continue

        # Last valid f0 for each track
        track_last = []
        for tf0 in f0_tracks:
            valid = np.where(np.isfinite(tf0[:slice_idx]))[0]
            track_last.append(float(tf0[valid[-1]]) if len(valid) > 0 else np.nan)

        peak_freqs = [p[0] for p in peaks]

        # Build and sort all (distance, track_idx, peak_idx) pairs
        pairs = []
        for ti, tlast in enumerate(track_last):
            if np.isnan(tlast):
                continue
            for pi, pf in enumerate(peak_freqs):
                d = abs(pf - tlast)
                if d < threshold:
                    pairs.append((d, ti, pi))
        pairs.sort()

        assigned_tracks: set[int] = set()
        assigned_peaks: set[int] = set()
        for _, ti, pi in pairs:
            if ti not in assigned_tracks and pi not in assigned_peaks:
                f0_tracks[ti][slice_idx] = peaks[pi][0]
                amp_tracks[ti][slice_idx] = peaks[pi][1]
                assigned_tracks.add(ti)
                assigned_peaks.add(pi)

        # Unmatched peaks start new tracks (up to max_peaks * 2 total)
        for pi, (f0, amp, _) in enumerate(peaks):
            if pi not in assigned_peaks and len(f0_tracks) < max_peaks * 2:
                tf0 = np.full(n_exchange, np.nan)
                ta = np.full(n_exchange, np.nan)
                tf0[slice_idx] = f0
                ta[slice_idx] = amp
                f0_tracks.append(tf0)
                amp_tracks.append(ta)

    return f0_tracks, amp_tracks


def _select_primary_track(
    f0_tracks: list[np.ndarray],
    amp_tracks: list[np.ndarray],
    min_coverage: float = 0.3,
    n_exchange: int = 1,
) -> np.ndarray:
    """Return the f0 track with the largest median |amplitude|.

    Tracks covering fewer than *min_coverage* of exchange slices are ignored.
    Falls back to the most populated track if all amplitudes are unavailable.
    """
    if not f0_tracks:
        return np.full(n_exchange, np.nan)

    min_valid = max(2, int(min_coverage * n_exchange))
    scores = []
    for tf0, tamp in zip(f0_tracks, amp_tracks):
        valid = np.isfinite(tamp)
        if np.sum(valid) < min_valid:
            scores.append(-np.inf)
        else:
            scores.append(float(np.nanmedian(np.abs(tamp[valid]))))

    best = int(np.argmax(scores))
    return f0_tracks[best].copy()


# ──────────────────────────────────────────────────────────────────────────────
# Exchange-model fitter
# ──────────────────────────────────────────────────────────────────────────────


def _fit_exchange_model(
    exchange_voltages: np.ndarray,
    j_values_hz: np.ndarray,
) -> ExchangeModelFit:
    """Fit J(V) = J_0 * exp((V - V_ref) / lever_arm)."""
    result = ExchangeModelFit()
    valid = np.isfinite(j_values_hz) & (j_values_hz > 0)
    if np.sum(valid) < 3:
        return result
    v, j = exchange_voltages[valid], j_values_hz[valid]
    try:
        slope, intercept = np.polyfit(v, np.log(j), 1)
        lever_arm_guess = 1.0 / slope if abs(slope) > 1e-12 else 0.1

        def _model(v_arr, j0, v_ref, lam):
            return j0 * np.exp((v_arr - v_ref) / lam)

        popt, _ = curve_fit(
            _model, v, j, p0=[np.exp(intercept), 0.0, lever_arm_guess], maxfev=5000
        )
        result.J_0 = float(popt[0])
        result.V_ref = float(popt[1])
        result.lever_arm = float(popt[2])
        result.success = bool(
            np.isfinite(popt).all() and result.J_0 > 0 and result.lever_arm != 0
        )
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        pass
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Core extraction
# ──────────────────────────────────────────────────────────────────────────────


def _extract_crot_params(
    freqs: np.ndarray,
    exchange_voltages: np.ndarray,
    signal_no_x180: np.ndarray,
    signal_with_x180: np.ndarray,
    max_peaks: int = 4,
) -> tuple[CROTFitResult, np.ndarray, np.ndarray]:
    """Extract CROT parameters from two 2-D analysis-signal maps.

    Parameters
    ----------
    freqs : (n_freq,)
    exchange_voltages : (n_exchange,)
    signal_no_x180, signal_with_x180 : (n_exchange, n_freq)
    max_peaks : maximum number of Lorentzians to fit per slice.

    Returns
    -------
    result : CROTFitResult
    f0_down : (n_exchange,)  primary peak track from no-x180 signal
    f0_up   : (n_exchange,)  primary peak track from with-x180 signal
    """
    n_exchange = signal_no_x180.shape[0]
    freq_range = float(np.ptp(freqs))

    # Fit multi-Lorentzian per slice
    peaks_no = [_fit_multi_lorentzian(freqs, signal_no_x180[i], max_peaks) for i in range(n_exchange)]
    peaks_with = [_fit_multi_lorentzian(freqs, signal_with_x180[i], max_peaks) for i in range(n_exchange)]

    # Build continuous tracks
    f0_tracks_no, amp_tracks_no = _build_tracks(peaks_no, n_exchange, freq_range, max_peaks)
    f0_tracks_with, amp_tracks_with = _build_tracks(peaks_with, n_exchange, freq_range, max_peaks)

    # Select the dominant track from each signal
    f0_down = _select_primary_track(f0_tracks_no, amp_tracks_no, n_exchange=n_exchange)
    f0_up = _select_primary_track(f0_tracks_with, amp_tracks_with, n_exchange=n_exchange)

    # MAD-based outlier rejection
    f0_down[~_mad_outlier_mask(f0_down)] = np.nan
    f0_up[~_mad_outlier_mask(f0_up)] = np.nan

    splitting = np.abs(f0_up - f0_down)
    valid = np.isfinite(splitting)

    result = CROTFitResult()
    if not np.any(valid):
        return result, f0_down, f0_up

    # Pick the optimal exchange index: largest splitting among slices where
    # BOTH tracks are clean (avoids selecting outlier-inflated values).
    both_clean = np.isfinite(f0_down) & np.isfinite(f0_up)
    if np.any(both_clean):
        candidate = np.where(both_clean, splitting, -np.inf)
        best_idx = int(np.argmax(candidate))
    else:
        best_idx = int(np.nanargmax(splitting))

    result.optimal_exchange_idx = best_idx
    result.exchange_coupling_J = float(splitting[best_idx])
    result.crot_frequency_down = float(f0_down[best_idx])
    result.crot_frequency_up = float(f0_up[best_idx])
    result.success = bool(
        np.isfinite(result.exchange_coupling_J) and result.exchange_coupling_J > 0
    )
    result.exchange_model = _fit_exchange_model(exchange_voltages[valid], splitting[valid])

    return result, f0_down, f0_up


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def fit_raw_data(
    ds_raw: xr.Dataset,
    qubit_pairs: list[Any],
    *,
    analysis_signal: str = "E_p2_given_p1_0",
    max_peaks: int = 4,
) -> tuple[xr.Dataset, dict[str, dict[str, Any]]]:
    """Run CROT spectroscopy analysis for every qubit pair.

    Parameters
    ----------
    ds_raw : xr.Dataset
        Dataset after ``process_joint_streams`` with coordinates
        ``spectator_x180``, ``exchange``, ``esr_frequency`` and variables
        ``{analysis_signal}_{pair_name}``.
    qubit_pairs : list
        Qubit pair objects (each must have a ``.name`` attribute).
    analysis_signal : str
        Which conditional expectation to fit.
    max_peaks : int
        Maximum number of Lorentzian peaks to fit per frequency slice.
        Use 4 when both qubit resonances fall inside the sweep window.

    Returns
    -------
    ds_fit : xr.Dataset
        Fitted peak positions (``f0_down_<pair>``, ``f0_up_<pair>``) vs exchange.
    fit_results : dict
        Per-pair fit results.
    """
    freqs = ds_raw.coords["esr_frequency"].values.astype(np.float64)
    exchange = ds_raw.coords["exchange"].values.astype(np.float64)
    fit_results: Dict[str, dict[str, Any]] = {}
    fit_vars: dict[str, xr.DataArray] = {}

    for qp in qubit_pairs:
        var_name = f"{analysis_signal}_{qp.name}"
        if var_name not in ds_raw.data_vars:
            logger.warning("No %s variable for pair %s — skipping.", var_name, qp.name)
            fit_results[qp.name] = asdict(CROTFitResult())
            continue

        da = ds_raw[var_name].astype(np.float64)
        if "measured_qubit" in da.dims:
            # The state update (crot_frequency_down/up, exchange_coupling_J)
            # describes the target qubit's conditional resonances, so fit the
            # "drive + measure target" slice. Select by label, falling back to
            # the leading position when labels are unavailable.
            coord = da.coords.get("measured_qubit")
            if coord is not None and "target" in set(np.atleast_1d(coord.values).tolist()):
                da = da.sel(measured_qubit="target")
            else:
                da = da.isel(measured_qubit=0)
        spectator_x180_vals = da.coords["spectator_x180"].values
        idx_no = int(np.argmin(spectator_x180_vals))
        idx_with = int(np.argmax(spectator_x180_vals))
        signal_no_x180 = da.isel(spectator_x180=idx_no).values
        signal_with_x180 = da.isel(spectator_x180=idx_with).values

        result, f0_down, f0_up = _extract_crot_params(
            freqs, exchange, signal_no_x180, signal_with_x180, max_peaks=max_peaks
        )
        result_dict = asdict(result)
        if result.exchange_model is not None:
            result_dict["exchange_model"] = asdict(result.exchange_model)
        fit_results[qp.name] = result_dict

        fit_vars[f"f0_down_{qp.name}"] = xr.DataArray(
            f0_down, dims=["exchange"], attrs={"long_name": "f_↓", "units": "Hz"}
        )
        fit_vars[f"f0_up_{qp.name}"] = xr.DataArray(
            f0_up, dims=["exchange"], attrs={"long_name": "f_↑", "units": "Hz"}
        )

    ds_fit = xr.Dataset(fit_vars, coords={"exchange": exchange})
    return ds_fit, fit_results


def log_fitted_results(
    fit_results: dict[str, dict[str, Any]],
    log_callable: Any | None = None,
) -> None:
    """Log CROT spectroscopy fit results for all qubit pairs."""
    _log = log_callable or logger.info
    for name, r in sorted(fit_results.items()):
        status = "OK" if r["success"] else "FAILED"
        j_mhz = r["exchange_coupling_J"] / 1e6
        msg = (
            f"  {name}: [{status}] J = {j_mhz:.3f} MHz, "
            f"f_↓ = {r['crot_frequency_down'] / 1e6:.3f} MHz, "
            f"f_↑ = {r['crot_frequency_up'] / 1e6:.3f} MHz"
        )
        _log(msg)
        em = r.get("exchange_model")
        if em and em.get("success"):
            _log(
                f"    Exchange model: J_0 = {em['J_0'] / 1e6:.3f} MHz, "
                f"V_ref = {em['V_ref'] * 1e3:.1f} mV, "
                f"lever_arm = {em['lever_arm'] * 1e3:.1f} mV"
            )