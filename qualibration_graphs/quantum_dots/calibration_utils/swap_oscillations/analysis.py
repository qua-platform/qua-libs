"""Analysis for the 2-D swap oscillation amplitude × duration sweep.

For each exchange amplitude V, the target (or control) qubit population
oscillates sinusoidally as a function of exchange duration — the SWAP
oscillation.  The dominant frequency is extracted via FFT and converted
to the full 2π oscillation period T(V).

After the per-amplitude FFT extraction, an exponential model is fitted
to the valid (V, T_2π) points:

    T_2π(V) = a · exp(−k · V) + c

This captures the physics that exchange coupling J grows exponentially
with barrier gate voltage, so the oscillation period drops exponentially.
The model parameters {a, k, c} are stored on the ``BalancedCz2QMacro``
so downstream nodes can evaluate T_2π(V) at any amplitude.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict, field
from typing import Any

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


@dataclass
class SwapOscillationFitResult:
    """Per-pair result of the swap oscillation analysis."""

    amplitudes_valid: list[float] = field(default_factory=list)
    """Amplitudes where a valid 2π period was extracted (V)."""
    t_2pi_valid: list[float] = field(default_factory=list)
    """2π oscillation times at each valid amplitude (ns)."""
    snr_valid: list[float] = field(default_factory=list)
    """SNR of the FFT peak at each valid amplitude."""
    role_selected: list[str] = field(default_factory=list)
    """Which signal ('control'/'target'/'difference') was selected at each valid amplitude."""
    best_amplitude: float = float("nan")
    """Amplitude with the highest SNR among valid extractions (V)."""
    best_t_2pi: float = float("nan")
    """2π oscillation time at the best amplitude (ns)."""
    best_snr: float = 0.0
    """SNR at the best amplitude."""
    n_valid: int = 0
    """Number of amplitudes with valid 2π extraction."""
    exchange_decay_model: dict = field(default_factory=dict)
    """Fitted polynomial model ``{"coeffs": [...], "degree": int}``."""
    model_fit_success: bool = False
    """Whether the polynomial model fit converged."""
    success: bool = False
    """True if at least one valid extraction was found."""


def _extract_oscillation_period(
    signal: np.ndarray,
    dt: float,
    *,
    snr_threshold: float = 5.0,
    min_oscillations: float = 0.5,
) -> tuple[float, float, bool]:
    """Extract the dominant oscillation period from a 1-D time trace.

    Uses FFT with parabolic peak interpolation for sub-bin frequency
    resolution.

    Parameters
    ----------
    signal : 1-D array
        Time-domain signal (e.g. qubit population vs exchange duration).
    dt : float
        Uniform time step between samples (ns).
    snr_threshold : float
        Minimum ratio of peak FFT power to median noise power.
    min_oscillations : float
        Reject if the extracted period implies fewer than this many
        oscillations within the observation window.

    Returns
    -------
    period : float
        Dominant oscillation period in the same units as *dt* (NaN on failure).
    snr : float
        Signal-to-noise ratio of the FFT peak.
    success : bool
        Whether the extraction passed all quality checks.
    """
    n = len(signal)
    if n < 8:
        return np.nan, 0.0, False

    signal_ac = signal - np.mean(signal)

    if np.std(signal_ac) < 1e-10:
        return np.nan, 0.0, False

    n_padded = max(n, 4 * n)
    fft_vals = np.fft.rfft(signal_ac, n=n_padded)
    freqs = np.fft.rfftfreq(n_padded, d=dt)

    power = np.abs(fft_vals[1:]) ** 2
    freqs = freqs[1:]

    if len(power) < 3:
        return np.nan, 0.0, False

    peak_idx = int(np.argmax(power))
    peak_power = float(power[peak_idx])

    mask = np.ones(len(power), dtype=bool)
    exclude_lo = max(0, peak_idx - 2)
    exclude_hi = min(len(power), peak_idx + 3)
    mask[exclude_lo:exclude_hi] = False
    noise_power = float(np.median(power[mask])) if mask.sum() > 3 else 1e-20
    snr = peak_power / max(noise_power, 1e-20)

    # Parabolic interpolation for sub-bin frequency resolution
    peak_freq = float(freqs[peak_idx])
    if 0 < peak_idx < len(power) - 1:
        alpha = np.log(power[peak_idx - 1] + 1e-30)
        beta = np.log(power[peak_idx] + 1e-30)
        gamma = np.log(power[peak_idx + 1] + 1e-30)
        denom = alpha - 2 * beta + gamma
        if abs(denom) > 1e-30:
            delta = 0.5 * (alpha - gamma) / denom
            df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0
            peak_freq += delta * df

    if peak_freq <= 0:
        return np.nan, float(snr), False

    period = 1.0 / peak_freq
    total_window = n * dt
    if period > total_window / min_oscillations:
        return np.nan, float(snr), False

    ok = snr >= snr_threshold
    return float(period), float(snr), ok


def _eval_poly(coeffs: list[float], v: float) -> float:
    """Evaluate a polynomial using Horner's method (pure Python, no numpy).

    *coeffs* is ordered highest-degree-first, matching ``np.polyfit`` /
    ``np.polyval`` convention: ``[c_n, c_{n-1}, ..., c_1, c_0]``.
    """
    result = 0.0
    for c in coeffs:
        result = result * v + c
    return result


def _fit_exchange_decay_model(
    amplitudes: np.ndarray,
    t_2pi: np.ndarray,
    snr: np.ndarray | None = None,
    *,
    min_points: int = 4,
    max_degree: int = 5,
) -> tuple[dict, bool]:
    """Fit T_2π(V) with a polynomial up to *max_degree*.

    Uses SNR-weighted ``np.polyfit``.  The best degree (1 … *max_degree*)
    is selected by choosing the highest degree whose coefficients are all
    well-conditioned (weighted R² > 0.9, or the highest degree tried).

    Parameters
    ----------
    amplitudes : 1-D array
        Valid amplitudes (V).
    t_2pi : 1-D array
        Corresponding 2π periods (ns).
    snr : 1-D array, optional
        Per-point SNR values used as fit weights.
    min_points : int
        Minimum data points required.
    max_degree : int
        Maximum polynomial degree (default 3).

    Returns
    -------
    model : dict
        ``{"coeffs": [c_n, ..., c_0], "degree": int}`` on success.
        Coefficients are highest-degree-first (``np.polyval`` convention).
    success : bool
    """
    if len(amplitudes) < min_points:
        logger.warning(
            "Too few points for polynomial fit (%d < %d).",
            len(amplitudes), min_points,
        )
        return {}, False

    sort_idx = np.argsort(amplitudes)
    v = amplitudes[sort_idx].astype(np.float64)
    t = t_2pi[sort_idx].astype(np.float64)
    w = np.sqrt(snr[sort_idx].astype(np.float64)) if snr is not None else None

    ss_tot = float(np.sum((t - np.mean(t)) ** 2))
    if ss_tot < 1e-20:
        return {}, False

    best_coeffs = None
    best_degree = 0
    best_r2 = -np.inf

    for deg in range(1, max_degree + 1):
        if len(v) <= deg:
            continue
        try:
            coeffs = np.polyfit(v, t, deg, w=w)
            residuals = t - np.polyval(coeffs, v)
            ss_res = float(np.sum(residuals ** 2))
            r2 = 1.0 - ss_res / ss_tot

            if r2 > best_r2:
                best_r2 = r2
                best_coeffs = coeffs
                best_degree = deg
        except (np.linalg.LinAlgError, ValueError):
            continue

    if best_coeffs is None or best_r2 < 0.5:
        logger.warning("Polynomial fit poor: best R² = %.3f.", best_r2)
        return {}, False

    model = {
        "type": "polynomial",
        "coeffs": [float(c) for c in best_coeffs],
        "degree": best_degree,
    }
    return model, True


def _build_candidate_signals(
    ds_raw: xr.Dataset,
    pair_name: str,
) -> dict[str, np.ndarray]:
    """Collect all available 2-D signal arrays for a qubit pair.

    Returns a dict mapping role name → (n_amplitudes, n_durations) array.
    Only roles whose data variables are present in *ds_raw* are included.
    """
    ctrl_key = f"state_control_{pair_name}"
    tgt_key = f"state_target_{pair_name}"
    candidates: dict[str, np.ndarray] = {}

    if ctrl_key in ds_raw.data_vars:
        candidates["control"] = ds_raw[ctrl_key].values.astype(np.float64)
    if tgt_key in ds_raw.data_vars:
        candidates["target"] = ds_raw[tgt_key].values.astype(np.float64)
    if "control" in candidates and "target" in candidates:
        candidates["difference"] = candidates["target"] - candidates["control"]

    return candidates


def analyse_swap_oscillations(
    ds_raw: xr.Dataset,
    qubit_pairs: list[Any],
    *,
    snr_threshold: float = 5.0,
    analysis_role: str = "best",
    min_oscillations: float = 0.5,
) -> tuple[xr.Dataset, dict[str, dict[str, Any]]]:
    """Extract 2π oscillation times from the 2-D swap oscillation data.

    For each amplitude row, the dominant oscillation period is found via
    FFT of the time-domain trace.  Low-SNR rows (no clear oscillation)
    are discarded.

    When ``analysis_role="best"`` (the default), every available signal
    (control, target, and their difference) is analysed at each amplitude
    and the result with the highest SNR wins.  This maximises the number
    of valid amplitudes and gives the most reliable T_2π curve.

    Parameters
    ----------
    ds_raw : xr.Dataset
        Raw dataset with dims ``(exchange_amplitude, exchange_duration)``
        and variables ``state_control_{pair}`` / ``state_target_{pair}``.
    qubit_pairs : list
        Qubit-pair objects with ``.name`` attribute.
    snr_threshold : float
        Minimum FFT peak-to-noise ratio.
    analysis_role : str
        ``"best"`` (default) analyses all signals and picks the highest
        SNR per amplitude.  ``"target"``, ``"control"``, or
        ``"difference"`` restrict to a single signal.
    min_oscillations : float
        Minimum number of full oscillation cycles within the observation
        window for a valid extraction.

    Returns
    -------
    ds_fit : xr.Dataset
        Contains ``t_2pi_{pair}``, ``snr_{pair}``, ``valid_{pair}``
        arrays indexed by ``exchange_amplitude``.
    fit_results : dict
        Maps pair name → :class:`SwapOscillationFitResult` (as dict).
    """
    amplitudes = ds_raw.coords["exchange_amplitude"].values.astype(np.float64)
    durations = ds_raw.coords["exchange_duration"].values.astype(np.float64)
    dt = float(durations[1] - durations[0]) if len(durations) > 1 else 1.0

    fit_results: dict[str, dict[str, Any]] = {}
    fit_vars: dict[str, xr.DataArray] = {}

    for qp in qubit_pairs:
        all_candidates = _build_candidate_signals(ds_raw, qp.name)

        if not all_candidates:
            logger.warning("No signal data found for %s; skipping.", qp.name)
            fit_results[qp.name] = asdict(SwapOscillationFitResult())
            continue

        if analysis_role == "best":
            candidates = all_candidates
        else:
            if analysis_role not in all_candidates:
                logger.warning(
                    "Requested role '%s' not available for %s; skipping.",
                    analysis_role, qp.name,
                )
                fit_results[qp.name] = asdict(SwapOscillationFitResult())
                continue
            candidates = {analysis_role: all_candidates[analysis_role]}

        n_amps = next(iter(candidates.values())).shape[0]
        t_2pi_arr = np.full(n_amps, np.nan)
        snr_arr = np.zeros(n_amps)
        valid_arr = np.zeros(n_amps, dtype=bool)
        role_arr: list[str] = [""] * n_amps

        for i in range(n_amps):
            best_period = np.nan
            best_snr = -1.0
            best_ok = False
            best_role = ""

            for role_name, data_2d in candidates.items():
                period, snr, ok = _extract_oscillation_period(
                    data_2d[i, :], dt,
                    snr_threshold=snr_threshold,
                    min_oscillations=min_oscillations,
                )
                if ok and snr > best_snr:
                    best_period = period
                    best_snr = snr
                    best_ok = True
                    best_role = role_name
                elif not best_ok and not np.isnan(period) and snr > best_snr:
                    best_period = period
                    best_snr = snr
                    best_role = role_name

            t_2pi_arr[i] = best_period
            snr_arr[i] = best_snr
            valid_arr[i] = best_ok
            role_arr[i] = best_role

        amps_valid = amplitudes[valid_arr].tolist()
        t_2pi_valid = t_2pi_arr[valid_arr].tolist()
        snr_valid = snr_arr[valid_arr].tolist()
        role_valid = [role_arr[j] for j in range(n_amps) if valid_arr[j]]

        if len(amps_valid) > 0:
            best_idx = int(np.argmax(snr_valid))
            model, model_ok = _fit_exchange_decay_model(
                np.array(amps_valid),
                np.array(t_2pi_valid),
                np.array(snr_valid),
            )
            result = SwapOscillationFitResult(
                amplitudes_valid=amps_valid,
                t_2pi_valid=t_2pi_valid,
                snr_valid=snr_valid,
                role_selected=role_valid,
                best_amplitude=amps_valid[best_idx],
                best_t_2pi=t_2pi_valid[best_idx],
                best_snr=snr_valid[best_idx],
                n_valid=len(amps_valid),
                exchange_decay_model=model,
                model_fit_success=model_ok,
                success=True,
            )
        else:
            result = SwapOscillationFitResult()

        fit_results[qp.name] = asdict(result)

        fit_vars[f"t_2pi_{qp.name}"] = xr.DataArray(
            t_2pi_arr,
            dims=["exchange_amplitude"],
            coords={"exchange_amplitude": amplitudes},
            attrs={"long_name": "2π oscillation period", "units": "ns"},
        )
        fit_vars[f"snr_{qp.name}"] = xr.DataArray(
            snr_arr,
            dims=["exchange_amplitude"],
            coords={"exchange_amplitude": amplitudes},
            attrs={"long_name": "FFT peak SNR", "units": ""},
        )
        fit_vars[f"valid_{qp.name}"] = xr.DataArray(
            valid_arr,
            dims=["exchange_amplitude"],
            coords={"exchange_amplitude": amplitudes},
            attrs={"long_name": "valid 2π extraction", "units": ""},
        )

    ds_fit = xr.Dataset(
        fit_vars,
        coords={"exchange_amplitude": amplitudes},
    )
    return ds_fit, fit_results


def log_fitted_results(
    fit_results: dict[str, dict[str, Any]],
    log_callable: Any | None = None,
) -> None:
    """Log swap oscillation analysis results for all qubit pairs."""
    _log = log_callable or logger.info
    for name, r in sorted(fit_results.items()):
        status = "OK" if r["success"] else "FAILED"
        _log(
            f"  {name}: [{status}] "
            f"{r['n_valid']} valid amplitudes, "
            f"best V = {r['best_amplitude']:.4f} V, "
            f"T_2π = {r['best_t_2pi']:.1f} ns, "
            f"SNR = {r['best_snr']:.1f}"
        )
        m = r.get("exchange_decay_model", {})
        if m and r.get("model_fit_success"):
            coeffs = m["coeffs"]
            deg = m.get("degree", len(coeffs) - 1)
            terms = []
            for i, c in enumerate(coeffs):
                power = deg - i
                if power == 0:
                    terms.append(f"{c:.2f}")
                elif power == 1:
                    terms.append(f"{c:.2f}·V")
                else:
                    terms.append(f"{c:.2f}·V^{power}")
            _log(f"    model (deg {deg}): T_2π(V) = {' + '.join(terms)}  ns")
        elif r["success"]:
            _log("    model: polynomial fit did not converge")
