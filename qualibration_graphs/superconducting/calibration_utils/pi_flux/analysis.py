# pylint: disable=too-many-arguments,too-many-positional-arguments
"""Analysis utilities for pi vs flux calibration: exponential fitting for flux distortions."""
from typing import Dict, List, Optional, Tuple, TypedDict

import numpy as np
import xarray as xr
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from scipy.optimize import curve_fit, least_squares, minimize

# Internal constants (spectroscopy freq-vs-flux hires curve; not node Parameters).
# Uniform flux samples for CubicSpline evaluation after branch extraction. Increasing N
# improves flux-axis sampling of the spline; cost is O(N) for spline eval and downstream
# numpy ops on the curve—~10x from 2k→20k is still small vs. peak/DP work on the raw map.
SPECTROSCOPY_HIRES_NUM_POINTS = 20000


# --- Fit result (exponential-sum model for flux step response) ---
class FluxDistortionExpFitResult(TypedDict):
    """Result of fitting the flux-line step response as a DC term plus decaying exponentials."""

    fit_successful: bool
    n_components_requested: int
    n_components_used: int
    a_tau_tuple: List[Tuple[float, float]]
    a_dc: float
    rms_error: float


# --- Dataset preprocessing and center-frequency extraction ---
def log_fitted_results(fit_results: Dict[str, FluxDistortionExpFitResult], log_callable=print) -> None:
    """Log the fitted pi vs flux results for each qubit."""
    # Surfaces fit quality and key parameters to the console so operators can spot failed fits before updating state.
    for qb, res in fit_results.items():
        if res["fit_successful"]:
            n_req = res.get("n_components_requested", "?")
            n_used = res.get("n_components_used", len(res["a_tau_tuple"]))
            log_callable(
                f"{qb}: SUCCESS (n_req={n_req}, n_used={n_used}), "
                f"a_dc={res['a_dc']:.3e}, rms={res['rms_error']:.3e}, "
                f"comps={res['a_tau_tuple']}"
            )
        else:
            log_callable(f"{qb}: FAILED")


def process_raw_dataset(ds: xr.Dataset, node) -> xr.Dataset:
    """Convert IQ to V, add amplitude/phase, add freq_full if detuning present."""
    # Normalises the raw hardware output into calibrated voltage units and attaches derived coordinates needed by downstream analysis steps.
    if "I" in ds or "Q" in ds:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])  # type: ignore
        ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    if "detuning" in ds.coords:
        full_freq = np.array([ds.detuning + q.xy.RF_frequency for q in node.namespace["qubits"]])
        ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
        ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def gaussian(x, a, x0, sigma, offset):
    """Gaussian function for peak fitting."""
    # Provides the model curve used to locate the qubit resonance peak within each spectroscopy frequency slice.
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + offset


def fit_gaussian(freqs, states):
    """Fit a Gaussian to extract center frequency from spectroscopy data."""
    # Called per (qubit, time) slice to turn the spectroscopy sweep into a single resonance frequency value.
    p0 = [
        float(np.max(states) - np.min(states)),
        float(freqs[np.argmax(states)]),
        float((freqs[-1] - freqs[0]) / 10),
        float(np.min(states)),
    ]
    try:
        popt, _ = curve_fit(gaussian, freqs, states, p0=p0)
        return float(popt[1])
    except Exception:
        return float("nan")


def extract_center_freqs_state(ds: xr.Dataset, freqs: np.ndarray) -> xr.DataArray:
    """Extract center frequencies from state discrimination dataset as a DataArray.

    Ensures we operate on the 'state' DataArray (not the whole Dataset) so apply_ufunc
    returns a DataArray, avoiding assignment errors downstream.
    """
    # Converts per-timestep state-discrimination spectra into a 2-D (qubit × time) array of resonance frequencies, which is the primary input for the flux-response calculation.
    freq_dim = "detuning" if "detuning" in ds.dims else "freq"
    state_da = ds["state"].transpose("qubit", "time", freq_dim)
    center_freqs = xr.apply_ufunc(
        lambda states: fit_gaussian(freqs, states),
        state_da,
        input_core_dims=[[freq_dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    return center_freqs.rename("center_frequency")


def extract_center_freqs_iq(ds: xr.Dataset, freqs: np.ndarray) -> xr.DataArray:
    """Extract center frequencies from IQ dataset using Gaussian fitting."""
    # Fallback path when state discrimination is not available; uses the I quadrature amplitude envelope to locate each spectroscopy peak.
    # iq_data = ds["IQ_abs"] if "IQ_abs" in ds.data_vars else ds.get("I", None)
    iq_data = ds.get("I", None)
    if iq_data is None:
        raise ValueError("Dataset is missing IQ_abs and I data variables for IQ analysis")
    stacked = iq_data.transpose("qubit", "time", "detuning")
    center_freqs = xr.apply_ufunc(
        lambda iq_slice: fit_gaussian(freqs, iq_slice),
        stacked,
        input_core_dims=[["detuning"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    return center_freqs


# --- Multi-exponential global fit (pi-flux step response) ---
#
# Replaces the old sequential-peeling + start-fractions approach. The user
# specifies only `n_exponentials`; the algorithm:
#   1. Seeds taus log-spaced across the data time range.
#   2. Solves a closed-form linear LS for amplitudes + a_dc at fixed taus.
#   3. Runs a joint nonlinear least-squares fit (scipy.optimize.least_squares,
#      method "trf") on (a_dc, a_i, tau_i) jointly. tau is reparameterised as
#      log(tau) to enforce positivity without bound activations.
#   4. Per-point weighting is 1/sqrt(local log-spacing) when the time grid is
#      log-spaced, uniform otherwise — this prevents densely-sampled early
#      times from dominating the loss.
#   5. Retries with n-1 components on solver failure, near-degenerate taus,
#      or amplitudes below a fraction of the signal swing.
#
# Bounds enforce physically meaningful results:
#   - a_dc constrained to [min(y) - 10%·swing, max(y) + 10%·swing]
#   - tau in [TAU_MIN_NS, TAU_MAX_FACTOR · t_max]
# Without these, the optimiser can trade a_dc against a runaway-tau exponential
# (mathematically equivalent fit, but the resulting IIR coefficients amp/a_dc
# become meaningless for filter generation).

TAU_MIN_NS = 0.1
TAU_MAX_FACTOR = 100.0  # tau_max = TAU_MAX_FACTOR * max(t)
TAU_PULSE_FACTOR = 20.0  # in finite-pulse mode tau_max also capped at TAU_PULSE_FACTOR * T_pulse to keep (1-exp(-T/tau)) above ~5% (otherwise amplitude deconvolution explodes)
ADC_PADDING_FRAC = 0.1  # widen [min(y), max(y)] by ±10%·swing for a_dc bounds


def single_exp_decay(t, amp, tau):
    """Single exponential decay function (legacy helper, kept for backward compat)."""
    return amp * np.exp(-t / tau)


def _detect_log_grid(t: np.ndarray) -> bool:
    """Heuristic: log-spaced if std/mean of log spacings is smaller than that of linear spacings."""
    # Auto-detects whether the user requested a log or linear time sweep so that residual weighting can adapt without an extra parameter.
    if len(t) < 4:
        return False
    pos = t[t > 0]
    if len(pos) < 4:
        return False
    lin_steps = np.diff(t)
    log_steps = np.diff(np.log(pos))
    lin_cv = np.std(lin_steps) / max(np.mean(lin_steps), 1e-12)
    log_cv = np.std(log_steps) / max(np.mean(log_steps), 1e-12)
    return log_cv < lin_cv


def _make_residual_weights(t: np.ndarray) -> np.ndarray:
    """Per-point weights that equalise contribution per log decade for log-spaced grids."""
    # On log-spaced grids, early-time points are densely packed and would dominate a uniform-weight LS fit; sqrt(local log spacing) restores roughly equal weighting per decade so the long-time tail is fit just as well as the short-time rise.
    if not _detect_log_grid(t):
        return np.ones_like(t, dtype=float)
    pos_min = t[t > 0].min()
    log_t = np.log(np.maximum(t, pos_min / 10))
    spacing = np.empty_like(log_t)
    spacing[1:-1] = (log_t[2:] - log_t[:-2]) / 2
    spacing[0] = log_t[1] - log_t[0]
    spacing[-1] = log_t[-1] - log_t[-2]
    return np.sqrt(np.maximum(spacing, 1e-12))


def _seed_taus(t: np.ndarray, n: int, t_pulse_ns: Optional[float] = None) -> np.ndarray:
    """Log-spaced n initial taus over the data time range, padded by factor 3.

    When ``t_pulse_ns`` is given (finite-pulse mode), the upper seed is
    additionally capped at ``TAU_PULSE_FACTOR * t_pulse_ns``. Beyond this
    cap the attenuation factor ``(1 - exp(-T_pulse/tau))`` becomes so small
    that the de-attenuated amplitude is essentially unconstrained by the
    data and the joint fit can run away to nonsense (the deconvolution
    SNR collapses).
    """
    # Provides a robust initial guess that scales with the user's chosen sweep duration; one tau per log decade of the data lets the joint nonlinear stage settle each component into its proper niche. The t_pulse_ns cap is a hard guard against the LTI-deconvolution blow-up mode where (1 - exp(-T/tau)) -> 0 and amp -> infinity become a numerical null direction.
    pos = t[t > 0]
    lo = float(pos.min()) / 3.0
    hi = float(pos.max()) * 10.0
    if t_pulse_ns is not None:
        hi = min(hi, TAU_PULSE_FACTOR * float(t_pulse_ns))
        if hi <= lo:
            hi = lo * 10.0
    if n == 1:
        return np.array([float(np.sqrt(lo * hi))])
    return np.logspace(np.log10(lo), np.log10(hi), n)


def _seed_amplitudes(
    t: np.ndarray,
    y: np.ndarray,
    taus: np.ndarray,
    t_pulse_ns: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Closed-form linear LS for (a_dc, a_1, ..., a_n) given fixed taus.

    Default model (``t_pulse_ns is None``):
        ``y = a_dc + Σ a_i exp(-t/tau_i)``

    Finite-pulse model (``t_pulse_ns`` given, Aggarwal arXiv:2503.08645 Eq. 17,
    used by the Ramsey-tail nodes 19b / 21c):
        ``y = a_dc + Σ a_i (1 - exp(-T_pulse/tau_i)) exp(-t/tau_i)``

    The returned ``a_i`` are the **true** step-response amplitudes, i.e. the
    finite-pulse attenuation factor (1 - exp(-T_pulse/tau_i)) is already
    deconvolved. Returns ``(amps, a_dc)``.
    """
    # Solving the linear sub-problem at the seed taus gives the joint nonlinear stage a near-optimal starting point essentially for free, dramatically improving convergence.
    # When t_pulse_ns is provided, basis columns include (1-exp(-T/tau_i)) so that the linear seed already estimates de-attenuated amplitudes; otherwise the seed would be off by orders of magnitude for tau_i >> T_pulse and the nonlinear stage would have to climb a much steeper hill.
    n = len(taus)
    M = np.empty((len(t), n + 1))
    M[:, 0] = 1.0
    for i, tau in enumerate(taus):
        basis = np.exp(-t / tau)
        if t_pulse_ns is not None:
            basis = basis * (1.0 - np.exp(-float(t_pulse_ns) / tau))
        M[:, i + 1] = basis
    coefs, *_ = np.linalg.lstsq(M, y, rcond=None)
    return coefs[1:], float(coefs[0])


def _fit_once(
    t: np.ndarray,
    y: np.ndarray,
    n: int,
    weights: np.ndarray,
    t_pulse_ns: Optional[float] = None,
) -> Dict:
    """Single attempt at a joint multi-exponential fit with n components.

    Parameters are packed as ``[a_dc, a_1, log_tau_1, ..., a_n, log_tau_n]``.
    The log-tau encoding keeps tau positive without explicit bound activations
    (which can make the trust-region solver hit boundary stalls).

    When ``t_pulse_ns`` is given, the model includes the finite-pulse
    attenuation factor ``(1 - exp(-T_pulse/tau_i))`` (Aggarwal arXiv:2503.08645
    Eq. 17). The returned ``a_i`` are the true step-response amplitudes (i.e.
    the attenuation factor is already deconvolved by the fit), so downstream
    IIR coefficients ``A_i = a_i / a_dc`` need no further correction.
    """
    # Returns a result dict that the caller (multi_exp_fit_global) can decide to accept or retry on degeneracy/failure.
    taus_seed = _seed_taus(t, n, t_pulse_ns=t_pulse_ns)
    amps_seed, adc_seed = _seed_amplitudes(t, y, taus_seed, t_pulse_ns=t_pulse_ns)

    log_tau_lo = np.log(TAU_MIN_NS)
    tau_hi = TAU_MAX_FACTOR * float(t.max())
    if t_pulse_ns is not None:
        tau_hi = min(tau_hi, TAU_PULSE_FACTOR * float(t_pulse_ns))
    log_tau_hi = np.log(max(tau_hi, TAU_MIN_NS * 10.0))
    swing = float(np.ptp(y))
    adc_lo = float(np.min(y)) - ADC_PADDING_FRAC * swing
    adc_hi = float(np.max(y)) + ADC_PADDING_FRAC * swing

    x0 = np.empty(1 + 2 * n)
    x0[0] = np.clip(adc_seed, adc_lo, adc_hi)
    for i in range(n):
        x0[1 + 2 * i] = amps_seed[i]
        x0[2 + 2 * i] = np.clip(np.log(max(taus_seed[i], TAU_MIN_NS)), log_tau_lo, log_tau_hi)

    lb = np.full_like(x0, -np.inf)
    ub = np.full_like(x0, np.inf)
    lb[0] = adc_lo
    ub[0] = adc_hi
    for i in range(n):
        lb[2 + 2 * i] = log_tau_lo
        ub[2 + 2 * i] = log_tau_hi

    t_pulse = None if t_pulse_ns is None else float(t_pulse_ns)

    def model(x):
        a_dc = x[0]
        out = np.full_like(t, a_dc, dtype=float)
        for i in range(n):
            amp = x[1 + 2 * i]
            tau = np.exp(x[2 + 2 * i])
            term = amp * np.exp(-t / tau)
            if t_pulse is not None:
                term = term * (1.0 - np.exp(-t_pulse / tau))
            out += term
        return out

    def residuals(x):
        return (model(x) - y) * weights

    try:
        result = least_squares(residuals, x0=x0, bounds=(lb, ub), method="trf", max_nfev=2000)
    except (RuntimeError, ValueError) as e:
        return {"success": False, "error": str(e), "n": n}

    x_opt = result.x
    a_dc_opt = float(x_opt[0])
    components = [(float(x_opt[1 + 2 * i]), float(np.exp(x_opt[2 + 2 * i]))) for i in range(n)]
    y_pred = model(x_opt)
    rms = float(np.sqrt(np.mean((y_pred - y) ** 2)))

    return {
        "success": bool(result.success),
        "n": n,
        "a_dc": a_dc_opt,
        "components": components,
        "rms": rms,
    }


# pylint: disable-next=too-many-branches,too-many-statements
def multi_exp_fit_global(
    t: np.ndarray,
    y: np.ndarray,
    n_exponentials: int,
    rel_amp_threshold: float = 0.02,
    tau_proximity_factor: float = 1.5,
    verbose: bool = True,
    t_pulse_ns: Optional[float] = None,
) -> Dict:
    """Global multi-exponential fit of a flux step / Ramsey-tail response.

    Default model (``t_pulse_ns is None``, used by the qubit-spectroscopy
    pipeline 19a which directly samples the step rise):
        ``y(t) = a_dc + Σ a_i exp(-t/tau_i)``     [Mooij arXiv:2503.04610 Eq. 1]

    Finite-pulse model (``t_pulse_ns`` given, used by the Ramsey-tail nodes
    19b / 21c which measure the residual after a finite long flux pulse of
    duration ``T_pulse = t_pulse_ns``):
        ``y(t) = a_dc + Σ a_i (1 - exp(-T_pulse/tau_i)) exp(-t/tau_i)``
                                            [Aggarwal arXiv:2503.08645 Eq. 17]

    Critically, the returned ``a_tau_tuple`` amplitudes ``a_i`` are the
    **true** (de-attenuated) step-response amplitudes in both modes, so the
    downstream IIR coefficient formula ``A_i = a_i / a_dc`` is identical for
    19a / 19b / 21c without further correction.

    The user only specifies ``n_exponentials``. Initial taus are log-spaced
    across the data range; amplitudes are seeded by a closed-form linear LS
    (with the same attenuation factor when ``t_pulse_ns`` is given);
    a joint nonlinear LS then refines all parameters together.

    Retry-with-fewer-components triggers (drop one component, re-fit with n-1):
      A: solver fails to converge.
      B: two taus within ``tau_proximity_factor`` of each other (near-degenerate).
      C: ``|amp_i| < rel_amp_threshold * (max(y) - min(y))`` (negligible vs signal swing).

    Returns a dict with keys: ``fit_successful``, ``n_components_requested``,
    ``n_components_used``, ``a_tau_tuple``, ``a_dc``, ``rms_error``.
    """
    # Replaces the previous sequential-peeling approach (which forced the user to pick start_fractions per component). This is far more robust because the user's only knob (n_exponentials) is interpretable, and the joint nonlinear fit handles cross-talk between timescales that the peeling approach handled poorly.
    weights = _make_residual_weights(t)
    n_req = int(n_exponentials)
    n_cur = max(1, n_req)
    signal_swing = max(float(np.ptp(y)), 1e-12)

    while n_cur >= 1:
        res = _fit_once(t, y, n_cur, weights, t_pulse_ns=t_pulse_ns)

        if not res["success"]:
            if verbose:
                print(f"  multi_exp: n={n_cur} solver failed → retry n-1")
            n_cur -= 1
            continue

        comps = res["components"]
        a_dc = res["a_dc"]

        # Trigger C: prune negligible amplitudes (relative to signal swing)
        keep = [i for i, (amp, _) in enumerate(comps) if abs(amp) >= rel_amp_threshold * signal_swing]
        if len(keep) < len(comps):
            if verbose:
                dropped = [i for i in range(len(comps)) if i not in keep]
                print(
                    f"  multi_exp: n={n_cur} components {dropped} have "
                    f"|amp| < {rel_amp_threshold:.0%}·signal_swing → retry "
                    f"with n={len(keep)}"
                )
            if len(keep) == 0:
                return {
                    "fit_successful": False,
                    "n_components_requested": n_req,
                    "n_components_used": 0,
                    "a_tau_tuple": [],
                    "a_dc": a_dc,
                    "rms_error": res["rms"],
                }
            n_cur = len(keep)
            continue

        # Trigger B: prune one of two near-degenerate taus (drop smaller-|amp|)
        order = np.argsort([tau for _, tau in comps])
        degenerate = False
        for k in range(len(order) - 1):
            i, j = int(order[k]), int(order[k + 1])
            if comps[j][1] / max(comps[i][1], 1e-12) < tau_proximity_factor:
                drop = i if abs(comps[i][0]) < abs(comps[j][0]) else j
                if verbose:
                    print(
                        f"  multi_exp: n={n_cur} taus {comps[i][1]:.2f},"
                        f"{comps[j][1]:.2f} ns within factor "
                        f"{tau_proximity_factor} → drop comp {drop}, retry"
                    )
                n_cur -= 1
                degenerate = True
                break
        if degenerate:
            continue

        # Sort components by descending tau for stable downstream ordering
        comps_sorted = sorted(comps, key=lambda c: -c[1])
        return {
            "fit_successful": True,
            "n_components_requested": n_req,
            "n_components_used": len(comps_sorted),
            "a_tau_tuple": comps_sorted,
            "a_dc": a_dc,
            "rms_error": res["rms"],
        }

    return {
        "fit_successful": False,
        "n_components_requested": n_req,
        "n_components_used": 0,
        "a_tau_tuple": [],
        "a_dc": float("nan"),
        "rms_error": float("nan"),
    }


# --- Spectroscopy 2D map: DP branch tracking ---
# (ported from coupler-flux notebook 10; simplified for single-branch qubit-flux spectroscopy)

_SPEC_DP_CFG = dict(
    max_candidates_per_column=3,
    min_peak_prominence=0.0002,
    min_peak_separation_hz=2e6,
    jump_penalty_weight=1.0,
    slope_penalty_weight=0.5,
    intensity_reward_weight=1.0,
    prominence_reward_weight=0.5,
    gap_penalty_weight=0.8,
    max_gap_columns=3,
    low_confidence_threshold=0.2,
)


def _preprocess_intensity_map(IQ_abs_2d: np.ndarray):
    """Per-column MAD normalization of a (n_detuning, n_flux) intensity map."""
    # Removes flux-dependent baseline offsets so that peak-finding thresholds are consistent across different flux bias columns.
    raw = IQ_abs_2d.copy()
    col_med = np.nanmedian(raw, axis=0, keepdims=True)
    col_mad = 1.4826 * np.nanmedian(np.abs(raw - col_med), axis=0, keepdims=True)
    col_mad = np.maximum(col_mad, 1e-12)
    return raw, (raw - col_med) / col_mad


def _extract_peak_candidates(freq_axis, intensity_col, cfg):
    """Multi-candidate peak extraction with prominence + SNR scoring."""
    # Produces the set of candidate resonance frequencies for one flux column that the DP tracker will then link into a continuous branch.
    from scipy.signal import find_peaks

    K = cfg.get("max_candidates_per_column", 3)
    min_prom = cfg.get("min_peak_prominence", 0.0002)
    min_sep_hz = cfg.get("min_peak_separation_hz", 2e6)
    freq_step = abs(freq_axis[1] - freq_axis[0]) if len(freq_axis) > 1 else 1.0
    min_dist = max(1, int(min_sep_hz / freq_step))

    col = intensity_col.copy()
    col_med = np.nanmedian(col)
    col_mad = max(1.4826 * np.nanmedian(np.abs(col - col_med)), 1e-15)

    try:
        peaks, props = find_peaks(col, prominence=min_prom, distance=min_dist, width=1)
    except Exception:
        peaks, props = np.array([], dtype=int), {"prominences": np.array([]), "widths": np.array([])}

    if len(peaks) == 0:
        idx = int(np.argmax(col))
        return [
            dict(
                freq=float(freq_axis[idx]),
                intensity=float(col[idx]),
                prominence=float(col[idx] - col_med),
                snr=float((col[idx] - col_med) / col_mad),
                rank=0,
            )
        ]

    order = np.argsort(-props["prominences"])[:K]
    return [
        dict(
            freq=float(freq_axis[peaks[oi]]),
            intensity=float(col[peaks[oi]]),
            prominence=float(props["prominences"][oi]),
            snr=float((col[peaks[oi]] - col_med) / col_mad),
            rank=rank,
        )
        for rank, oi in enumerate(order)
    ]


def _track_branch_dp(flux_axis, freq_axis, all_candidates, cfg):
    """Viterbi-style DP branch tracker — returns best path through candidates."""
    # Connects per-column peak candidates into a globally optimal continuous branch by penalising large frequency jumps, slope discontinuities, and gaps, so the correct qubit mode is tracked even with avoided crossings.
    n_cols = len(flux_axis)
    jump_w = cfg.get("jump_penalty_weight", 1.0)
    slope_w = cfg.get("slope_penalty_weight", 0.5)
    int_w = cfg.get("intensity_reward_weight", 1.0)
    prom_w = cfg.get("prominence_reward_weight", 0.5)
    gap_pen = cfg.get("gap_penalty_weight", 0.8)
    max_gap = cfg.get("max_gap_columns", 3)
    freq_range = max(abs(freq_axis[-1] - freq_axis[0]), 1.0)

    all_int = [c["intensity"] for col_c in all_candidates for c in col_c]
    int_scale = max(np.percentile(all_int, 95) - np.percentile(all_int, 5), 1e-12) if all_int else 1.0
    int_med = float(np.median(all_int)) if all_int else 0.0
    all_prom = [c["prominence"] for col_c in all_candidates for c in col_c]
    prom_scale = max(np.percentile(all_prom, 95), 1e-12) if all_prom else 1.0

    def reward(c):
        return int_w * (c["intensity"] - int_med) / int_scale + prom_w * c["prominence"] / prom_scale

    INF = -1e18
    dp_score = [np.full(len(all_candidates[j]), INF) for j in range(n_cols)]
    dp_back = [[None] * len(all_candidates[j]) for j in range(n_cols)]

    for k, c in enumerate(all_candidates[0]):
        dp_score[0][k] = reward(c)

    for j in range(1, n_cols):
        for k, cand in enumerate(all_candidates[j]):
            nr = reward(cand)
            best_val, best_ptr = INF, None
            for g in range(min(j, max_gap + 1)):
                jp = j - 1 - g
                if jp < 0:
                    break
                gc = gap_pen * g if g > 0 else 0.0
                dx = float(flux_axis[j] - flux_axis[jp])
                for kp, cp in enumerate(all_candidates[jp]):
                    jump_cost = jump_w * abs(cand["freq"] - cp["freq"]) / freq_range
                    slope_cost = 0.0
                    bp = dp_back[jp][kp]
                    if bp is not None:
                        jpp, kpp = bp
                        dx_prev = float(flux_axis[jp] - flux_axis[jpp])
                        if dx > 0 and dx_prev > 0:
                            s1 = (cand["freq"] - cp["freq"]) / dx
                            s0 = (cp["freq"] - all_candidates[jpp][kpp]["freq"]) / dx_prev
                            slope_cost = slope_w * abs(s1 - s0) / freq_range * min(dx, dx_prev)
                    val = dp_score[jp][kp] + nr - jump_cost - slope_cost - gc
                    if val > best_val:
                        best_val, best_ptr = val, (jp, kp)
            dp_score[j][k] = best_val
            dp_back[j][k] = best_ptr

    # Backtrack
    j, k = n_cols - 1, int(np.argmax(dp_score[n_cols - 1]))
    path = []
    while j is not None:
        path.append((j, all_candidates[j][k]["freq"], dp_score[j][k]))
        bp = dp_back[j][k]
        if bp is None:
            break
        j, k = bp
    path.reverse()

    tracked_flux = np.array([float(flux_axis[p[0]]) for p in path])
    tracked_freq = np.array([p[1] for p in path])
    cumul = np.array([p[2] for p in path])
    increments = np.diff(cumul, prepend=0.0)
    inc_min, inc_max = increments.min(), increments.max()
    conf = (increments - inc_min) / (inc_max - inc_min) if inc_max > inc_min else np.ones(len(increments))

    return dict(tracked_flux=tracked_flux, tracked_freq=tracked_freq, tracked_confidence=conf, n_points=len(path))


def _clean_spectroscopy_branch(
    flux: np.ndarray,
    freq: np.ndarray,
    resid_sigma: float = 3.0,
    resid_floor_hz: float = 5e6,
    deriv_jump_sigma: float = 4.0,
    edge_frac: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove outliers and tail-rebound artefacts from a DP-tracked branch.

    Works for both parabolic and cosine-like shapes by enforcing
    *smoothness* rather than monotonicity.

    Steps:
      1. Detect abrupt derivative jumps; if near an edge, truncate that tail
         (catches DP tracker latching onto a different feature at flux extremes).
      2. Savitzky-Golay smooth on the truncated data + residual outlier removal
         (catches isolated stray peaks in the interior).
    """
    # Removes flux-edge artefacts and isolated stray peaks from the DP-tracked branch before spline interpolation, ensuring the final curve is smooth enough for accurate frequency-to-flux inversion.
    from scipy.signal import savgol_filter

    n = len(flux)
    if n < 6:
        return flux, freq

    # --- 1. Edge-only derivative-jump tail truncation ---
    dflux = np.diff(flux)
    dfreq = np.diff(freq)
    safe_dflux = np.where(np.abs(dflux) < 1e-12, 1e-12, dflux)
    deriv = dfreq / safe_dflux
    dderiv = np.abs(np.diff(deriv))

    med_dd = np.median(dderiv)
    mad_dd = np.median(np.abs(dderiv - med_dd))
    sigma_dd = max(1.4826 * mad_dd, 1e9)
    thresh = deriv_jump_sigma * sigma_dd

    n_dd = len(dderiv)
    edge_n = max(3, int(n_dd * edge_frac))

    lo, hi = 0, n - 1
    for k in range(min(edge_n, n_dd)):
        if dderiv[k] > thresh:
            lo = k + 2
    for k in range(n_dd - 1, max(n_dd - 1 - edge_n, -1), -1):
        if dderiv[k] > thresh:
            hi = min(hi, k + 1)

    lo = min(lo, n - 4)
    hi = max(hi, lo + 3)
    flux, freq = flux[lo : hi + 1], freq[lo : hi + 1]

    if len(flux) < 6:
        return flux, freq

    # --- 1b. Remove isolated points (large flux gap on both sides) ---
    if len(flux) > 4:
        gaps = np.abs(np.diff(flux))
        med_gap = np.median(gaps)
        left_gap = np.concatenate([[gaps[0]], gaps])
        right_gap = np.concatenate([gaps, [gaps[-1]]])
        isolated = (left_gap > 3 * med_gap) & (right_gap > 3 * med_gap)
        if isolated.sum() < len(flux) - 3:
            flux, freq = flux[~isolated], freq[~isolated]

    if len(flux) < 6:
        return flux, freq

    # --- 2. Iterative SavGol smooth + residual outlier removal ---
    survivors = np.ones(len(flux), dtype=bool)
    for _iter in range(5):
        f_s = flux[survivors]
        q_s = freq[survivors]
        m = len(f_s)
        if m < 6:
            break
        win = min(m, max(7, 2 * (m // 8) + 1))
        if win % 2 == 0:
            win += 1
        poly_order = min(3, win - 1)
        smooth = savgol_filter(q_s, window_length=win, polyorder=poly_order)
        resid = q_s - smooth
        med_r = np.median(resid)
        mad = np.median(np.abs(resid - med_r))
        sigma_r = max(1.4826 * mad, resid_floor_hz)
        inner_keep = np.abs(resid - med_r) <= resid_sigma * sigma_r
        if inner_keep.all():
            break
        survivors[np.where(survivors)[0][~inner_keep]] = False

    flux_c, freq_c = flux[survivors], freq[survivors]
    if len(flux_c) < 4:
        return flux, freq
    return flux_c, freq_c


def _extract_spectroscopy_curve(
    ds_raw, qubit_name: str, qubit_rf_freq: float, cfg: Optional[dict] = None
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract high-resolution freq-vs-flux curve from raw spectroscopy data.

    Pipeline: MAD-normalize -> multi-candidate peak extraction ->
    DP branch tracking -> confidence filter -> smoothness cleaning ->
    CubicSpline interpolation.

    Returns (flux_hires, freq_hires), each 1D of length ``SPECTROSCOPY_HIRES_NUM_POINTS``,
    or None on failure.
    """
    # Orchestrates the full spectroscopy-map-to-curve pipeline; the resulting high-resolution curve is used as the frequency-to-flux lookup table for computing the flux step response.
    from scipy.interpolate import CubicSpline

    cfg = {**_SPEC_DP_CFG, **(cfg or {})}
    conf_thresh = cfg.get("low_confidence_threshold", 0.2)

    iq_abs = ds_raw.IQ_abs.sel(qubit=qubit_name).values  # (n_det, n_flux)
    flux_axis = ds_raw.flux_bias.values
    freq_axis = ds_raw.full_freq.sel(qubit=qubit_name).values  # (n_det,) absolute Hz

    _, map_norm = _preprocess_intensity_map(iq_abs)

    all_candidates = [_extract_peak_candidates(freq_axis, map_norm[:, j], cfg) for j in range(len(flux_axis))]

    branch = _track_branch_dp(flux_axis, freq_axis, all_candidates, cfg)
    if branch["n_points"] < 4:
        return None

    keep = branch["tracked_confidence"] >= conf_thresh
    flux_kept = branch["tracked_flux"][keep]
    freq_kept = branch["tracked_freq"][keep]
    if len(flux_kept) < 4:
        return None

    sort_idx = np.argsort(flux_kept)
    flux_s, freq_s = flux_kept[sort_idx], freq_kept[sort_idx]

    flux_s, freq_s = _clean_spectroscopy_branch(flux_s, freq_s)
    if len(flux_s) < 4:
        return None

    # Final smoothing of control points to remove point-to-point noise
    if len(flux_s) >= 11:
        from scipy.signal import savgol_filter as _sgf

        win_s = min(len(flux_s), max(11, 2 * (len(flux_s) // 6) + 1))
        if win_s % 2 == 0:
            win_s += 1
        freq_s = _sgf(freq_s, window_length=win_s, polyorder=3)

    flux_hires = np.linspace(flux_s[0], flux_s[-1], SPECTROSCOPY_HIRES_NUM_POINTS)
    return flux_hires, CubicSpline(flux_s, freq_s)(flux_hires)


# --- Qualibrate node loaders (two experiment types) ---
# ``_load_spectroscopy_curve``: qubit spectroscopy vs flux run (``ds_raw``) for this pi-flux pipeline.
# ``_load_dispersion_curve``: separate dispersion / Ramsey calibration via
# ``extras["{coupler}_dispersion_load_id"]`` (``ds_fit``).


def _read_node_data_dict(run_id: int) -> dict:
    """Return the dict from ``read_node_data`` for a completed node run (storage path from config)."""
    # Resolves the storage path from the Qualibrate config and loads all saved datasets for the given node run, enabling cross-node data sharing without manual path management.
    from qualibrate.core.utils.node.content import read_node_data
    from qualibrate.core.utils.node.path_solver import get_node_dir_path
    from qualibrate_config.resolvers import get_qualibrate_config, get_qualibrate_config_path

    base_path = get_qualibrate_config(get_qualibrate_config_path()).storage.location
    node_dir = get_node_dir_path(run_id, base_path)
    return read_node_data(node_dir, run_id, base_path)


def _load_spectroscopy_curve(
    run_id: int, qubit_name: str, qubit_rf_freq: float
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load a **qubit spectroscopy vs flux** node by ``run_id`` and build the hires freq--flux curve.

    Uses ``ds_raw`` from that run with :func:`_extract_spectroscopy_curve` (not the dispersion/Ramsey loader).
    """
    # Fetches a previously saved qubit-spectroscopy-vs-flux node and runs the full DP branch-extraction pipeline to produce the high-resolution curve needed for frequency-to-flux mapping.
    try:
        data = _read_node_data_dict(run_id)
        ds_raw = data["ds_raw"]
        curve = _extract_spectroscopy_curve(ds_raw, qubit_name, qubit_rf_freq)
        if curve is not None:
            print(
                f"  Loaded spectroscopy curve for {qubit_name} from run #{run_id}: "
                f"{len(curve[0])} pts, flux=[{curve[0][0]:.4f}, {curve[0][-1]:.4f}] V"
            )
        return curve
    except Exception as e:
        print(f"  WARNING: Failed to load spectroscopy curve for {qubit_name} " f"from run #{run_id}: {e}")
        return None


def _flux_amp_from_curve(
    detuning_hz: float,
    idle_freq_hz: float,
    curve_flux: np.ndarray,
    curve_freq: np.ndarray,
) -> Optional[float]:
    """Inverse lookup: given a detuning from idle frequency, find the Z-pulse amplitude.

    Finds all zero-crossings of (curve_freq − target_freq), then returns the
    flux distance from idle for the nearest crossing.  Works for any curve
    shape (parabolas, avoided crossings, etc.).

    Returns None if the target frequency has no crossing on the curve
    (caller should fall back to quad_term).
    """
    # Converts a measured frequency detuning into the flux-pulse amplitude required to reach it, using the spectroscopy curve as the reference instead of a simple quadratic model.
    target_freq = idle_freq_hz - abs(detuning_hz)

    idle_flux = float(curve_flux[int(np.argmin(np.abs(curve_freq - idle_freq_hz)))])

    diff = curve_freq - target_freq
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        return None

    best_amp = None
    for i in sign_changes:
        f1, f2 = curve_freq[i], curve_freq[i + 1]
        x1, x2 = curve_flux[i], curve_flux[i + 1]
        frac = (target_freq - f1) / (f2 - f1) if abs(f2 - f1) > 0 else 0.0
        amp = abs(x1 + frac * (x2 - x1) - idle_flux)
        if best_amp is None or amp < best_amp:
            best_amp = amp

    return float(best_amp)


def _derive_flux_amp(
    detuning_hz: float,
    idle_freq_hz: float,
    curve: Tuple[np.ndarray, np.ndarray],
    branch: Optional[str] = None,
) -> Optional[float]:
    """Derive signed flux offset for a target detuning on a given branch.

    Unlike :func:`_flux_amp_from_curve` (which picks the nearest crossing
    regardless of branch), this function can restrict the search to one side of
    the freq-vs-flux curve.

    Parameters
    ----------
    detuning_hz : float
        Positive detuning in Hz below idle frequency.
    idle_freq_hz : float
        Qubit idle frequency (Hz).
    curve : tuple of (flux_array, freq_array)
        High-resolution freq-vs-flux curve.
    branch : str or None
        ``"right"`` (flux >= idle flux), ``"left"`` (flux <= idle flux),
        or ``None`` to search the full curve (for monotonic coupler curves).

    Returns
    -------
    float or None
        Flux offset from idle, or None if the target frequency is not found.
    """
    target_freq = idle_freq_hz - abs(detuning_hz)
    curve_flux, curve_freq = curve

    idle_idx = int(np.argmin(np.abs(curve_freq - idle_freq_hz)))
    idle_flux = float(curve_flux[idle_idx])

    if branch is not None:
        if branch == "right":
            mask = curve_flux >= idle_flux
        else:
            mask = curve_flux <= idle_flux
        b_flux = curve_flux[mask]
        b_freq = curve_freq[mask]
    else:
        # Full curve search — for monotonic curves with no sweetspot
        b_flux = curve_flux
        b_freq = curve_freq
    if len(b_flux) < 2:
        return None

    diff = b_freq - target_freq
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        return None

    # Pick the first crossing on this branch
    i = sign_changes[0]
    f1, f2 = b_freq[i], b_freq[i + 1]
    x1, x2 = b_flux[i], b_flux[i + 1]
    frac = (target_freq - f1) / (f2 - f1) if abs(f2 - f1) > 0 else 0.0
    flux_val = x1 + frac * (x2 - x1)
    return float(flux_val - idle_flux)  # signed: + for right, − for left


def _load_dispersion_curve(
    node, qubit, coupler, frequency_var: str = "abs_peak_frequency"
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load qubit frequency vs coupler flux from a **separate** dispersion / Ramsey calibration run.

    Identified by ``qubit.extras["<coupler>_dispersion_load_id"]`` (not ``spectroscopy_run_id``).
    Parses ``ds_fit`` from that node. Layouts:

    * **Ramsey-style** (e.g. ramsey-vs-coupler-flux): ``coupler_flux`` dimension, ``qubit_frequency``.
    * **Legacy** coupler-flux spectroscopy: ``coupler_flux`` + ``frequency_var`` indexed by ``qubit_pair``.

    Parameters
    ----------
    node : QualibrationNode
        Used for ``node.machine.qubit_pairs`` when selecting ``qubit_pair``.
    qubit : Qubit
        Target qubit.
    coupler : Coupler
        Coupler whose dispersion was measured.
    frequency_var : str, optional
        Frequency variable name in the **legacy** layout (ignored for Ramsey-style).

    Returns
    -------
    (flux_bias, frequency) as 1D arrays, or None if load_id is missing or load fails.
    """
    # Provides the coupler-flux dispersion curve from a dedicated calibration run stored in qubit.extras, acting as a fallback when no spectroscopy or Ramsey run ID is specified for the current node.
    load_id = qubit.extras.get(f"{coupler.name}_dispersion_load_id") if hasattr(qubit, "extras") else None
    print(f"Loading dispersion curve for {qubit.name} and {coupler.name} with load_id {load_id}")
    if load_id is None:
        return None
    try:
        data = _read_node_data_dict(load_id)
        ds_fit = data["ds_fit"]

        if "qubit_frequency" in ds_fit.data_vars and "coupler_flux" in ds_fit.dims:
            flux_bias = ds_fit.coupler_flux.values
            if "qubit_pair" in ds_fit.dims:
                qp_names_in_ds = [str(qp) for qp in ds_fit.qubit_pair.values]
                qubit_match = None
                for pair_name, pair in node.machine.qubit_pairs.items():
                    if pair.coupler.name == coupler.name and str(pair_name) in qp_names_in_ds:
                        qubit_match = str(pair_name)
                        break
                if qubit_match is None:
                    print(f"No qubit match found for {qubit.name} and {coupler.name}")
                    return None
                frequency = ds_fit["qubit_frequency"].sel(qubit_pair=qubit_match).values
            else:
                frequency = ds_fit["qubit_frequency"].values
            return (flux_bias, frequency)

        flux_bias = ds_fit["coupler_flux"].values
        frequency = ds_fit[frequency_var].sel(qubit_pair=pair.name).values
        return (flux_bias, frequency)
    except Exception as e:
        print(f"Error loading dispersion curve: {e}")
        return None


# --- Frequency--flux mapping (coupler dispersion vs qubit-Z branch) ---


def _frequency_to_flux(
    measured_abs_freq: np.ndarray,
    flux_bias: np.ndarray,
    abs_peak_frequency: np.ndarray,
    n_flux_fine: int = 1000,
) -> np.ndarray:
    """Map measured absolute qubit frequency to reconstructed coupler flux via interpolated dispersion curve.

    Builds a continuous mapping by:
    1. Sorting the curve by flux_bias and interpolating frequency onto a fine flux grid.
    2. Inverting to get flux(freq) and interpolating measured frequencies onto that inverse (with
       duplicate frequencies from non-monotonic regions resolved by keeping the first flux).
    Output flux is clipped to [flux_bias.min(), flux_bias.max()].
    """
    # Converts the time-series of measured qubit frequencies into equivalent coupler flux values, which is the key step for reconstructing the flux step response from spectroscopy data.
    flux_min, flux_max = flux_bias.min(), flux_bias.max()
    idx_s = np.argsort(flux_bias)
    flux_s = flux_bias[idx_s]
    freq_s = abs_peak_frequency[idx_s]

    flux_fine = np.linspace(flux_min, flux_max, n_flux_fine)
    freq_fine = np.interp(flux_fine, flux_s, freq_s)

    idx_inv = np.argsort(freq_fine)
    freq_sorted = freq_fine[idx_inv]
    flux_sorted = flux_fine[idx_inv]
    freq_unique, idx_unique = np.unique(freq_sorted, return_index=True)
    flux_unique = flux_sorted[idx_unique]

    measured_flat = np.asarray(measured_abs_freq, dtype=float).ravel()
    out = np.interp(
        measured_flat,
        freq_unique,
        flux_unique,
        left=flux_min,
        right=flux_max,
    )
    out = np.clip(out, flux_min, flux_max)
    return out.reshape(np.shape(measured_abs_freq))


def _frequency_to_flux_deviation(
    measured_abs_freq: np.ndarray,
    curve_flux: np.ndarray,
    curve_freq: np.ndarray,
    idle_freq: float,
    use_upper_branch: bool = True,
) -> np.ndarray:
    """Map measured absolute qubit frequency to flux *deviation* from idle.

    Unlike ``_frequency_to_flux`` this function:
    * subtracts the idle-point flux so the result is a signed deviation,
    * selects the correct monotonic branch (flux >= idle_flux) for
      Z-pulse experiments,
    * uses CubicSpline with extrapolation for modest out-of-range
      frequencies (returns NaN only for clearly non-physical values).
    """
    # Used for the qubit Z-flux branch to produce a signed flux deviation (step response) rather than an absolute flux, which is the physical quantity that the distortion correction filters act on.
    from scipy.interpolate import CubicSpline as _CS

    idle_idx = int(np.argmin(np.abs(curve_freq - idle_freq)))
    idle_flux = float(curve_flux[idle_idx])

    branch_mask = curve_flux >= idle_flux if use_upper_branch else curve_flux <= idle_flux
    b_flux = curve_flux[branch_mask]
    b_freq = curve_freq[branch_mask]

    if len(b_flux) < 4:
        return np.full_like(measured_abs_freq, np.nan, dtype=float)

    sort_idx = np.argsort(b_freq)
    freq_asc = b_freq[sort_idx]
    flux_asc = b_flux[sort_idx]

    cs_inv = _CS(freq_asc, flux_asc, extrapolate=True)
    measured_flat = np.asarray(measured_abs_freq, dtype=float).ravel()
    mapped = cs_inv(measured_flat)
    result = mapped - idle_flux

    flux_range = float(b_flux.max() - idle_flux)
    bad = (result < -0.01) | (result > 2.0 * flux_range)
    result[bad] = np.nan
    return result.reshape(np.shape(measured_abs_freq))


# --- fit_raw_data orchestration ---


def _assign_coords_qubit_flux(
    ds: xr.Dataset,
    center_freqs: xr.DataArray,
    dfs: np.ndarray,
    qubits: list,
) -> Tuple[xr.Dataset, xr.DataArray]:
    """Qubit-flux nodes: attach freq_full, detuning, and quadratic flux placeholder coords."""
    # Attaches freq_full, detuning, and quadratic-flux placeholder coordinates to the dataset so that downstream plotting and fitting functions have consistent axis labels. With the new flux-domain sweep, dfs = freq_points - f_idle so freq_full = dfs + f_idle gives exact absolute frequencies.
    ds = ds.assign_coords(
        {
            "freq_full": (
                ["qubit", "detuning"],
                np.array([dfs + q.xy.RF_frequency for q in qubits]),
            ),
            "detuning": (
                ["qubit", "detuning"],
                np.array([dfs for _ in qubits]),
            ),
            "flux": (
                ["qubit", "detuning"],
                np.array(
                    [
                        (
                            np.sqrt(np.maximum(0.0, dfs / q.freq_vs_flux_01_quad_term))
                            if getattr(q, "freq_vs_flux_01_quad_term", None) and q.freq_vs_flux_01_quad_term != 0
                            else np.full_like(dfs, np.nan, dtype=float)
                        )
                        for q in qubits
                    ]
                ),
            ),
        }
    )
    return ds, center_freqs


def _load_ramsey_curve(
    run_id: Optional[int],
    qubit_name: str,
    qubit_rf_freq: Optional[float] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load qubit freq vs Z-flux curve from a Ramsey-vs-flux calibration run (node 23).

    Reads ``ds_fit.unfolded_frequency`` (GHz, dims: qubit × flux_bias) which
    contains the **Ramsey oscillation frequency** (detuning from the drive),
    and converts it to **absolute qubit frequency** in Hz via::

        f_qubit = RF_frequency + artificial_detuning - ramsey_freq

    NaN values (e.g. failed fits at some flux points) are removed before returning.

    Parameters
    ----------
    run_id : int or None
        Node-23 run ID.
    qubit_name : str
        Qubit identifier.
    qubit_rf_freq : float or None
        Qubit idle frequency (Hz).  Required for the detuning→absolute conversion.
        When ``None``, the function falls back to returning raw Ramsey oscillation
        frequencies (legacy behaviour, will likely produce wrong results downstream).

    Returns
    -------
    (flux_bias, frequency_hz) as 1-D arrays, or None if run_id is None or load fails.
    """
    if run_id is None:
        return None
    print(f"  Loading Ramsey curve for {qubit_name} from run #{run_id}")
    try:
        data = _read_node_data_dict(run_id)
        ds_fit = data["ds_fit"]
        flux_bias = ds_fit.flux_bias.values

        if "f_qubit_vs_flux" in ds_fit:
            freq_hz = ds_fit["f_qubit_vs_flux"].sel(qubit=qubit_name).values * 1e9  # GHz → Hz
        else:
            ramsey_freq_hz = ds_fit["unfolded_frequency"].sel(qubit=qubit_name).values * 1e9  # GHz → Hz
            if qubit_rf_freq is not None:
                artif_det_hz = 0.0
                if "artifitial_detuning" in ds_fit:
                    artif_det_hz = float(ds_fit["artifitial_detuning"].sel(qubit=qubit_name).values) * 1e6
                freq_hz = qubit_rf_freq + artif_det_hz - ramsey_freq_hz
            else:
                freq_hz = ramsey_freq_hz

        mask = ~np.isnan(freq_hz)
        if not np.any(mask):
            print(f"  WARNING: All NaN in Ramsey curve for {qubit_name} from run #{run_id}")
            return None
        print(
            f"  Loaded Ramsey curve for {qubit_name}: {mask.sum()} pts, "
            f"flux=[{flux_bias[mask][0]:.4f}, {flux_bias[mask][-1]:.4f}] V, "
            f"freq=[{freq_hz[mask].min()/1e9:.4f}, {freq_hz[mask].max()/1e9:.4f}] GHz"
        )
        return flux_bias[mask], freq_hz[mask]
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"  WARNING: Failed to load Ramsey curve for {qubit_name} from run #{run_id}: {e}")
        return None


def _flux_response_qubit_flux_branch(
    center_freqs: xr.DataArray,
    qubits: list,
    use_spec: bool,
    spec_run_id: Optional[int],
    use_ramsey: bool = False,
    ramsey_run_id: Optional[int] = None,
    flux_amp_for_detuning: Optional[float] = None,
) -> Tuple[xr.DataArray, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Spectroscopy hires curve, Ramsey curve, or quadratic ``freq_vs_flux_01_quad_term`` fallback.

    Priority order:
    1. Spectroscopy vs Z-flux (``use_spec=True`` + ``spec_run_id``)
    2. Ramsey vs Z-flux / node-23 (``use_ramsey=True`` + ``ramsey_run_id``)
    3. Quadratic ``freq_vs_flux_01_quad_term`` from QUAM state
    """
    # Selects the best available freq-to-flux mapping for each qubit and applies it to the measured center frequencies to generate the flux step response time series.
    spec_curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    flux_response = xr.DataArray(
        np.full_like(center_freqs.values, np.nan),
        coords=center_freqs.coords,
        dims=center_freqs.dims,
    )

    for i, q in enumerate(qubits):
        curve = None
        # Path 1: spectroscopy vs Z-flux
        if use_spec and spec_run_id is not None:
            curve = _load_spectroscopy_curve(spec_run_id, q.name, q.xy.RF_frequency)
            if curve is not None:
                spec_curves[q.name] = curve
        # Path 2: Ramsey vs Z-flux (node 23)
        if curve is None and use_ramsey and ramsey_run_id is not None:
            curve = _load_ramsey_curve(ramsey_run_id, q.name, q.xy.RF_frequency)
        # Path 3: quad_term fallback
        if curve is not None:
            abs_freq_q = center_freqs.sel(qubit=q.name).values + q.xy.RF_frequency
            # Determine branch: compare flux_amp_for_detuning to the idle flux on the curve
            use_upper_branch = True
            if flux_amp_for_detuning is not None:
                idle_idx = int(np.argmin(np.abs(curve[1] - q.xy.RF_frequency)))
                use_upper_branch = float(flux_amp_for_detuning) >= float(curve[0][idle_idx])
            flux_response.values[i, :] = _frequency_to_flux_deviation(
                abs_freq_q,
                curve[0],
                curve[1],
                q.xy.RF_frequency,
                use_upper_branch=use_upper_branch,
            )
        else:
            qt = getattr(q, "freq_vs_flux_01_quad_term", None) or np.nan
            if np.isfinite(qt) and qt != 0:
                flux_response.values[i, :] = np.sqrt(center_freqs.sel(qubit=q.name).values / qt)

    return flux_response, spec_curves


def _assign_coords_coupler_flux(ds: xr.Dataset, dfs: np.ndarray, qubits: list) -> xr.Dataset:
    """Coupler-flux nodes: dispersion-only coords (no static detuning shift)."""
    # Attaches the same set of axis coordinates as the qubit-flux variant but without any detuning offset, because coupler-flux experiments observe qubit frequency changes via coupler-mediated dispersion rather than direct Z-drive.
    return ds.assign_coords(
        {
            "freq_full": (
                ["qubit", "detuning"],
                np.array([dfs + q.xy.RF_frequency for q in qubits]),
            ),
            "detuning": (
                ["qubit", "detuning"],
                np.array([dfs for _ in qubits]),
            ),
            "flux": (
                ["qubit", "detuning"],
                np.full((len(qubits), len(dfs)), np.nan, dtype=float),
            ),
        }
    )


def _flux_response_coupler_branch(
    center_freqs: xr.DataArray,
    node,
    qubits: list,
    qubit_pairs: list,
) -> xr.DataArray:
    """Map center frequencies to coupler flux using :func:`_load_dispersion_curve` and :func:`_frequency_to_flux`."""
    # Converts each qubit's measured resonance frequency time series into the corresponding coupler flux step response using the per-coupler dispersion curve loaded from a prior calibration run.
    abs_freq = center_freqs + xr.DataArray(
        [q.xy.RF_frequency for q in qubits],
        coords={"qubit": [q.name for q in qubits]},
        dims=["qubit"],
    )
    flux_response = xr.full_like(center_freqs, np.nan, dtype=float)
    for i, q in enumerate(qubits):
        if i >= len(qubit_pairs):
            break
        coupler = qubit_pairs[i].coupler
        curve = _load_dispersion_curve(node, q, coupler)
        if curve is not None:
            flux_bias, abs_peak = curve
            abs_freq_q = abs_freq.sel(qubit=q.name).values
            flux_response.values[i, :] = _frequency_to_flux(abs_freq_q, flux_bias, abs_peak)

    coupler_flux_amp = getattr(node.parameters, "coupler_flux_amplitude_in_v", None)
    if coupler_flux_amp is not None:
        flux_response = flux_response + coupler_flux_amp
    return flux_response


def _attach_spec_curve_vars(
    ds: xr.Dataset,
    spec_curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    spec_run_id: Optional[int],
) -> xr.Dataset:
    """Attach hires spectroscopy curves and run id when spectroscopy path populated ``spec_curves``."""
    # Saves the extracted spectroscopy calibration curves into the output dataset so they are available for plotting and for reproducing the analysis from saved data.
    if not spec_curves:
        return ds
    qubit_names_sc = list(spec_curves.keys())
    ds["spec_curve_flux"] = xr.DataArray(
        np.array([spec_curves[qn][0] for qn in qubit_names_sc]),
        dims=["spec_qubit", "spec_pts"],
        coords={"spec_qubit": qubit_names_sc},
    )
    ds["spec_curve_freq"] = xr.DataArray(
        np.array([spec_curves[qn][1] for qn in qubit_names_sc]),
        dims=["spec_qubit", "spec_pts"],
        coords={"spec_qubit": qubit_names_sc},
    )
    ds.attrs["spectroscopy_run_id"] = int(spec_run_id)
    return ds


def _fit_pi_flux_all_qubits(
    flux_response: xr.DataArray,
    qubits: list,
    n_exponentials: int,
) -> Dict[str, FluxDistortionExpFitResult]:
    """Run :func:`multi_exp_fit_global` per qubit and collect fit results."""
    # Loops over all qubits and runs the global multi-exponential fit on each qubit's flux step response, collecting results into a dict keyed by qubit name for downstream state update. NaN/zero-time samples are filtered before fitting to keep the log-time weighting well-defined.
    fit_results: Dict[str, FluxDistortionExpFitResult] = {}
    for q in qubits:
        qf = flux_response.sel(qubit=q.name)
        t_data = np.asarray(qf.time.values, dtype=float)
        y_data = np.asarray(qf.values, dtype=float)
        mask = np.isfinite(y_data) & (t_data > 0)
        if mask.sum() < max(2 * n_exponentials + 1, 4):
            fit_results[q.name] = {
                "fit_successful": False,
                "n_components_requested": int(n_exponentials),
                "n_components_used": 0,
                "a_tau_tuple": [],
                "a_dc": float("nan"),
                "rms_error": float("nan"),
            }
            continue
        fit_results[q.name] = multi_exp_fit_global(t_data[mask], y_data[mask], n_exponentials, verbose=True)
    return fit_results


def fit_raw_data(ds: xr.Dataset, node) -> tuple[xr.Dataset, Dict[str, FluxDistortionExpFitResult]]:
    """Compute center_freqs, detuning/freq_full/flux coords, flux_response, and fit cascade."""
    # Top-level analysis entry point called by the calibration node: extracts resonance frequencies, maps them to flux, and fits the exponential distortion model for each qubit.
    qubits = node.namespace["qubits"]

    dfs = (
        node.namespace.get("sweep_axes", {}).get("detuning").values
        if "sweep_axes" in node.namespace
        else (ds.get("detuning").values if "detuning" in ds.dims else ds.get("freq").values)
    )

    if "detuning" not in ds.dims and "freq" in ds.dims:
        ds = ds.rename({"freq": "detuning"})

    if node.parameters.use_state_discrimination and "state" in ds.data_vars:
        center_freqs = extract_center_freqs_state(ds, dfs)
    else:
        center_freqs = extract_center_freqs_iq(ds, dfs)

    qubit_pairs = node.namespace.get("qubit_pairs")
    spec_run_id = getattr(node.parameters, "spectroscopy_run_id", None)

    if qubit_pairs is None:
        ds, center_freqs = _assign_coords_qubit_flux(ds, center_freqs, dfs, qubits)
        use_spec = getattr(node.parameters, "use_spectroscopy_data", False)
        use_ramsey = getattr(node.parameters, "use_ramsey_data", False)
        ramsey_run_id = getattr(node.parameters, "ramsey_run_id", None)
        flux_amp_for_detuning = node.namespace.get("flux_amp_for_detuning")
        if flux_amp_for_detuning is None:
            flux_amp_for_detuning = getattr(node.parameters, "flux_amp_for_detuning_in_volt", None)
            # Backward compat: old runs stored flux_sweep_min/max_volt
            if flux_amp_for_detuning is None:
                fmin = getattr(node.parameters, "flux_sweep_min_volt", None)
                fmax = getattr(node.parameters, "flux_sweep_max_volt", None)
                if fmin is not None and fmax is not None:
                    flux_amp_for_detuning = (fmin + fmax) / 2
        flux_response, spec_curves = _flux_response_qubit_flux_branch(
            center_freqs,
            qubits,
            use_spec,
            spec_run_id,
            use_ramsey=use_ramsey,
            ramsey_run_id=ramsey_run_id,
            flux_amp_for_detuning=flux_amp_for_detuning,
        )
    else:
        ds = _assign_coords_coupler_flux(ds, dfs, qubits)
        spec_curves = {}
        flux_response = _flux_response_coupler_branch(center_freqs, node, qubits, qubit_pairs)

    ds = ds.copy()
    ds["center_freqs"] = center_freqs
    ds["flux_response"] = flux_response
    ds = _attach_spec_curve_vars(ds, spec_curves, spec_run_id)

    fit_results = _fit_pi_flux_all_qubits(flux_response, qubits, node.parameters.n_exponentials)
    return ds, fit_results


# %% {Analyze_data}
def _to_python(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    # Needed because fit_results contains numpy scalars and arrays that are not directly JSON-serialisable when saving node outputs.
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_python(v) for v in obj)
    return obj


# ---------------------------------------------------------------------------
# Legacy fitting helpers (kept for reference only)
# ---------------------------------------------------------------------------
# These two functions were the previous public API for multi-exponential
# fitting, driven by a hand-tuned start_fractions list. They are no longer
# imported by any calibration node — every distortion node (qubit and coupler,
# qubitspec / ramsey / short) now uses multi_exp_fit_global() above, driven by a
# single n_exponentials parameter. Retained only for historical reference;
# do not call these from new code.


def sequential_exp_fit(
    t: np.ndarray,
    y: np.ndarray,
    start_fractions: List[float],
    fixed_taus: List[float] = None,
    a_dc: float = None,
    verbose: bool = 1,
) -> Tuple[List[Tuple[float, float]], float, np.ndarray]:
    """LEGACY: sequential exponential peeling fit (use multi_exp_fit_global instead)."""
    components = []
    t_offset = t - t[0]

    window = max(5, len(y) // 20)
    rolling_var = np.array([np.var(y[i : i + window]) for i in range(len(y) - window)])
    var_threshold = np.mean(rolling_var) * 0.1
    if a_dc is None:
        try:
            flat_start = np.where(rolling_var < var_threshold)[0][-1]
            a_dc = np.mean(y[flat_start:])
        except IndexError:
            print("No flat region found, using last point of the signal as constant term")
            a_dc = y[-1]

    if verbose:
        print(f"\nFitted constant term: {a_dc:.3e}")

    y_residual = y.copy() - a_dc

    for i, start_frac in enumerate(start_fractions):
        start_idx = int(len(t) * start_frac)
        if verbose:
            print(f"\nFitting component {i + 1} using data from t = {t[start_idx]:.1f} ns (fraction: {start_frac:.3f})")
        try:
            if fixed_taus is not None:
                tau_fixed = fixed_taus[i]
                p0 = [y_residual[start_idx]]
                if verbose:
                    print(f"Using fixed tau = {tau_fixed:.3f} ns")
                t_fit = t_offset[start_idx:]
                y_fit = y_residual[start_idx:]
                popt, _ = curve_fit(lambda t, amp: single_exp_decay(t, amp, tau_fixed), t_fit, y_fit, p0=p0)
                amp = popt[0]
                tau = tau_fixed
            else:
                p0 = [y_residual[start_idx], t_offset[start_idx] / 3]
                bounds = ([-np.inf, 0.1], [np.inf, np.inf])
                t_fit = t_offset[start_idx:]
                y_fit = y_residual[start_idx:]
                popt, _ = curve_fit(single_exp_decay, t_fit, y_fit, p0=p0, bounds=bounds)
                amp, tau = popt
            components.append((amp, tau))
            if verbose:
                tau_status = "(fixed)" if fixed_taus is not None else ""
                print(f"Found component: amplitude = {amp:.3e}, tau = {tau:.3f} ns {tau_status}")
            y_residual -= amp * np.exp(-t_offset / tau)
        except (RuntimeError, ValueError) as e:
            if verbose:
                print(f"Warning: Fitting failed for component {i + 1}: {e}")
            break

    return components, a_dc, y_residual


def optimize_start_fractions(t, y, start_fractions, bounds_scale=0.5, fixed_taus=None, a_dc=None, verbose=1):
    """LEGACY: optimise start_fractions via Nelder-Mead (use multi_exp_fit_global instead)."""
    if fixed_taus is not None:
        if len(fixed_taus) != len(start_fractions):
            raise ValueError("fixed_taus must have the same length as start_fractions")
        if any(tau <= 0 for tau in fixed_taus):
            raise ValueError("All fixed_taus values must be positive")

    def objective(x):
        if not np.all(np.diff(x) < 0):
            return 1e6
        components, _, residual = sequential_exp_fit(t, y, x, fixed_taus=fixed_taus, a_dc=a_dc, verbose=0)
        if len(components) == len(start_fractions):
            return np.sqrt(np.mean(residual**2))
        return 1e6

    bounds = [(s * (1 - bounds_scale), s * (1 + bounds_scale)) for s in start_fractions]
    if verbose > 0:
        print("\nOptimizing start_fractions using scipy.optimize.minimize...")
        print(f"Initial values: {[f'{f:.5f}' for f in start_fractions]}")
        print(f"Bounds: ±{bounds_scale * 100}% around initial values")

    result = minimize(
        objective,
        x0=start_fractions,
        bounds=bounds,
        method="Nelder-Mead",
        options={"disp": True, "maxiter": 200},
    )

    if result.success:
        best_fractions = result.x
        components, a_dc, best_residual = sequential_exp_fit(
            t, y, best_fractions, fixed_taus=fixed_taus, a_dc=a_dc, verbose=False
        )
        best_rms = np.sqrt(np.mean(best_residual**2))
        if verbose > 0:
            print("\nOptimization successful!")
            print(f"Initial fractions: {[f'{f:.5f}' for f in start_fractions]}")
            print(f"Optimized fractions: {[f'{f:.5f}' for f in best_fractions]}")
            if fixed_taus is not None:
                print(f"Fixed taus: {[f'{tau:.3f} ns' for tau in fixed_taus]}")
            print(f"Final RMS: {best_rms:.3e}")
            print(f"Number of iterations: {result.nit}")
    else:
        if verbose > 0:
            print("\nOptimization failed. Using initial values.")
        best_fractions = start_fractions
        components, a_dc, best_residual = sequential_exp_fit(
            t, y, best_fractions, fixed_taus=fixed_taus, a_dc=a_dc, verbose=False
        )
        best_rms = np.sqrt(np.mean(best_residual**2))

    components = [(amp * np.exp(t[0] / tau), tau) for amp, tau in components]
    if verbose > 0:
        print("Optimized components [(a1, tau1), (a2, tau2)...]:")
        print(components)
    return result.success, best_fractions, components, a_dc, best_rms
