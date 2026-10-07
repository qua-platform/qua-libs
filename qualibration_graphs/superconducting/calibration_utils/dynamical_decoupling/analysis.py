import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from scipy.optimize import least_squares

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V


ALPHA_BOUNDS = (1.0, 3.0)  # stretch exponent: 1 = exponential, 2 = Gaussian
ALPHA_RELIABLE_MAX = 1.5  # above this, the first-order noise spectrum is unreliable
MIN_COHERENCE_FOR_MEASURED_ERROR = 0.2  # below this after M windows, use the fitted error per round
MAX_RELATIVE_ERROR_FOR_SELECTION = 0.5  # noisier errors per round cannot be optimal
FLOOR_SHIFT_SIGMA = 3.0  # decay floor vs fewest-pulse curve: flag if > 3 sigma ...
FLOOR_SHIFT_MIN_FRACTION = 0.05  # ... and > 5% of the amplitude
NOISE_FIT_F_REF_HZ = 1e6  # reference frequency of S_f(f) = A (f_ref / f)^beta + C


@dataclass
class FitParameters:
    """Stores the relevant dynamical decoupling fit parameters for a single qubit"""

    sequence: str
    """Name of the DD sequence (e.g. "CPMG", "XY4")."""
    pulses_per_window: int
    """Selected number of pi pulses per window."""
    tau_opt: float
    """Pulse half-spacing tau at the selected N, in s (pulses are separated by 2 * tau + t_pi, up to 4 ns rounding)."""
    T2: float
    """T2 under the DD sequence at the selected N, in s."""
    T2_error: float
    alpha: float
    """Stretch exponent of the decay exp(-(t / T2)^alpha) at the selected N."""
    alpha_error: float
    num_rounds: int
    """Number of consecutive windows (rounds) M over which the error per round is averaged."""
    error_per_round: float
    """Average dephasing (phase-flip) error per round over M rounds at the selected N, (1 - C(M)^(1/M)) / 2, from the
    measured coherence C(M) after M windows (from the fit if that point was not measured or C(M) < 0.2)."""
    error_per_round_error: float
    error_per_round_fit: float
    """Same quantity from the fitted decay, (1 - exp(-(M window / T2)^alpha / M)) / 2, as a cross-check."""
    extra_error_per_round: float
    """Error per round above the best N (0 if the selected N is the best one)."""
    window: float
    """Idle window duration, in s."""
    at_boundary: bool
    """True if the selection lies at the edge of the scanned N values."""
    fit_check_ok: bool
    """False if the measured and fitted errors per round disagree by more than 3 sigma."""
    floor_shifted_pulses_per_window: List[int]
    """Numbers of pulses whose decay settles at a different level than the curve with the fewest pulses (non-dephasing
    process such as leakage or heating suspected). They are excluded from the decision."""
    noise_amplitude: float
    """Noise spectrum fit S_f(f) = A * (1 MHz / f)^beta + C: A, one-sided PSD at 1 MHz in Hz (NaN if not fitted)."""
    noise_amplitude_error: float
    noise_exponent: float
    """Noise spectrum fit: beta (1 = 1/f noise, 0 = white)."""
    noise_exponent_error: float
    noise_floor: float
    """Noise spectrum fit: white floor C in Hz."""
    noise_floor_error: float
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Logs the node-specific fitted results for all qubits from the fit results."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, r in fit_results.items():
        s = f"Results for qubit {q} ({r['sequence']}): "
        if not r["success"]:
            log_callable(s + "FAIL!")
            continue
        s += (
            f"optimal {r['pulses_per_window']} pulses per {1e6 * r['window']:.2f} us window "
            f"(tau = {1e9 * r['tau_opt']:.0f} ns) | "
            f"error per round (avg over {r['num_rounds']}) = {1e2 * r['error_per_round']:.3f} "
            f"+/- {1e2 * r['error_per_round_error']:.3f} % (+{1e2 * r['extra_error_per_round']:.3f} % vs best) | "
            f"T2 = {1e6 * r['T2']:.1f} +/- {1e6 * r['T2_error']:.1f} us, alpha = {r['alpha']:.2f}"
        )
        if r["at_boundary"]:
            s += " | WARNING: optimum at the edge of the scanned N range"
        if np.isfinite(r["noise_amplitude"]):
            s += (
                f" | noise: S_f = {r['noise_amplitude']:.3g} Hz (1 MHz / f)^{r['noise_exponent']:.2f}"
                f" + {r['noise_floor']:.3g} Hz"
            )
        if r["floor_shifted_pulses_per_window"]:
            s += f" | WARNING: decay level shifted (leakage/heating?) for N = {r['floor_shifted_pulses_per_window']}"
        if not r["fit_check_ok"]:
            s += " | WARNING: measured and fitted errors per round disagree"
        log_callable(s + " SUCCESS!")


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    # Datasets from the former 06c_cpmg node used "n_window" for the pulses-per-window axis
    if "n_window" in ds.dims:
        ds = ds.rename(n_window="pulses_per_window")
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the DD decay curves and select the number of pulses per window for each qubit.

    For each qubit, all decay curves (one per number of pulses per window) are fitted jointly to
    a_N * exp(-(t / T2_N)^alpha_N) + offset_N, each with its own dephased level offset_N.
    """
    signal = ds.state if node.parameters.use_state_discrimination else ds.I
    return fit_dd_data(
        signal,
        T1={q.name: q.T1 for q in node.namespace["qubits"]},
        fit_noise_spectrum=node.parameters.visualization.show_noise_spectrum,
        num_rounds=node.parameters.decision.num_rounds,
        max_extra_error_per_round=node.parameters.decision.max_extra_error_per_round,
        uncertainty_margin_sigma=node.parameters.decision.uncertainty_margin_sigma,
    )


def fit_dd_data(
    signal: xr.DataArray,
    T1: Optional[Dict[str, Optional[float]]] = None,
    fit_noise_spectrum: bool = True,
    num_rounds: int = 10,
    max_extra_error_per_round: float = 1e-3,
    uncertainty_margin_sigma: float = 1.0,
) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """Fit a signal with dims (qubit, pulses_per_window, point).

    Required coords: time (per point, or per pulses_per_window and point), tau and pulse_spacing (per qubit and
    pulses_per_window), all in ns, and the scalar window_ns. The "windows" coordinate (per point) locates the point
    measured after num_rounds windows, from which the error per round is taken directly; without it, the fitted value
    is used. T1 (in s, per qubit) is subtracted from the dephasing rate for the noise spectrum; qubits without T1 use
    the raw T2.
    """
    signal = signal.transpose("qubit", "pulses_per_window", "point")
    qubits = signal.qubit.values
    n_values = signal.pulses_per_window.values
    window_ns = float(signal.window_ns)
    shape = (len(qubits), len(n_values))
    k_rounds = None
    if "windows" in signal.coords and np.any(signal.windows.values == num_rounds):
        k_rounds = int(np.argmax(signal.windows.values == num_rounds))

    offset, offset_err, floor_shift, floor_shift_err = (np.full(shape, np.nan) for _ in range(4))
    floor_flag = np.zeros(shape, dtype=bool)
    amplitude, T2, T2_err, alpha, alpha_err, p_fit, p_fit_err, p_meas, p_meas_err = (
        np.full(shape, np.nan) for _ in range(9)
    )
    fit_curve = np.full(signal.shape, np.nan)
    coherence = np.full(signal.shape, np.nan)

    for iq, q in enumerate(qubits):
        sig = signal.sel(qubit=q)
        y = sig.values
        t = np.broadcast_to(sig.time.transpose(..., "point").values, y.shape)
        res = _joint_stretched_exp_fit(t, y, window_ns, num_rounds)
        if res is None:
            continue
        floor_flag[iq] = _floor_shifted(res)
        offset[iq], offset_err[iq] = res["offset"], res["offset_error"]
        floor_shift[iq], floor_shift_err[iq] = res["floor_shift"], res["floor_shift_error"]
        amplitude[iq], T2[iq], T2_err[iq] = res["amplitude"], res["T2"], res["T2_error"]
        alpha[iq], alpha_err[iq] = res["alpha"], res["alpha_error"]
        p_fit[iq], p_fit_err[iq] = res["p_round"], res["p_round_error"]
        for i_n in range(len(n_values)):
            a = amplitude[iq, i_n]
            fit_curve[iq, i_n] = a * np.exp(-((t[i_n] / T2[iq, i_n]) ** alpha[iq, i_n])) + offset[iq, i_n]
            coherence[iq, i_n] = (y[i_n] - offset[iq, i_n]) / a
            if k_rounds is not None:
                c = coherence[iq, i_n, k_rounds]
                if c >= MIN_COHERENCE_FOR_MEASURED_ERROR:
                    p_meas[iq, i_n] = (1 - c ** (1 / num_rounds)) / 2
                    p_meas_err[iq, i_n] = c ** (1 / num_rounds - 1) / (2 * num_rounds) * res["residual_std"] / abs(a)

    dims2 = ["qubit", "pulses_per_window"]
    ds_fit = xr.Dataset(
        {
            "offset": (dims2, offset, {"long_name": "dephased level of each decay"}),
            "offset_error": (dims2, offset_err),
            "floor_shift": (dims2, floor_shift, {"long_name": "dephased level relative to the fewest-pulse curve"}),
            "floor_shift_error": (dims2, floor_shift_err),
            "amplitude": (dims2, amplitude),
            "T2": (dims2, T2, {"long_name": "T2 under DD", "units": "ns"}),
            "T2_error": (dims2, T2_err, {"long_name": "T2 error", "units": "ns"}),
            "alpha": (dims2, alpha, {"long_name": "stretch exponent"}),
            "alpha_error": (dims2, alpha_err, {"long_name": "stretch exponent error"}),
            "error_per_round_measured": (dims2, p_meas, {"long_name": "measured average error per round"}),
            "error_per_round_measured_error": (dims2, p_meas_err),
            "error_per_round_fit": (dims2, p_fit, {"long_name": "fitted average error per round"}),
            "error_per_round_fit_error": (dims2, p_fit_err),
            "fit": (signal.dims, fit_curve),
            "coherence": (signal.dims, coherence, {"long_name": "normalized coherence"}),
        },
        coords=signal.coords,
    )
    ds_fit["signal"] = signal
    ds_fit["num_rounds"] = num_rounds
    ds_fit["floor_shifted"] = (dims2, floor_flag, {"long_name": "decay level shifted (leakage / heating suspected)"})
    measured = np.isfinite(ds_fit.error_per_round_measured)
    ds_fit["error_per_round"] = ds_fit.error_per_round_measured.where(measured, ds_fit.error_per_round_fit)
    ds_fit["error_per_round_error"] = ds_fit.error_per_round_measured_error.where(
        measured, ds_fit.error_per_round_fit_error
    )
    ds_fit["error_per_round_is_measured"] = measured
    ds_fit = _add_noise_spectrum(ds_fit, T1 or {}, fit_noise_spectrum)

    ds_fit, fit_results = _select_pulses_per_window(ds_fit, max_extra_error_per_round, uncertainty_margin_sigma)
    return ds_fit, fit_results


def _add_noise_spectrum(ds_fit: xr.Dataset, T1: Dict[str, Optional[float]], fit: bool = True) -> xr.Dataset:
    """First-order (Cywinski et al.) dephasing noise spectrum from T2 under DD.

    For N evenly spaced pi pulses the filter function peaks at f0 = 1 / (2 * pulse spacing) = N / (2 * window), and
    keeping only this fundamental gives chi(t) = (4 / pi^2) S_omega(omega0) t (two-sided, rad^2/s). With chi = t / T_phi
    and S_f = S_omega / (4 pi^2) in Hz^2/Hz, the one-sided frequency-noise PSD is S_f(f0) = 1 / (8 T_phi), with the pure
    dephasing rate 1/T_phi = 1/T2 - 1/(2 T1). Strictly valid for exponential decay (alpha = 1).
    If fit is True, the reliable points are fitted to S_f(f) = A * (1 MHz / f)^beta + C.
    """
    T1_ns = xr.DataArray(
        [1e9 * T1[q] if T1.get(q) else np.inf for q in ds_fit.qubit.values], coords={"qubit": ds_fit.qubit.values}
    )
    gamma_phi = 1e9 * (1 / ds_fit.T2 - 1 / (2 * T1_ns))  # 1/s
    gamma_phi_err = 1e9 * ds_fit.T2_error / ds_fit.T2**2
    positive = gamma_phi > 0
    ds_fit["noise_frequency"] = 1e9 / (2 * ds_fit.pulse_spacing)
    ds_fit["noise_frequency"].attrs = {"long_name": "DD filter frequency f0", "units": "Hz"}
    ds_fit["noise_psd"] = (gamma_phi / 8).where(positive)
    ds_fit["noise_psd"].attrs = {"long_name": "one-sided frequency-noise PSD S_f(f0)", "units": "Hz^2/Hz"}
    ds_fit["noise_psd_error"] = (gamma_phi_err / 8).where(positive)
    ds_fit["T1_subtracted"] = np.isfinite(T1_ns)

    names = ["amplitude", "exponent", "floor"]
    values = {k: np.full(ds_fit.sizes["qubit"], np.nan) for k in names + [f"{k}_error" for k in names]}
    if fit:
        for iq, q in enumerate(ds_fit.qubit.values):
            f = ds_fit.sel(qubit=q)
            res = fit_noise_power_law(
                f.noise_frequency.values, f.noise_psd.values, f.noise_psd_error.values, f.alpha.values
            )
            if res is not None:
                for k, v in res.items():
                    values[k][iq] = v
    for k, v in values.items():
        ds_fit[f"noise_fit_{k}"] = (("qubit",), v)
    return ds_fit


def fit_noise_power_law(f_hz, psd, psd_err, alpha=None, min_points: int = 4):
    """Fit S_f(f) = A * (F_REF / f)^beta + C in log space, weighted by the relative errors.

    Only points with positive PSD, finite relative error below 100% and alpha <= ALPHA_RELIABLE_MAX are used.
    Returns a dict with amplitude (A, Hz at F_REF), exponent (beta), floor (C, Hz) and their errors, or None if there
    are fewer than min_points usable points, the fit fails, or the power law is not constrained by the data (beta at
    its bounds or A not resolved from zero).
    """
    f_hz, psd, psd_err = (np.asarray(x, dtype=float) for x in (f_hz, psd, psd_err))
    use = np.isfinite(f_hz) & np.isfinite(psd) & (psd > 0) & np.isfinite(psd_err) & (psd_err > 0) & (psd_err < psd)
    if alpha is not None:
        use &= np.asarray(alpha) <= ALPHA_RELIABLE_MAX
    if np.count_nonzero(use) < min_points:
        return None
    x, y, sigma_log = NOISE_FIT_F_REF_HZ / f_hz[use], psd[use], psd_err[use] / psd[use]
    scale = float(np.max(y))

    def model(p):
        return p[0] * x ** p[1] + p[2]

    def residuals(p):
        return (np.log(model(p)) - np.log(y / scale)) / sigma_log

    p0 = [float(np.median(y / scale)), 1.0, float(np.min(y / scale)) / 2]
    lower, upper = [1e-9, 0.0, 0.0], [np.inf, 3.0, np.inf]
    try:
        fit = least_squares(residuals, p0, bounds=(lower, upper))
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not fit.success:
        return None
    dof = max(len(y) - 3, 1)
    try:
        cov = np.linalg.pinv(fit.jac.T @ fit.jac) * max(2 * fit.cost / dof, 1.0)
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        err = np.full(3, np.nan)
    A, beta, C = fit.x
    if not (lower[1] + 1e-3 < beta < upper[1] - 1e-3) or not (err[0] < A):
        return None
    return {
        "amplitude": A * scale,
        "amplitude_error": err[0] * scale,
        "exponent": beta,
        "exponent_error": err[1],
        "floor": C * scale,
        "floor_error": err[2] * scale,
    }


def _floor_shifted(res: dict) -> np.ndarray:
    """Curves whose offset differs from the fewest-pulse curve by > FLOOR_SHIFT_SIGMA sigma and > a fraction of their
    amplitude."""
    shift, err = res["floor_shift"], res["floor_shift_error"]
    significant = np.abs(shift) > FLOOR_SHIFT_SIGMA * err
    return significant & (np.abs(shift) > FLOOR_SHIFT_MIN_FRACTION * np.abs(res["amplitude"]))


def _joint_stretched_exp_fit(t: np.ndarray, y: np.ndarray, window_ns: float, num_rounds: int = 1):
    """Fit y[n, :] = a_n * exp(-(t[n, :] / T2_n)^alpha_n) + offset_n for all curves (each with its own dephased level).

    Returns a dict with per-curve offset, its shift relative to the first (fewest pulses) curve, amplitude, T2, alpha
    (with errors), the average error per round over M windows p_n = (1 - exp(-(M window / T2_n)^alpha_n / M)) / 2 with
    its error (propagated with the T2-alpha correlation) and the residual standard deviation, or None if the fit fails.
    """
    n = y.shape[0]
    valid = np.isfinite(y) & np.isfinite(t)
    if not np.any(valid):
        return None
    off0 = np.array([np.mean(y[i, valid[i]][-3:]) for i in range(n)])
    a0 = y[:, 0] - off0
    k = y.shape[1] // 3
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma0 = -np.log(np.clip((y[:, k] - off0) / a0, 0.05, 0.95)) / (t[:, k] - t[:, 0])
    gamma0 = np.where(np.isfinite(gamma0) & (gamma0 > 0), gamma0, 1 / np.nanmax(t))
    # Rates scaled by the longest time so that all parameters are of order 1
    t_scale = float(np.nanmax(t))
    p0 = np.concatenate([off0, a0, gamma0 * t_scale, np.full(n, 1.2)])
    lower = np.concatenate([np.full(2 * n, -np.inf), np.zeros(n), np.full(n, ALPHA_BOUNDS[0])])
    upper = np.concatenate([np.full(3 * n, np.inf), np.full(n, ALPHA_BOUNDS[1])])
    io, ia, ig, ial = (slice(k_ * n, (k_ + 1) * n) for k_ in range(4))

    def residuals(p):
        a, g, al = p[ia, None], p[ig, None], p[ial, None]
        return (a * np.exp(-((g * t / t_scale) ** al)) + p[io, None] - y)[valid]

    try:
        fit = least_squares(residuals, p0, bounds=(lower, upper))
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not fit.success:
        return None

    p = fit.x
    dof = max(np.count_nonzero(valid) - len(p), 1)
    residual_var = 2 * fit.cost / dof
    try:
        cov = np.linalg.pinv(fit.jac.T @ fit.jac) * residual_var
    except np.linalg.LinAlgError:
        cov = np.full((len(p), len(p)), np.nan)
    var = np.clip(np.diag(cov), 0, None)

    offsets = p[io]
    shift = offsets - offsets[0]
    shift_var = np.array([var[i] + var[0] - 2 * cov[i, 0] for i in range(n)])

    g, al = p[ig], p[ial]
    g_var, al_var = var[ig], var[ial]
    g_al_cov = np.array([cov[2 * n + i, 3 * n + i] for i in range(n)])
    with np.errstate(divide="ignore", invalid="ignore"):
        T2 = np.where(g > 0, t_scale / g, np.nan)
        T2_err = T2 * np.sqrt(g_var) / g
        # p = (1 - exp(-x / M)) / 2 with x = (M window / T2)^alpha, error propagated with the T2-alpha correlation
        u = g * num_rounds * window_ns / t_scale
        x = u**al
        p_round = (1 - np.exp(-x / num_rounds)) / 2
        dp_dx = np.exp(-x / num_rounds) / (2 * num_rounds)
        dx_dg = al * x / g
        dx_dal = x * np.log(u)
        p_round_err = dp_dx * np.sqrt(
            np.clip(dx_dg**2 * g_var + dx_dal**2 * al_var + 2 * dx_dg * dx_dal * g_al_cov, 0, None)
        )
    return {
        "offset": offsets,
        "offset_error": np.sqrt(var[io]),
        "floor_shift": shift,
        "floor_shift_error": np.sqrt(np.clip(shift_var, 0, None)),
        "amplitude": p[ia],
        "T2": T2,
        "T2_error": T2_err,
        "alpha": al,
        "alpha_error": np.sqrt(al_var),
        "p_round": p_round,
        "p_round_error": p_round_err,
        "residual_std": float(np.sqrt(residual_var)),
    }


def _select_pulses_per_window(ds_fit: xr.Dataset, max_extra_error: float, margin_sigma: float):
    """Select the number of pulses per window and populate the FitParameters.

    The best N has the lowest average error per round p_best. The selection is the fewest pulses whose extra error
    p_N - p_best is within max_extra_error, allowing margin_sigma standard errors of measurement uncertainty.
    """
    valid = (
        np.isfinite(ds_fit.T2)
        & np.isfinite(ds_fit.T2_error)
        & (ds_fit.T2_error < ds_fit.T2)
        & np.isfinite(ds_fit.error_per_round)
        & (ds_fit.error_per_round_error <= MAX_RELATIVE_ERROR_FOR_SELECTION * abs(ds_fit.error_per_round))
        & ~ds_fit.floor_shifted
    )
    ds_fit["valid"] = valid

    window_ns = float(ds_fit.window_ns)
    num_rounds = int(ds_fit.num_rounds)
    # Datasets from the former 06c_cpmg node have no sequence coordinate
    sequence = str(ds_fit.sequence.values) if "sequence" in ds_fit.coords else "CPMG"
    n_values = ds_fit.pulses_per_window.values
    success, n_sel, n_best, p_best_list = [], [], [], []
    extra = np.full((ds_fit.sizes["qubit"], len(n_values)), np.nan)
    fit_results = {}
    for iq, q in enumerate(ds_fit.qubit.values):
        f = ds_fit.sel(qubit=q)
        p = f.error_per_round.where(f.valid).values
        p_err = f.error_per_round_error.values
        ok = bool(np.any(np.isfinite(p)))
        i_best, i_sel = 0, 0
        if ok:
            i_best = int(np.nanargmin(p))
            extra[iq] = p - p[i_best]
            margin = margin_sigma * np.sqrt(p_err**2 + p_err[i_best] ** 2)
            margin[i_best] = 0.0
            accepted = np.isfinite(p) & (extra[iq] <= max_extra_error + margin)
            i_sel = int(np.argmax(accepted))
        success.append(ok)
        n_sel.append(int(n_values[i_sel]))
        n_best.append(int(n_values[i_best]))
        p_best_list.append(float(p[i_best]) if ok else np.nan)
        sel = f.isel(pulses_per_window=i_sel)
        p_meas, p_meas_err = float(sel.error_per_round_measured), float(sel.error_per_round_measured_error)
        p_fit, p_fit_err = float(sel.error_per_round_fit), float(sel.error_per_round_fit_error)
        fit_check_ok = not (np.isfinite(p_meas) and abs(p_meas - p_fit) > 3 * np.hypot(p_meas_err, p_fit_err))
        fit_results[q] = FitParameters(
            sequence=sequence,
            pulses_per_window=int(n_values[i_sel]),
            tau_opt=1e-9 * float(sel.tau),
            T2=1e-9 * float(sel.T2),
            T2_error=1e-9 * float(sel.T2_error),
            alpha=float(sel.alpha),
            alpha_error=float(sel.alpha_error),
            num_rounds=num_rounds,
            error_per_round=float(sel.error_per_round),
            error_per_round_error=float(sel.error_per_round_error),
            error_per_round_fit=p_fit,
            extra_error_per_round=float(extra[iq, i_sel]) if ok else np.nan,
            window=1e-9 * window_ns,
            at_boundary=ok and len(n_values) > 1 and (i_best == len(n_values) - 1 or i_sel == 0),
            fit_check_ok=fit_check_ok,
            floor_shifted_pulses_per_window=[int(x) for x in n_values[f.floor_shifted.values]],
            noise_amplitude=float(f.noise_fit_amplitude),
            noise_amplitude_error=float(f.noise_fit_amplitude_error),
            noise_exponent=float(f.noise_fit_exponent),
            noise_exponent_error=float(f.noise_fit_exponent_error),
            noise_floor=float(f.noise_fit_floor),
            noise_floor_error=float(f.noise_fit_floor_error),
            success=ok,
        )
    coords = {"qubit": ds_fit.qubit.values}
    ds_fit["extra_error_per_round"] = (("qubit", "pulses_per_window"), extra)
    ds_fit["pulses_per_window_selected"] = xr.DataArray(n_sel, coords=coords)
    ds_fit["pulses_per_window_best"] = xr.DataArray(n_best, coords=coords)
    ds_fit["error_per_round_best"] = xr.DataArray(p_best_list, coords=coords)
    ds_fit["max_extra_error_per_round"] = max_extra_error
    ds_fit["success"] = xr.DataArray(success, coords=coords)
    return ds_fit, fit_results
