"""Fixed-amplitude geometric CZ cphase duration analysis.

Extracts Ramsey I/Q phases from a duration-only exchange sweep (X90 vs Y90
analysis) and finds the exchange duration where the conditional phase
``phi1 - phi0 - pi`` crosses pi.  The common-mode Stark shift from the
barrier gate cancels in the subtraction; the ``-pi`` removes the known
parity-readout offset between branches.

If the dataset only has the legacy ``experiment_type`` axis (no
``analysis_axis``), falls back to fitting damped sinusoids to each branch
(deprecated).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


@dataclass
class DampedSinusoidFit:
    """Fitted parameters for a single damped sinusoid trace."""

    frequency: float = 0.0
    amplitude: float = 0.0
    phase: float = 0.0
    offset: float = 0.0
    T2: float = np.inf
    success: bool = False


def _damped_sinusoid(
    t: np.ndarray,
    frequency: float,
    amplitude: float,
    phase: float,
    offset: float,
    decay: float,
) -> np.ndarray:
    return (
        amplitude * np.cos(2 * np.pi * frequency * t + phase) * np.exp(-t / decay)
        + offset
    )


def _fit_damped_sinusoid(times_ns: np.ndarray, signal: np.ndarray) -> DampedSinusoidFit:
    """Fit a damped sinusoid to a 1-D time trace."""
    result = DampedSinusoidFit()
    if len(times_ns) < 6:
        return result

    try:
        signal_centered = signal - np.nanmean(signal)
        fft_vals = np.fft.rfft(signal_centered)
        fft_freqs = np.fft.rfftfreq(len(times_ns), d=(times_ns[1] - times_ns[0]))
        fft_magnitudes = np.abs(fft_vals[1:])
        peak_idx = int(np.argmax(fft_magnitudes)) + 1
        freq_guess = float(fft_freqs[peak_idx])
        phase_guess = float(np.angle(fft_vals[peak_idx]))

        amp_guess = float(np.nanmax(signal) - np.nanmin(signal)) / 2.0
        offset_guess = float(np.nanmean(signal))
        decay_guess = float(times_ns[-1] - times_ns[0])

        freq_guess = max(freq_guess, 1e-6 / (times_ns[-1] - times_ns[0]))

        popt, _ = curve_fit(
            _damped_sinusoid,
            times_ns,
            signal,
            p0=[freq_guess, amp_guess, phase_guess, offset_guess, decay_guess],
            bounds=(
                [0, 0, -2 * np.pi, -np.inf, 1.0],
                [np.inf, np.inf, 2 * np.pi, np.inf, np.inf],
            ),
            maxfev=10000,
        )
        result.frequency = float(popt[0])
        result.amplitude = float(popt[1])
        result.phase = float(popt[2])
        result.offset = float(popt[3])
        result.T2 = float(popt[4])
        result.success = bool(np.isfinite(popt).all() and result.frequency > 0)
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        pass

    return result


@dataclass
class GeometricCZDurationFitResult:
    """Aggregated fixed-amplitude cphase fit result for one qubit pair."""

    exchange_amplitude: float = 0.0
    """Fixed barrier gate voltage used for the exchange pulse (V)."""
    optimal_duration: float = 0.0
    """Exchange pulse duration where summed phase reaches pi (ns)."""
    phase_ctrl_ground_frequency: float = 0.0
    """Fitted phase oscillation frequency with control in |0> (cycles/ns)."""
    phase_ctrl_excited_frequency: float = 0.0
    """Fitted phase oscillation frequency with control in |1> (cycles/ns)."""
    phase_ctrl_ground_phase: float = 0.0
    """Fitted phase offset with control in |0> (rad)."""
    phase_ctrl_excited_phase: float = 0.0
    """Fitted phase offset with control in |1> (rad)."""
    phase_sum_at_optimal: float = 0.0
    """Summed fitted phase at the selected duration, modulo 2pi (rad)."""
    conditional_phase_rate: float = 0.0
    """Mean conditional phase accumulation rate (rad/ns)."""
    success: bool = False


def _select_experiment_trace(
    data_array: xr.DataArray,
    experiment_type: int,
) -> np.ndarray:
    """Return the requested experiment trace as float64."""
    if "experiment_type" in data_array.coords:
        return data_array.sel(experiment_type=experiment_type).values.astype(np.float64)
    return data_array.values.astype(np.float64)[experiment_type]


def _phase_sum_pi_time(
    durations_ns: np.ndarray,
    frequency_sum: float,
    phase_sum: float,
) -> tuple[float, bool]:
    """Find the first root within the swept duration range.

    Solves ``2*pi*frequency_sum*t + phase_sum = pi + 2*pi*k`` and chooses the
    smallest solution not below the first swept duration.
    """
    if frequency_sum <= 0 or not np.isfinite(frequency_sum + phase_sum):
        return np.nan, False

    t_min = float(np.nanmin(durations_ns))
    t_max = float(np.nanmax(durations_ns))
    base = (np.pi - phase_sum) / (2 * np.pi * frequency_sum)
    period = 1.0 / frequency_sum

    k = np.ceil((t_min - base) / period)
    t_pi = base + k * period
    if t_pi < t_min:
        t_pi += period

    step = (
        float(np.nanmedian(np.diff(np.sort(durations_ns))))
        if len(durations_ns) > 1
        else 0.0
    )
    in_sweep = t_pi <= t_max + 0.5 * step
    return float(t_pi), bool(in_sweep)


def _phase_mod_2pi(phase_rad: float | np.ndarray) -> float | np.ndarray:
    """Wrap a phase to [0, 2*pi)."""
    return np.mod(phase_rad, 2 * np.pi)


def _first_duration_linear_crossing(
    durations_ns: np.ndarray,
    y: np.ndarray,
    target: float,
) -> tuple[float, bool]:
    """First crossing of ``target`` along increasing duration via linear interpolation."""
    d = np.asarray(durations_ns, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if d.size < 2 or y.size != d.size:
        return np.nan, False

    t_min = float(np.nanmin(d))
    t_max = float(np.nanmax(d))
    step = (
        float(np.nanmedian(np.diff(np.sort(d))))
        if len(d) > 1
        else 0.0
    )

    for i in range(len(d) - 1):
        y0, y1 = y[i], y[i + 1]
        t0, t1 = d[i], d[i + 1]
        if not np.isfinite(y0) or not np.isfinite(y1):
            continue
        if y0 == y1:
            if y0 == target:
                return float(t0), bool(t_min <= t0 <= t_max + 0.5 * step)
            continue
        crosses = (y0 <= target <= y1) or (y1 <= target <= y0)
        if not crosses:
            continue
        frac = (target - y0) / (y1 - y0)
        t_star = t0 + frac * (t1 - t0)
        in_sweep = t_min <= t_star <= t_max + 0.5 * step
        return float(t_star), bool(in_sweep)

    return np.nan, False


def fit_raw_data(
    ds_raw: xr.Dataset,
    qubit_pairs: list[Any],
    *,
    exchange_amplitude: float,
    analysis_signal: str = "E_p2_given_p1_0",
    quadrature_signal_center: float = 0.5,
) -> tuple[xr.Dataset, dict[str, dict[str, Any]]]:
    """Run fixed-amplitude geometric cphase analysis for every qubit pair.

    Parameters
    ----------
    ds_raw : xr.Dataset
        Dataset after ``process_parity_streams`` with coordinates
        ``(control_state, analysis_axis, exchange_duration)`` for quadrature
        readout, or legacy ``(experiment_type, exchange_duration)``.
    qubit_pairs : list
        Qubit pair objects (each must have a ``.name`` attribute).
    exchange_amplitude : float
        Fixed exchange amplitude used during acquisition.
    analysis_signal : str
        Which conditional expectation to analyse.
    quadrature_signal_center : float
        Signal value for the center of the Ramsey I/Q circle (default 0.5).
    """
    durations = ds_raw.coords["exchange_duration"].values.astype(np.float64)
    fit_results: Dict[str, dict[str, Any]] = {}
    fit_vars: dict[str, Any] = {}

    for qp in qubit_pairs:
        var_name = f"{analysis_signal}_{qp.name}"
        if var_name not in ds_raw.data_vars:
            logger.warning("No %s variable for pair %s; skipping.", var_name, qp.name)
            fit_results[qp.name] = asdict(GeometricCZDurationFitResult())
            continue

        data_array = ds_raw[var_name]
        if "analysis_axis" in data_array.dims:
            fit_vars.update(
                _fit_quadrature_branch(
                    qp.name,
                    data_array,
                    durations,
                    float(exchange_amplitude),
                    float(quadrature_signal_center),
                )
            )
            fit_results[qp.name] = fit_vars.pop(f"_result_dict_{qp.name}")
            continue

        signal_ctrl_ground = _select_experiment_trace(data_array, 0)
        signal_ctrl_excited = _select_experiment_trace(data_array, 1)

        fit_ground = _fit_damped_sinusoid(durations, signal_ctrl_ground)
        fit_excited = _fit_damped_sinusoid(durations, signal_ctrl_excited)

        frequency_sum = fit_ground.frequency + fit_excited.frequency
        phase_sum = fit_ground.phase + fit_excited.phase
        t_pi, root_in_sweep = _phase_sum_pi_time(
            durations,
            frequency_sum,
            phase_sum,
        )

        phase_sum_at_optimal = np.nan
        if np.isfinite(t_pi):
            phase_sum_at_optimal = _phase_mod_2pi(
                2 * np.pi * frequency_sum * t_pi + phase_sum
            )

        success = bool(fit_ground.success and fit_excited.success and root_in_sweep)

        result = GeometricCZDurationFitResult(
            exchange_amplitude=float(exchange_amplitude),
            optimal_duration=float(t_pi) if np.isfinite(t_pi) else 0.0,
            phase_ctrl_ground_frequency=float(fit_ground.frequency),
            phase_ctrl_excited_frequency=float(fit_excited.frequency),
            phase_ctrl_ground_phase=float(fit_ground.phase),
            phase_ctrl_excited_phase=float(fit_excited.phase),
            phase_sum_at_optimal=(
                float(phase_sum_at_optimal)
                if np.isfinite(phase_sum_at_optimal)
                else 0.0
            ),
            conditional_phase_rate=0.0,
            success=success,
        )
        fit_results[qp.name] = asdict(result)

        if fit_ground.success:
            fit_vars[f"fitted_phase_ctrl_ground_{qp.name}"] = xr.DataArray(
                _damped_sinusoid(
                    durations,
                    fit_ground.frequency,
                    fit_ground.amplitude,
                    fit_ground.phase,
                    fit_ground.offset,
                    fit_ground.T2,
                ),
                dims=["exchange_duration"],
                attrs={"long_name": "fit phase signal (control ground)", "units": ""},
            )
        if fit_excited.success:
            fit_vars[f"fitted_phase_ctrl_excited_{qp.name}"] = xr.DataArray(
                _damped_sinusoid(
                    durations,
                    fit_excited.frequency,
                    fit_excited.amplitude,
                    fit_excited.phase,
                    fit_excited.offset,
                    fit_excited.T2,
                ),
                dims=["exchange_duration"],
                attrs={"long_name": "fit phase signal (control excited)", "units": ""},
            )

        fit_vars[f"phase_sum_{qp.name}"] = xr.DataArray(
            2 * np.pi * frequency_sum * durations + phase_sum,
            dims=["exchange_duration"],
            attrs={"long_name": "summed fitted phase (unwrapped)", "units": "rad"},
        )
        fit_vars[f"t_pi_cphase_{qp.name}"] = xr.DataArray(
            float(t_pi) if np.isfinite(t_pi) else np.nan,
            attrs={"long_name": "duration where summed phase equals pi", "units": "ns"},
        )

    ds_fit = xr.Dataset(
        fit_vars,
        coords={"exchange_duration": durations},
        attrs={"exchange_amplitude": float(exchange_amplitude)},
    )
    return ds_fit, fit_results


def _linear_fit_crossing(
    durations_ns: np.ndarray,
    y: np.ndarray,
    target: float,
    trim_fraction: float = 0.15,
) -> tuple[float, bool]:
    """Fit a line to the phase trace and find where it crosses ``target``.

    Used as a fallback when the raw point-to-point crossing detection fails
    due to noise (e.g. reduced visibility from SPAM errors).  The conditional
    phase is approximately linear in duration since J is constant during the
    exchange pulse flat-top.

    Parameters
    ----------
    durations_ns : array
        Duration sweep values in ns.
    y : array
        Unwrapped conditional phase trace.
    target : float
        Phase value to cross (typically π).
    trim_fraction : float
        Fraction of early points to exclude (turn-on transient region).
    """
    d = np.asarray(durations_ns, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if d.size < 4:
        return np.nan, False

    n_trim = max(1, int(len(d) * trim_fraction))
    d_fit = d[n_trim:]
    y_fit = y[n_trim:]

    valid = np.isfinite(d_fit) & np.isfinite(y_fit)
    if np.sum(valid) < 3:
        return np.nan, False

    coeffs = np.polyfit(d_fit[valid], y_fit[valid], 1)
    slope, intercept = coeffs

    if slope <= 0:
        return np.nan, False

    t_crossing = (target - intercept) / slope

    t_min = float(np.nanmin(d))
    t_max = float(np.nanmax(d))
    step = float(np.nanmedian(np.diff(np.sort(d)))) if len(d) > 1 else 0.0
    in_sweep = t_min <= t_crossing <= t_max + 0.5 * step

    return float(t_crossing), bool(in_sweep)


def _fit_quadrature_branch(
    name: str,
    data: xr.DataArray,
    durations: np.ndarray,
    exchange_amplitude: float,
    center: float,
) -> dict[str, Any]:
    """Quadrature phases and conditional-phase pi crossing for one pair."""
    fit_vars: dict[str, Any] = {}

    i0 = data.sel(control_state=0, analysis_axis=0).values.astype(np.float64)
    q0 = data.sel(control_state=0, analysis_axis=1).values.astype(np.float64)
    i1 = data.sel(control_state=1, analysis_axis=0).values.astype(np.float64)
    q1 = data.sel(control_state=1, analysis_axis=1).values.astype(np.float64)

    phi0 = np.arctan2(q0 - center, i0 - center)
    phi1 = np.arctan2(q1 - center, i1 - center)
    phi0_unwrapped = np.unwrap(phi0)
    phi1_unwrapped = np.unwrap(phi1)

    # Conditional phase via complex conjugate product.
    # angle(z1·z0*) = φ₁−φ₀ ≈ π (parity offset) at zero exchange.
    # Negating shifts the branch cut away from the initial value:
    # angle(−z1·z0*) = φ₁−φ₀−π, starting near 0 → safe for unwrap.
    # The common-mode Stark shift cancels exactly in the product.
    z0 = (i0 - center) + 1j * (q0 - center)
    z1 = (i1 - center) + 1j * (q1 - center)
    delta_phi_unwrapped = np.unwrap(np.angle(-z1 * np.conj(z0)))

    if len(durations) > 1:
        conditional_phase_rate = float(
            np.nanmean(np.gradient(delta_phi_unwrapped, durations))
        )
    else:
        conditional_phase_rate = 0.0

    t_pi, root_in_sweep = _first_duration_linear_crossing(
        durations,
        delta_phi_unwrapped,
        np.pi,
    )

    if not root_in_sweep or not np.isfinite(t_pi):
        t_pi, root_in_sweep = _linear_fit_crossing(
            durations, delta_phi_unwrapped, np.pi
        )

    success = bool(root_in_sweep)

    fit_vars[f"conditional_phase_{name}"] = xr.DataArray(
        delta_phi_unwrapped,
        dims=["exchange_duration"],
        attrs={
            "long_name": "conditional phase φ₁−φ₀−π",
            "units": "rad",
        },
    )
    fit_vars[f"phi_ctrl_ground_{name}"] = xr.DataArray(
        phi0_unwrapped,
        dims=["exchange_duration"],
        attrs={"long_name": "unwrapped phase (control |0>)", "units": "rad"},
    )
    fit_vars[f"phi_ctrl_excited_{name}"] = xr.DataArray(
        phi1_unwrapped,
        dims=["exchange_duration"],
        attrs={"long_name": "unwrapped phase (control |1>)", "units": "rad"},
    )
    fit_vars[f"t_pi_cphase_{name}"] = xr.DataArray(
        float(t_pi) if np.isfinite(t_pi) else np.nan,
        attrs={"long_name": "duration where conditional phase equals pi", "units": "ns"},
    )

    phase_sum_at_optimal = (
        float(_phase_mod_2pi(np.pi)) if success and np.isfinite(t_pi) else 0.0
    )

    result = GeometricCZDurationFitResult(
        exchange_amplitude=float(exchange_amplitude),
        optimal_duration=float(t_pi) if np.isfinite(t_pi) else 0.0,
        phase_ctrl_ground_frequency=0.0,
        phase_ctrl_excited_frequency=0.0,
        phase_ctrl_ground_phase=0.0,
        phase_ctrl_excited_phase=0.0,
        phase_sum_at_optimal=phase_sum_at_optimal,
        conditional_phase_rate=conditional_phase_rate,
        success=success,
    )

    fit_vars[f"_result_dict_{name}"] = asdict(result)
    return fit_vars


def log_fitted_results(
    fit_results: dict[str, dict[str, Any]],
    log_callable: Any | None = None,
) -> None:
    """Log fixed-amplitude geometric cphase fit results for all qubit pairs."""
    _log = log_callable or logger.info
    for name, r in sorted(fit_results.items()):
        status = "OK" if r["success"] else "FAILED"
        rate = r.get("conditional_phase_rate", 0.0) or 0.0
        f0 = r.get("phase_ctrl_ground_frequency", 0.0) or 0.0
        f1 = r.get("phase_ctrl_excited_frequency", 0.0) or 0.0
        if f0 != 0.0 or f1 != 0.0:
            msg = (
                f"  {name}: [{status}] "
                f"V = {r['exchange_amplitude']:.4f} V, "
                f"t_pi = {r['optimal_duration']:.1f} ns, "
                f"f0 = {f0 * 1e3:.3f} MHz, "
                f"f1 = {f1 * 1e3:.3f} MHz"
            )
        else:
            msg = (
                f"  {name}: [{status}] "
                f"V = {r['exchange_amplitude']:.4f} V, "
                f"t_pi = {r['optimal_duration']:.1f} ns, "
                f"dphi/dt = {rate * 1e3:.5f} mrad/ns"
            )
        _log(msg)
