"""Fixed-duration geometric CZ cphase amplitude analysis.

The exchange coupling J increases exponentially with the barrier gate voltage V,
so the phase accumulated at a fixed exchange duration t_fix is:

    φ(V) = φ_start + ΔΦ·(exp(β·(V − V₀)) − 1)

where V₀ = V[0] (sweep start), φ_start = φ(V₀), and ΔΦ is the total phase-swing
coefficient.  The signal from each cphase Ramsey branch follows an
exponentially-chirped sinusoid in V:

    signal(V) = A · cos(φ(V)) · exp(−γ·V) + C

The CZ operating condition is that the SUM of the exchange-induced phase
accumulations (referenced to V = V₀) from both branches equals π:

    Δφ_ground(V) + Δφ_excited(V) = π

where  Δφᵢ(V) = 2π·f₀ᵢ·(exp(βᵢ·V) − exp(βᵢ·V₀))

This accumulated-phase representation is monotonically increasing, starts
exactly at 0 at V = V₀, and makes the π-crossing unambiguous.

Fitting strategy
----------------
Both branches are driven by the same physical exchange coupling J(V) ∝ exp(β·V),
so β is identical for both.  A shared β is estimated jointly from the Hilbert
instantaneous phases of both branches, then each branch is fit with β held fixed.

1. Subtract DC; apply Hilbert transform to each signal to obtain the
   instantaneous accumulated phase.
2. Jointly estimate shared β via a 1-D grid search: for each β candidate the
   optimal ΔΦ for each branch is determined analytically (linear LS), and the
   combined residual over both phases selects the best β.
3. With β fixed, fit ΔΦ and phase for each branch individually (curve_fit
   refinement on the full signal).
4. Find V_π via bisection to machine precision.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np
import xarray as xr
from scipy.optimize import brentq, curve_fit
from scipy.signal import hilbert

logger = logging.getLogger(__name__)


# ── Fit result dataclasses ─────────────────────────────────────────────


@dataclass
class ExpChirpFitResult:
    """Fit parameters for one exponentially-chirped Ramsey branch."""

    f0: float = 0.0
    """Base frequency prefactor: Δφ(V) = 2π·f₀·(exp(β·V) − exp(β·V₀))  [rad/V_unit]."""
    beta: float = 0.0
    """Exponential growth rate (1/V)."""
    phi0: float = 0.0
    """Phase argument offset (internal use; φ(V) = 2π·f₀·exp(β·V) + φ₀)."""
    amplitude: float = 0.0
    """Signal envelope amplitude."""
    gamma: float = 0.0
    """Signal envelope exponential decay rate (1/V)."""
    offset: float = 0.0
    """DC offset."""
    V0: float = 0.0
    """Reference voltage (= V[0], sweep start); accumulated phase is 0 here."""
    success: bool = False


@dataclass
class GeometricCZAmplitudeFitResult:
    """Aggregated fixed-duration cphase fit result for one qubit pair."""

    exchange_duration: float = 0.0
    """Fixed exchange pulse duration used (ns)."""
    optimal_amplitude: float = 0.0
    """Exchange amplitude where summed accumulated phase reaches π (V)."""
    conditional_phase_at_optimal: float = 0.0
    """Conditional phase φ₁ − φ₀ at the selected amplitude (rad; quadrature analysis)."""
    phase_ctrl_ground_f0: float = 0.0
    """f₀ for control-ground branch."""
    phase_ctrl_ground_beta: float = 0.0
    """β (1/V) for control-ground branch."""
    phase_ctrl_excited_f0: float = 0.0
    """f₀ for control-excited branch."""
    phase_ctrl_excited_beta: float = 0.0
    """β (1/V) for control-excited branch."""
    phase_sum_at_optimal: float = 0.0
    """Accumulated phase sum at the selected amplitude (rad); ideally equals π."""
    success: bool = False


# ── Core helpers ───────────────────────────────────────────────────────


def _select_experiment_trace(
    data_array: xr.DataArray,
    experiment_type: int,
) -> np.ndarray:
    if "experiment_type" in data_array.coords:
        return data_array.sel(experiment_type=experiment_type).values.astype(np.float64)
    return data_array.values.astype(np.float64)[experiment_type]


def _accumulated_phase(V: np.ndarray, r: ExpChirpFitResult) -> np.ndarray:
    """Phase accumulated relative to V₀: Δφ(V) = 2π·f₀·(exp(β·V) − exp(β·V₀)).

    Exactly 0 at V = V₀ (sweep start) and grows monotonically with V.
    """
    return 2 * np.pi * r.f0 * (np.exp(r.beta * V) - np.exp(r.beta * r.V0))


def _phase_at(V: np.ndarray, r: ExpChirpFitResult) -> np.ndarray:
    """Absolute phase argument φ(V) = 2π·f₀·exp(β·V) + φ₀ (for signal reconstruction)."""
    return 2 * np.pi * r.f0 * np.exp(r.beta * V) + r.phi0


def _reconstructed_signal(V: np.ndarray, r: ExpChirpFitResult) -> np.ndarray:
    return r.amplitude * np.cos(_phase_at(V, r)) * np.exp(-r.gamma * V) + r.offset


def _hilbert_accum_phase(
    signal: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Return (accumulated_phase, phi_start, envelope, offset) via Hilbert transform."""
    offset = float(np.mean(signal))
    sig_centered = signal - offset
    analytic = hilbert(sig_centered)
    inst_phase = np.unwrap(np.angle(analytic))
    envelope = np.abs(analytic)
    phi_start = float(inst_phase[0])
    accum = inst_phase - inst_phase[0]
    return accum, phi_start, envelope, offset


# ── Shared β estimation ────────────────────────────────────────────────


def _estimate_shared_beta(
    V: np.ndarray,
    accum_phase_0: np.ndarray,
    accum_phase_1: np.ndarray,
) -> float:
    """1-D grid search for the β shared by both Ramsey branches.

    For each β candidate the optimal ΔΦ for each branch is computed in closed
    form (linear LS), and the combined residual over both accumulated phases
    selects β.  Using both branches jointly halves the noise on the β estimate
    and enforces the physical constraint that both oscillations share the same
    exponential growth rate.
    """
    V0 = V[0]
    beta_grid = np.logspace(0, np.log10(100), 400)
    best_beta = 5.0
    best_residual = np.inf

    for beta_try in beta_grid:
        basis = np.exp(beta_try * (V - V0)) - 1.0
        bb = float(np.dot(basis, basis))
        if bb < 1e-12:
            continue

        DPhi0 = float(np.dot(basis, accum_phase_0)) / bb
        DPhi1 = float(np.dot(basis, accum_phase_1)) / bb

        if DPhi0 <= 0 or DPhi1 <= 0:
            continue

        residual = float(
            np.sum((accum_phase_0 - DPhi0 * basis) ** 2)
            + np.sum((accum_phase_1 - DPhi1 * basis) ** 2)
        )
        if residual < best_residual:
            best_residual = residual
            best_beta = float(beta_try)

    return best_beta


# ── Per-branch fitting ─────────────────────────────────────────────────


def _fit_exp_chirp(
    V: np.ndarray,
    signal: np.ndarray,
    *,
    fixed_beta: float | None = None,
    accum_phase: np.ndarray | None = None,
    phi_start: float | None = None,
    envelope: np.ndarray | None = None,
    offset: float | None = None,
) -> ExpChirpFitResult:
    """Fit exponentially-chirped sinusoid to signal(V).

    When *fixed_beta* is given (shared β from the joint estimate), the grid
    search is skipped and only the linear ΔΦ and nonlinear refinement steps run.

    Pre-computed Hilbert quantities (accum_phase, phi_start, envelope, offset)
    may be passed to avoid recomputing them.
    """
    V = np.asarray(V, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)

    if len(V) < 8 or np.std(signal) < 1e-10:
        return ExpChirpFitResult()

    V0 = V[0]

    if accum_phase is None or phi_start is None or envelope is None or offset is None:
        accum_phase, phi_start, envelope, offset = _hilbert_accum_phase(signal)

    # ── Determine β ───────────────────────────────────────────────────
    if fixed_beta is not None:
        best_beta = float(fixed_beta)
        basis = np.exp(best_beta * (V - V0)) - 1.0
        bb = float(np.dot(basis, basis))
        best_DeltaPhi = (
            float(np.dot(basis, accum_phase)) / bb if bb > 1e-12 else max(float(accum_phase[-1]), 0.1)
        )
    else:
        # 1-D grid search
        beta_grid = np.logspace(0, np.log10(100), 200)
        best_beta = 5.0
        best_DeltaPhi = max(float(accum_phase[-1]), 0.1)
        best_residual = np.inf

        for beta_try in beta_grid:
            basis_try = np.exp(beta_try * (V - V0)) - 1.0
            bb = float(np.dot(basis_try, basis_try))
            if bb < 1e-12:
                continue
            DeltaPhi_try = float(np.dot(basis_try, accum_phase)) / bb
            if DeltaPhi_try <= 0:
                continue
            residual = float(np.sum((accum_phase - DeltaPhi_try * basis_try) ** 2))
            if residual < best_residual:
                best_residual = residual
                best_beta = float(beta_try)
                best_DeltaPhi = DeltaPhi_try

    # ── Nonlinear refinement of ΔΦ (β held fixed when shared_beta given) ──
    def _accum_model(V_: np.ndarray, log_DeltaPhi: float, beta: float) -> np.ndarray:
        return np.exp(log_DeltaPhi) * (np.exp(beta * (V_ - V0)) - 1.0)

    success = False
    DeltaPhi_fit, beta_fit = best_DeltaPhi, best_beta

    p0 = [np.log(max(best_DeltaPhi, 1e-8)), best_beta]
    if fixed_beta is not None:
        # Only free parameter is ΔΦ; β is pinned
        def _accum_model_fixed(V_: np.ndarray, log_DeltaPhi: float) -> np.ndarray:
            return np.exp(log_DeltaPhi) * (np.exp(fixed_beta * (V_ - V0)) - 1.0)

        try:
            popt, _ = curve_fit(
                _accum_model_fixed, V, accum_phase,
                p0=[p0[0]], bounds=([-20.0], [20.0]),
                maxfev=2_000,
            )
            DeltaPhi_fit = float(np.exp(popt[0]))
            beta_fit = fixed_beta
            success = True
        except Exception:
            success = True  # linear LS solution is still valid
    else:
        try:
            popt, _ = curve_fit(
                _accum_model, V, accum_phase,
                p0=p0, bounds=([-20.0, 0.1], [20.0, 100.0]),
                maxfev=5_000,
            )
            DeltaPhi_fit = float(np.exp(popt[0]))
            beta_fit = float(popt[1])
            success = True
        except Exception:
            success = best_DeltaPhi > 0

    # ── Convert to (f₀, β, φ₀) interface ─────────────────────────────
    # φ_start + ΔΦ·(exp(β·(V−V₀))−1)
    #   = (φ_start − ΔΦ) + ΔΦ·exp(−β·V₀)·exp(β·V)
    #   ≡ φ₀ + 2π·f₀·exp(β·V)
    f0_fit = DeltaPhi_fit * np.exp(-beta_fit * V0) / (2 * np.pi)
    phi0_fit = float(phi_start) - DeltaPhi_fit

    # ── Envelope fit for A and γ ──────────────────────────────────────
    amp_fit = float(np.mean(envelope))
    gamma_fit = 0.0
    if np.all(envelope > 0):
        try:
            A_mat = np.column_stack([np.ones_like(V), -V])
            coeffs = np.linalg.lstsq(A_mat, np.log(envelope), rcond=None)[0]
            amp_fit = float(np.exp(coeffs[0]))
            gamma_fit = max(float(coeffs[1]), 0.0)
        except Exception:
            pass

    return ExpChirpFitResult(
        f0=f0_fit,
        beta=beta_fit,
        phi0=phi0_fit,
        amplitude=amp_fit,
        gamma=gamma_fit,
        offset=float(offset),
        V0=float(V0),
        success=success,
    )


# ── Pi-crossing ────────────────────────────────────────────────────────


def _interp_v_at_phase(
    V: np.ndarray,
    phi_unwrapped: np.ndarray,
    target: float = np.pi,
) -> tuple[float, bool]:
    """First amplitude where unwrapped phase reaches *target* (linear interpolation)."""
    above = phi_unwrapped >= target
    if not np.any(above):
        return np.nan, False
    idx = int(np.argmax(above))
    if idx == 0:
        v_pi = float(V[0])
        return v_pi, bool(V[0] <= v_pi <= V[-1])
    v_lo, v_hi = float(V[idx - 1]), float(V[idx])
    p_lo, p_hi = float(phi_unwrapped[idx - 1]), float(phi_unwrapped[idx])
    if abs(p_hi - p_lo) < 1e-15:
        return v_hi, bool(V[0] <= v_hi <= V[-1])
    frac = (target - p_lo) / (p_hi - p_lo)
    v_pi = float(v_lo + frac * (v_hi - v_lo))
    return v_pi, bool(V[0] <= v_pi <= V[-1])


def _fit_quadrature_phases(
    ds_raw: xr.Dataset,
    var_name: str,
    amplitudes: np.ndarray,
    quadrature_signal_center: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    bool,
]:
    """Return (phi0_u, phi1_u, delta_u, v_pi, cond_at_opt, root_in_sweep)."""
    data = ds_raw[var_name]
    center = float(quadrature_signal_center)
    i0 = data.sel(control_state=0, analysis_axis=0).values.astype(np.float64)
    q0 = data.sel(control_state=0, analysis_axis=1).values.astype(np.float64)
    i1 = data.sel(control_state=1, analysis_axis=0).values.astype(np.float64)
    q1 = data.sel(control_state=1, analysis_axis=1).values.astype(np.float64)

    phi0 = np.arctan2(q0 - center, i0 - center)
    phi1 = np.arctan2(q1 - center, i1 - center)

    phi0_u = np.unwrap(phi0)
    phi1_u = np.unwrap(phi1)

    # Conditional phase via complex conjugate product.
    # angle(−z1·z0*) = φ₁−φ₀−π, starting near 0 → safe for unwrap.
    z0 = (i0 - center) + 1j * (q0 - center)
    z1 = (i1 - center) + 1j * (q1 - center)
    delta_u = np.unwrap(np.angle(-z1 * np.conj(z0)))

    v_pi, root_ok = _interp_v_at_phase(amplitudes, delta_u, np.pi)
    cond_at_opt = (
        float(np.interp(v_pi, amplitudes, delta_u))
        if np.isfinite(v_pi) and root_ok
        else float("nan")
    )
    return phi0_u, phi1_u, delta_u, v_pi, cond_at_opt, root_ok


def _find_phase_sum_pi(
    V_sweep: np.ndarray,
    r_ground: ExpChirpFitResult,
    r_excited: ExpChirpFitResult,
) -> tuple[float, bool]:
    """Find the first V in the sweep where accumulated phase sum = π.

    Δφ_sum(V) = Δφ_ground(V) + Δφ_excited(V)

    Both terms are monotonically increasing and zero at V₀, so the sum is
    monotone and the crossing is unique.  After coarse bracketing, bisection
    refines to machine precision.
    """

    def _sum_at(v: float) -> float:
        v_arr = np.array([v])
        return float(
            _accumulated_phase(v_arr, r_ground)[0]
            + _accumulated_phase(v_arr, r_excited)[0]
        )

    V_dense = np.linspace(V_sweep[0], V_sweep[-1], 10_000)
    accum = _accumulated_phase(V_dense, r_ground) + _accumulated_phase(V_dense, r_excited)

    above = accum >= np.pi
    if not np.any(above):
        return np.nan, False

    idx = int(np.argmax(above))
    if idx == 0:
        v_pi = float(V_dense[0])
        return v_pi, V_sweep[0] <= v_pi <= V_sweep[-1]

    v_lo, v_hi = float(V_dense[idx - 1]), float(V_dense[idx])
    try:
        v_pi = float(brentq(lambda v: _sum_at(v) - np.pi, v_lo, v_hi, xtol=1e-12))
        return v_pi, V_sweep[0] <= v_pi <= V_sweep[-1]
    except Exception:
        frac = (np.pi - accum[idx - 1]) / (accum[idx] - accum[idx - 1])
        v_pi = float(v_lo + frac * (v_hi - v_lo))
        return v_pi, True


# ── Public API ─────────────────────────────────────────────────────────


def fit_raw_data(
    ds_raw: xr.Dataset,
    qubit_pairs: list[Any],
    *,
    exchange_duration: float,
    analysis_signal: str = "E_p2_given_p1_0",
    quadrature_signal_center: float = 0.5,
) -> tuple[xr.Dataset, dict[str, dict[str, Any]]]:
    """Run fixed-duration geometric cphase amplitude analysis for every qubit pair.

    With ``analysis_axis`` in the raw data, phases are extracted via arctan2(X90, Y90)
    quadratures and the conditional phase φ₁−φ₀−π is computed vs amplitude until it
    reaches π. Legacy datasets with only ``experiment_type`` use exponential-chirp
    fitting and the accumulated-phase sum criterion.
    """
    amplitudes = ds_raw.coords["exchange_amplitude"].values.astype(np.float64)
    fit_results: Dict[str, dict[str, Any]] = {}
    fit_vars: dict[str, xr.DataArray] = {}

    for qp in qubit_pairs:
        var_name = f"{analysis_signal}_{qp.name}"
        if var_name not in ds_raw.data_vars:
            logger.warning("No %s variable for pair %s; skipping.", var_name, qp.name)
            fr = asdict(GeometricCZAmplitudeFitResult())
            fr["quadrature_analysis"] = False
            fit_results[qp.name] = fr
            continue

        data_array = ds_raw[var_name]
        use_quadrature = "analysis_axis" in data_array.dims

        if use_quadrature:
            phi0_u, phi1_u, delta_u, v_pi, cond_at_opt, root_in_sweep = _fit_quadrature_phases(
                ds_raw,
                var_name,
                amplitudes,
                quadrature_signal_center,
            )
            success = bool(
                len(amplitudes) >= 2
                and np.isfinite(v_pi)
                and root_in_sweep
            )
            cond_val = float(cond_at_opt) if np.isfinite(cond_at_opt) else 0.0
            result = GeometricCZAmplitudeFitResult(
                exchange_duration=float(exchange_duration),
                optimal_amplitude=float(v_pi) if np.isfinite(v_pi) else 0.0,
                conditional_phase_at_optimal=cond_val,
                phase_sum_at_optimal=cond_val,
                success=success,
            )
            out = asdict(result)
            out["quadrature_analysis"] = True
            fit_results[qp.name] = out

            fit_vars[f"phi_ctrl_ground_{qp.name}"] = xr.DataArray(
                phi0_u,
                dims=["exchange_amplitude"],
                attrs={
                    "long_name": "Ramsey phase (control |0>, unwrapped)",
                    "units": "rad",
                },
            )
            fit_vars[f"phi_ctrl_excited_{qp.name}"] = xr.DataArray(
                phi1_u,
                dims=["exchange_amplitude"],
                attrs={
                    "long_name": "Ramsey phase (control |1>, unwrapped)",
                    "units": "rad",
                },
            )
            fit_vars[f"conditional_phase_{qp.name}"] = xr.DataArray(
                delta_u,
                dims=["exchange_amplitude"],
                attrs={
                    "long_name": "conditional phase φ₁−φ₀−π",
                    "units": "rad",
                },
            )
            fit_vars[f"v_pi_cphase_{qp.name}"] = xr.DataArray(
                float(v_pi) if np.isfinite(v_pi) else np.nan,
                attrs={
                    "long_name": "amplitude where conditional phase = π",
                    "units": "V",
                },
            )
            continue

        sig_ground = _select_experiment_trace(data_array, 0)
        sig_excited = _select_experiment_trace(data_array, 1)

        accum_0, phi_start_0, envelope_0, offset_0 = _hilbert_accum_phase(sig_ground)
        accum_1, phi_start_1, envelope_1, offset_1 = _hilbert_accum_phase(sig_excited)

        shared_beta = _estimate_shared_beta(amplitudes, accum_0, accum_1)

        fit_ground = _fit_exp_chirp(
            amplitudes,
            sig_ground,
            fixed_beta=shared_beta,
            accum_phase=accum_0,
            phi_start=phi_start_0,
            envelope=envelope_0,
            offset=offset_0,
        )
        fit_excited = _fit_exp_chirp(
            amplitudes,
            sig_excited,
            fixed_beta=shared_beta,
            accum_phase=accum_1,
            phi_start=phi_start_1,
            envelope=envelope_1,
            offset=offset_1,
        )

        v_pi, root_in_sweep = np.nan, False
        if fit_ground.success and fit_excited.success:
            v_pi, root_in_sweep = _find_phase_sum_pi(amplitudes, fit_ground, fit_excited)

        phase_sum_at_optimal = np.nan
        if np.isfinite(v_pi):
            v_arr = np.array([v_pi])
            phase_sum_at_optimal = float(
                _accumulated_phase(v_arr, fit_ground)[0]
                + _accumulated_phase(v_arr, fit_excited)[0]
            )

        success = bool(fit_ground.success and fit_excited.success and root_in_sweep)

        result = GeometricCZAmplitudeFitResult(
            exchange_duration=float(exchange_duration),
            optimal_amplitude=float(v_pi) if np.isfinite(v_pi) else 0.0,
            conditional_phase_at_optimal=0.0,
            phase_ctrl_ground_f0=float(fit_ground.f0),
            phase_ctrl_ground_beta=float(fit_ground.beta),
            phase_ctrl_excited_f0=float(fit_excited.f0),
            phase_ctrl_excited_beta=float(fit_excited.beta),
            phase_sum_at_optimal=(
                float(phase_sum_at_optimal) if np.isfinite(phase_sum_at_optimal) else 0.0
            ),
            success=success,
        )
        out = asdict(result)
        out["quadrature_analysis"] = False
        fit_results[qp.name] = out

        if fit_ground.success:
            fit_vars[f"fitted_phase_ctrl_ground_{qp.name}"] = xr.DataArray(
                _reconstructed_signal(amplitudes, fit_ground),
                dims=["exchange_amplitude"],
                attrs={"long_name": "fit phase signal (control ground)", "units": ""},
            )
        if fit_excited.success:
            fit_vars[f"fitted_phase_ctrl_excited_{qp.name}"] = xr.DataArray(
                _reconstructed_signal(amplitudes, fit_excited),
                dims=["exchange_amplitude"],
                attrs={"long_name": "fit phase signal (control excited)", "units": ""},
            )

        fit_vars[f"phase_sum_{qp.name}"] = xr.DataArray(
            _accumulated_phase(amplitudes, fit_ground)
            + _accumulated_phase(amplitudes, fit_excited),
            dims=["exchange_amplitude"],
            attrs={
                "long_name": "accumulated phase sum (ref V=V_min)",
                "units": "rad",
            },
        )
        fit_vars[f"v_pi_cphase_{qp.name}"] = xr.DataArray(
            float(v_pi) if np.isfinite(v_pi) else np.nan,
            attrs={"long_name": "amplitude where accumulated phase sum = π", "units": "V"},
        )

    ds_fit = xr.Dataset(
        fit_vars,
        coords={"exchange_amplitude": amplitudes},
        attrs={"exchange_duration": float(exchange_duration)},
    )
    return ds_fit, fit_results


def log_fitted_results(
    fit_results: dict[str, dict[str, Any]],
    log_callable: Any | None = None,
) -> None:
    """Log fixed-duration geometric cphase amplitude fit results for all qubit pairs."""
    _log = log_callable or logger.info
    for name, r in sorted(fit_results.items()):
        status = "OK" if r["success"] else "FAILED"
        if r.get("quadrature_analysis"):
            msg = (
                f"  {name}: [{status}] "
                f"t = {r['exchange_duration']:.0f} ns, "
                f"V_π = {r['optimal_amplitude']:.4f} V, "
                f"Δφ = {r['conditional_phase_at_optimal']:.3f} rad"
            )
        else:
            msg = (
                f"  {name}: [{status}] "
                f"t = {r['exchange_duration']:.0f} ns, "
                f"V_π = {r['optimal_amplitude']:.4f} V, "
                f"β_ground = {r['phase_ctrl_ground_beta']:.2f} /V, "
                f"β_excited = {r['phase_ctrl_excited_beta']:.2f} /V"
            )
        _log(msg)