"""Analysis module for Context Aware Fidelity Estimation (CAFE) of a CZ gate.

Reference: D. M. Debroy et al., "Context Aware Fidelity Estimation", arXiv:2303.17565.

The state-averaged return probability F_n after n cycles is fitted with Eq. (B9) of the
paper, for a cycle that applies the unitary C with probability 1 - p_depol and fully
depolarizes otherwise:

    F_n = 1/d + (1 - p)^n [ (d + |tr((C_ref^†)^n C^n)|^2) / (d (d + 1)) - 1/d ] - eps_spam,

with d = 4. C is built from ``fsim_unitary(Δθ, Δγ, Δφ)`` (Eq. 2) and C_ref from the
reference gate, both followed by X⊗X for DECAF. With an ideal CZ reference this is Eq. (4).

The error budget follows Eqs. (5)-(7), evaluating the fitted curve at n = 1:

    1 - F      = 1 - F_1 / (1 - eps_spam)
    eps_incoh  = 1 - F_1(no coherent error) / (1 - eps_spam)
    eps_coh    = 1 - F_1(p_depol = 0) / (1 - eps_spam)

The paper's Eqs. (6) and (7) have the two conditions swapped; the definitions above follow
its text ("eps_incoh estimates the average gate infidelity when no coherent control errors
are present").

A quadratic fit F_n ≈ a - b n - c n^2 over the shallow depths (Eq. 8) is reported as a
cross-check: eps_incoh ≈ b / a and eps_coh ≈ c / a.

For DECAF, Δγ only enters as a global phase at even n, so it is fixed to zero.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from scipy.optimize import least_squares

from .circuits import CZ, NUM_STATES, cycle_unitary, fsim_unitary

DIM = 4
PARAM_NAMES = ("p_depol", "delta_theta", "delta_gamma", "delta_phi", "spam")
_LOWER = np.array([0.0, 0.0, -0.5, -0.5, -0.2])
_UPPER = np.array([0.999, 0.5, 0.5, 0.5, 0.5])
MIN_R_SQUARED = 0.9
MAX_CHI2_RED_UNCONVERGED = 4.0


@dataclass
class QuadraticBudget:
    """Error budget from the quadratic approximation of Eq. (8)."""

    fidelity: float = float("nan")
    incoherent_error: float = float("nan")
    coherent_error: float = float("nan")
    max_depth: int = 0


@dataclass
class FitResults:
    """CAFE fit results for one qubit pair and one variant (CAFE or DECAF)."""

    variant: str
    success: bool
    fidelity: float = float("nan")
    """Average gate fidelity of one cycle against the reference, corrected for SPAM (Eq. 5)."""
    fidelity_error: float = float("nan")
    incoherent_error: float = float("nan")
    """Infidelity with coherent errors removed (Eq. 6)."""
    incoherent_error_error: float = float("nan")
    coherent_error: float = float("nan")
    """Infidelity with incoherent errors removed (Eq. 7)."""
    coherent_error_error: float = float("nan")
    spam: float = float("nan")
    p_depol: float = float("nan")
    delta_theta: float = float("nan")
    """Fitted swap angle (rad). Individual angles are weakly constrained; trust the budget."""
    delta_gamma: float = float("nan")
    """Fitted single-qubit phase (rad). Fixed to 0 for DECAF."""
    delta_phi: float = float("nan")
    """Fitted conditional-phase error (rad)."""
    r_squared: float = float("nan")
    quadratic: QuadraticBudget = field(default_factory=QuadraticBudget)
    message: str = ""
    """Why the fit was flagged as failed (empty on success)."""


def model_fidelity(
    depths: np.ndarray,
    p_depol: float,
    delta_theta: float,
    delta_gamma: float,
    delta_phi: float,
    spam: float,
    variant: str = "cafe",
    reference_gate: np.ndarray = CZ,
) -> np.ndarray:
    """State-averaged return probability after ``depths`` cycles (Eq. B9 minus SPAM)."""
    depths = np.asarray(depths, dtype=int)
    cycle = cycle_unitary(fsim_unitary(delta_theta, delta_gamma, delta_phi), variant)
    cycle_ref = cycle_unitary(reference_gate, variant)
    overlap_sq = np.array([_overlap_sq(cycle, cycle_ref, int(n)) for n in depths])
    return _fidelity_from_overlap(depths, p_depol, overlap_sq) - spam


def _overlap_sq(cycle: np.ndarray, cycle_ref: np.ndarray, n: int) -> float:
    """|tr((C_ref^†)^n C^n)|^2."""
    return float(abs(np.trace(np.linalg.matrix_power(cycle_ref.conj().T, n) @ np.linalg.matrix_power(cycle, n))) ** 2)


def _fidelity_from_overlap(depths: np.ndarray, p_depol: float, overlap_sq: np.ndarray) -> np.ndarray:
    survival = (1 - p_depol) ** np.asarray(depths, dtype=float)
    return 1 / DIM + survival * ((DIM + overlap_sq) / (DIM * (DIM + 1)) - 1 / DIM)


def error_budget(params: np.ndarray, variant: str, reference_gate: np.ndarray) -> np.ndarray:
    """Return (1 - F, eps_incoh, eps_coh) from the fitted parameters (Eqs. 5-7)."""
    p_depol, delta_theta, delta_gamma, delta_phi, spam = params
    norm = 1 - spam
    f1 = model_fidelity([1], p_depol, delta_theta, delta_gamma, delta_phi, spam, variant, reference_gate)[0]
    f1_no_coherent = _fidelity_from_overlap(np.array([1]), p_depol, np.array([DIM**2]))[0] - spam
    f1_no_incoherent = model_fidelity([1], 0.0, delta_theta, delta_gamma, delta_phi, spam, variant, reference_gate)[0]
    return np.array([1 - f1 / norm, 1 - f1_no_coherent / norm, 1 - f1_no_incoherent / norm])


def _propagate(func: Callable[[np.ndarray], np.ndarray], params: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Standard deviations of ``func(params)`` from the parameter covariance (linear propagation)."""
    base = func(params)
    jac = np.zeros((base.size, params.size))
    for i in range(params.size):
        step = 1e-6 * max(1.0, abs(params[i]))
        shifted = params.copy()
        shifted[i] += step
        jac[:, i] = (func(shifted) - base) / step
    return np.sqrt(np.clip(np.diag(jac @ cov @ jac.T), 0, None))


def fit_cafe_curve(
    depths: np.ndarray,
    fidelity: np.ndarray,
    sigma: np.ndarray,
    variant: str = "cafe",
    reference_gate: np.ndarray = CZ,
    quadratic_max_depth: int = 4,
) -> Tuple[FitResults, np.ndarray]:
    """Fit one CAFE curve and return the fit results and the best-fit parameter vector."""
    depths = np.asarray(depths, dtype=int)
    fidelity = np.asarray(fidelity, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-4)
    free = np.ones(len(PARAM_NAMES), dtype=bool)
    if variant == "decaf":
        free[PARAM_NAMES.index("delta_gamma")] = False

    def full_params(x: np.ndarray) -> np.ndarray:
        params = np.zeros(len(PARAM_NAMES))
        params[free] = x
        return params

    def residuals(x: np.ndarray) -> np.ndarray:
        model = model_fidelity(depths, *full_params(x), variant=variant, reference_gate=reference_gate)
        return (model - fidelity) / sigma

    finite = np.isfinite(fidelity)
    if finite.sum() < 3:
        return FitResults(variant=variant, success=False, message="fewer than 3 finite data points"), np.full(5, np.nan)
    depths, fidelity, sigma = depths[finite], fidelity[finite], sigma[finite]

    spam_guess = float(np.clip(1 - fidelity[np.argmin(depths)], _LOWER[4], _UPPER[4]))
    best = None
    for theta0 in (0.01, 0.1):
        for gamma0 in (-0.1, 0.0, 0.1):
            for phi0 in (-0.1, 0.0, 0.1):
                x0 = np.array([0.01, theta0, gamma0, phi0, spam_guess])[free]
                try:
                    result = least_squares(
                        residuals, x0, bounds=(_LOWER[free], _UPPER[free]), method="trf", max_nfev=500
                    )
                except ValueError:
                    continue
                if best is None or result.cost < best.cost:
                    best = result
    if best is None:
        return FitResults(variant=variant, success=False, message="optimizer failed"), np.full(5, np.nan)

    params = full_params(best.x)
    dof = max(len(depths) - free.sum(), 1)
    chi2_red = 2 * best.cost / dof
    cov_free = np.linalg.pinv(best.jac.T @ best.jac) * max(1.0, chi2_red)
    cov = np.zeros((len(PARAM_NAMES), len(PARAM_NAMES)))
    cov[np.ix_(free, free)] = cov_free

    budget = error_budget(params, variant, reference_gate)
    budget_err = _propagate(lambda p: error_budget(p, variant, reference_gate), params, cov)

    model = model_fidelity(depths, *params, variant=variant, reference_gate=reference_gate)
    ss_res = float(np.sum((fidelity - model) ** 2))
    ss_tot = float(np.sum((fidelity - fidelity.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    infidelity, eps_incoh, eps_coh = budget
    results = FitResults(
        variant=variant,
        success=True,
        fidelity=float(1 - infidelity),
        fidelity_error=float(budget_err[0]),
        incoherent_error=float(eps_incoh),
        incoherent_error_error=float(budget_err[1]),
        coherent_error=float(eps_coh),
        coherent_error_error=float(budget_err[2]),
        spam=float(params[4]),
        p_depol=float(params[0]),
        delta_theta=float(params[1]),
        delta_gamma=float(params[2]),
        delta_phi=float(params[3]),
        r_squared=float(r_squared),
        quadratic=fit_quadratic(depths, fidelity, sigma, quadratic_max_depth),
    )
    _apply_success_rules(results, ss_tot, float(np.sum(sigma**2)), best, chi2_red)
    return results, params


def _apply_success_rules(
    results: FitResults, ss_tot: float, noise_var: float, optimizer_result, chi2_red: float
) -> None:
    """Flag the fit as failed if it did not converge or gives an unphysical budget."""
    reasons: List[str] = []
    # Near an ideal gate the cost is flat along the angles, so the optimizer can stop on its
    # evaluation limit with an already good fit; accept it when the residuals match the noise.
    if not optimizer_result.success and chi2_red > MAX_CHI2_RED_UNCONVERGED:
        reasons.append(f"optimizer did not converge ({optimizer_result.message})")
    if not 0.0 <= results.fidelity <= 1.0:
        reasons.append(f"fidelity {results.fidelity:.4f} outside [0, 1]")
    if results.incoherent_error < -results.incoherent_error_error:
        reasons.append("negative incoherent error")
    if results.coherent_error < -results.coherent_error_error:
        reasons.append("negative coherent error")
    # R² is only meaningful when the data varies by more than the shot noise
    if ss_tot > noise_var and results.r_squared < MIN_R_SQUARED:
        reasons.append(f"R² = {results.r_squared:.3f} < {MIN_R_SQUARED}")
    if reasons:
        results.success = False
        results.message = "; ".join(reasons)


def fit_quadratic(depths: np.ndarray, fidelity: np.ndarray, sigma: np.ndarray, max_depth: int) -> QuadraticBudget:
    """Fit F_n ≈ a - b n - c n^2 over depths <= ``max_depth`` (Eq. 8)."""
    mask = np.asarray(depths) <= max_depth
    if mask.sum() < 3:
        return QuadraticBudget(max_depth=max_depth)
    n = np.asarray(depths, dtype=float)[mask]
    c2, c1, c0 = np.polyfit(n, np.asarray(fidelity)[mask], 2, w=1 / np.asarray(sigma)[mask])
    a, b, c = c0, -c1, -c2
    return QuadraticBudget(
        fidelity=float(1 - (b + c) / a),
        incoherent_error=float(b / a),
        coherent_error=float(c / a),
        max_depth=int(max_depth),
    )


def log_fitted_results(fit_results: Dict[str, Dict[str, FitResults]], log_callable=None):
    """Log the CAFE error budget per qubit pair and variant."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for qp_name, per_variant in fit_results.items():
        for variant, fr in per_variant.items():
            header = f"Results for qubit pair {qp_name} ({variant.upper()}): " + (
                "SUCCESS!\n" if fr.success else f"FAIL! {fr.message}\n"
            )
            body = (
                f"\tCycle fidelity F = {fr.fidelity:.5f} ± {fr.fidelity_error:.5f}\n"
                f"\tIncoherent error = {fr.incoherent_error:.2e} ± {fr.incoherent_error_error:.1e}\n"
                f"\tCoherent error   = {fr.coherent_error:.2e} ± {fr.coherent_error_error:.1e}\n"
                f"\tSPAM offset      = {fr.spam:.4f}, R² = {fr.r_squared:.3f}\n"
                f"\tQuadratic check (n ≤ {fr.quadratic.max_depth}): F = {fr.quadratic.fidelity:.5f}, "
                f"incoh = {fr.quadratic.incoherent_error:.2e}, coh = {fr.quadratic.coherent_error:.2e}"
            )
            log_callable(header + body)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Average the return probability over the 16 SIC states and add its binomial error."""
    shots = node.parameters.num_shots
    p = ds.p_return
    fidelity = p.mean(dim="state")
    # Each state contributes an independent binomial estimate of P(|00>)
    fidelity_std = np.sqrt((p * (1 - p)).sum(dim="state") / shots) / NUM_STATES
    ds = ds.assign(fidelity=fidelity, fidelity_std=fidelity_std)
    ds.fidelity.attrs = {"long_name": "state-averaged return probability", "units": ""}
    ds.fidelity_std.attrs = {"long_name": "binomial standard error", "units": ""}
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, Dict[str, FitResults]]]:
    """Fit every (qubit pair, variant) CAFE curve and add the fitted curves to the dataset."""
    reference_gates: Dict[str, np.ndarray] = node.namespace["reference_gates"]
    quadratic_max_depth = node.parameters.quadratic_fit_max_depth

    depths = ds.depth.values
    depth_fine = np.arange(0, int(depths.max()) + 1)
    qp_names = [str(name) for name in ds.qubit_pair.values]
    variants = [str(v) for v in ds.variant.values]

    fit_results: Dict[str, Dict[str, FitResults]] = {}
    curves = np.full((len(qp_names), len(variants), len(depth_fine)), np.nan)
    quad_curves = np.full_like(curves, np.nan)
    for iq, qp_name in enumerate(qp_names):
        fit_results[qp_name] = {}
        for iv, variant in enumerate(variants):
            sel = ds.sel(qubit_pair=qp_name, variant=variant)
            fr, params = fit_cafe_curve(
                depths,
                sel.fidelity.values,
                sel.fidelity_std.values,
                variant=variant,
                reference_gate=reference_gates[qp_name],
                quadratic_max_depth=quadratic_max_depth,
            )
            fit_results[qp_name][variant] = fr
            if np.all(np.isfinite(params)):
                curves[iq, iv] = model_fidelity(depth_fine, *params, variant, reference_gates[qp_name])
            if np.isfinite(fr.quadratic.fidelity):
                quad_curves[iq, iv] = _quadratic_curve(depths, sel, fr.quadratic.max_depth, depth_fine)

    coords = {"qubit_pair": qp_names, "variant": variants, "depth_fine": depth_fine}
    dims = ("qubit_pair", "variant", "depth_fine")
    ds_fit = ds.assign(
        fit_curve=xr.DataArray(curves, dims=dims, coords=coords),
        quadratic_curve=xr.DataArray(quad_curves, dims=dims, coords=coords),
    )
    for key in ("fidelity", "incoherent_error", "coherent_error", "spam", "success"):
        values = [[getattr(fit_results[q][v], key) for v in variants] for q in qp_names]
        ds_fit = ds_fit.assign_coords({f"fit_{key}": (("qubit_pair", "variant"), np.array(values))})
    return ds_fit, fit_results


def _quadratic_curve(depths: np.ndarray, sel: xr.Dataset, max_depth: int, depth_fine: np.ndarray) -> np.ndarray:
    mask = depths <= max_depth
    sigma = np.maximum(sel.fidelity_std.values[mask], 1e-4)
    coeffs = np.polyfit(depths[mask], sel.fidelity.values[mask], 2, w=1 / sigma)
    curve = np.polyval(coeffs, depth_fine).astype(float)
    curve[depth_fine > max_depth] = np.nan
    return curve


def primary_variant(variants) -> Optional[str]:
    """Variant used for the node outcome: CAFE if it was measured, otherwise the first one."""
    variants = list(variants)
    if not variants:
        return None
    return "cafe" if "cafe" in variants else variants[0]
