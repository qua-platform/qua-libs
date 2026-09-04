"""Multi-exponential fit of flux-line step / Ramsey-tail responses.

Shared by qubit and coupler long-distortion nodes (qubitspec + Ramsey) and by
the IIR stage of short-distortion (cryoscope) analysis.

Fit pipeline:

    flowchart TD
        ty["y(t) flux step / Ramsey-tail response"]
            --> weights["residual weights\n  (√log-spacing on log grids)"]
        weights --> seed["_seed_params: log-spaced taus\n  + linear LS for (a_i, a_dc)"]
        seed --> nls["_fit_once: joint nonlinear LS\n  params [a_dc, a_1, log_tau_1, …]"]
        nls --> prune{"prune / retry with fewer n?\n  solver fail | |a_i| tiny | taus too close"}
        prune -->|yes| nls
        prune -->|no| out["FitParameters\n  a_dc, {(a_i, tau_i)}, rms"]
        out --> iir["IIR taps: A_i = a_i / a_dc"]

Models
------
1. Ideal step (``t_pulse_ns is None``) — Hellings et al. arXiv:2503.04610 Eq. (1):

       y(t) = a_dc + Σ_i a_i exp(-t / tau_i)

2. Finite-length pulse (``t_pulse_ns = T_pulse``) — Aggarwal et al.
   arXiv:2503.08645 Appendix H, Eq. (H1):

       y(t) = a_dc + Σ_i a_i (1 - exp(-T_pulse / tau_i)) exp(-t / tau_i)

   Returned ``a_i`` are de-attenuated (the charging factor is folded into the
   model, not into the stored amplitude), so IIR taps are always
   ``A_i = a_i / a_dc`` (Rol et al. arXiv:1907.04818 Eq. (S22)).

Pruning
-------
Starts at ``n_exponentials`` and retries with fewer components when the solver
fails, two taus lie within ``tau_proximity_factor``, or an amplitude falls below
``rel_amp_threshold · swing``. Bounds keep ``a_dc`` near the data range and
``tau`` in ``[TAU_MIN_NS, TAU_MAX_FACTOR · t_max]`` (further capped vs
``T_pulse`` in finite-pulse mode so ``1 - exp(-T/tau)`` stays numerically useful).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

# Bounds keep the fit physically usable for IIR generation:
#   - a_dc in [min(y), max(y)] padded by ADC_PADDING_FRAC · swing
#   - tau in [TAU_MIN_NS, TAU_MAX_FACTOR · t_max] (also capped vs T_pulse in finite-pulse mode)
TAU_MIN_NS = 0.1
TAU_MAX_FACTOR = 100.0
TAU_PULSE_FACTOR = 20.0  # keep (1 - exp(-T/tau)) above ~5% so deconvolution stays stable
ADC_PADDING_FRAC = 0.1


@dataclass
class FitParameters:
    """Long-distortion multi-exp fit parameters for one channel."""

    success: bool
    n_components_requested: int
    n_components_used: int
    a_tau_tuple: List[Tuple[float, float]]
    a_dc: float
    rms_error: float


def _make_residual_weights(t: NDArray[np.floating]) -> NDArray[np.floating]:
    """Uniform weights, or sqrt(log-spacing) on log grids so early points do not dominate."""
    if len(t) < 4:
        return np.ones_like(t, dtype=float)
    pos = t[t > 0]
    if len(pos) < 4:
        return np.ones_like(t, dtype=float)
    lin_cv = np.std(np.diff(t)) / max(np.mean(np.diff(t)), 1e-12)
    log_cv = np.std(np.diff(np.log(pos))) / max(np.mean(np.diff(np.log(pos))), 1e-12)
    if log_cv >= lin_cv:
        return np.ones_like(t, dtype=float)

    pos_min = pos.min()
    log_t = np.log(np.maximum(t, pos_min / 10))
    spacing = np.empty_like(log_t)
    spacing[1:-1] = (log_t[2:] - log_t[:-2]) / 2
    spacing[0] = log_t[1] - log_t[0]
    spacing[-1] = log_t[-1] - log_t[-2]
    return np.sqrt(np.maximum(spacing, 1e-12))


def _seed_params(
    t: NDArray[np.floating],
    y: NDArray[np.floating],
    n: int,
    t_pulse_ns: Optional[float] = None,
) -> Tuple[NDArray[np.floating], NDArray[np.floating], float]:
    """Log-spaced tau seeds plus closed-form linear LS for ``(amps, a_dc)``."""
    pos = t[t > 0]
    lo = float(pos.min()) / 3.0
    hi = float(pos.max()) * 10.0
    if t_pulse_ns is not None:
        hi = min(hi, TAU_PULSE_FACTOR * float(t_pulse_ns))
        if hi <= lo:
            hi = lo * 10.0
    taus = np.array([float(np.sqrt(lo * hi))]) if n == 1 else np.logspace(np.log10(lo), np.log10(hi), n)

    M = np.empty((len(t), n + 1))
    M[:, 0] = 1.0
    for i, tau in enumerate(taus):
        basis = np.exp(-t / tau)
        if t_pulse_ns is not None:
            basis = basis * (1.0 - np.exp(-float(t_pulse_ns) / tau))
        M[:, i + 1] = basis
    coefs, *_ = np.linalg.lstsq(M, y, rcond=None)
    return taus, coefs[1:], float(coefs[0])


def _fit_once(
    t: NDArray[np.floating],
    y: NDArray[np.floating],
    n: int,
    weights: NDArray[np.floating],
    t_pulse_ns: Optional[float] = None,
) -> dict[str, Any]:
    """One joint nonlinear LS with ``n`` components; params ``[a_dc, a_1, log_tau_1, …]``."""
    taus_seed, amps_seed, adc_seed = _seed_params(t, y, n, t_pulse_ns=t_pulse_ns)

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
        out = np.full_like(t, x[0], dtype=float)
        for i in range(n):
            amp = x[1 + 2 * i]
            tau = np.exp(x[2 + 2 * i])
            term = amp * np.exp(-t / tau)
            if t_pulse is not None:
                term = term * (1.0 - np.exp(-t_pulse / tau))
            out = out + term
        return out

    def residuals(x):
        return (model(x) - y) * weights

    try:
        result = least_squares(residuals, x0=x0, bounds=(lb, ub), method="trf", max_nfev=2000)
    except (RuntimeError, ValueError) as e:
        return {"success": False, "error": str(e), "n": n}

    x_opt = result.x
    components = [(float(x_opt[1 + 2 * i]), float(np.exp(x_opt[2 + 2 * i]))) for i in range(n)]
    y_pred = model(x_opt)
    return {
        "success": bool(result.success),
        "n": n,
        "a_dc": float(x_opt[0]),
        "components": components,
        "rms": float(np.sqrt(np.mean((y_pred - y) ** 2))),
    }


def multi_exp_fit_global(
    t: NDArray[np.floating],
    y: NDArray[np.floating],
    n_exponentials: int,
    rel_amp_threshold: float = 0.02,
    tau_proximity_factor: float = 1.5,
    verbose: bool = True,
    t_pulse_ns: Optional[float] = None,
) -> FitParameters:
    """Fit ``y(t)`` as DC plus decaying exponentials; return :class:`FitParameters`.

    Models
    ------
    * Step (``t_pulse_ns is None``): ``y = a_dc + Σ a_i exp(-t/tau_i)``
      [Hellings et al. arXiv:2503.04610 Eq. (1)].
    * Finite pulse: each term also has ``(1 - exp(-T_pulse/tau_i))``
      [Aggarwal et al. arXiv:2503.08645 App. H Eq. (H1)]. Returned ``a_i`` are
      de-attenuated, so IIR taps are ``A_i = a_i / a_dc`` in both modes.

    Starts at ``n_exponentials`` and retries with fewer components when the
    solver fails, two taus are within ``tau_proximity_factor``, or an amplitude
    is below ``rel_amp_threshold · swing``.
    """
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
        a_dc = float(res["a_dc"])

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
                return FitParameters(
                    success=False,
                    n_components_requested=n_req,
                    n_components_used=0,
                    a_tau_tuple=[],
                    a_dc=a_dc,
                    rms_error=float(res["rms"]),
                )
            n_cur = len(keep)
            continue

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

        comps_sorted = [(float(a), float(tau)) for a, tau in sorted(comps, key=lambda c: -c[1])]
        return FitParameters(
            success=True,
            n_components_requested=n_req,
            n_components_used=len(comps_sorted),
            a_tau_tuple=comps_sorted,
            a_dc=a_dc,
            rms_error=float(res["rms"]),
        )

    return FitParameters(
        success=False,
        n_components_requested=n_req,
        n_components_used=0,
        a_tau_tuple=[],
        a_dc=float("nan"),
        rms_error=float("nan"),
    )
