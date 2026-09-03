"""FIR filter utilities for cryoscope flux-line predistortion.

Shared by qubit and coupler short-time distortion calibrations
(``qubit_flux_short_distortion`` / ``coupler_flux_short_distortion``).

Provides forward FIR extraction via regularised Toeplitz inversion,
inverse-FIR computation for predistortion, and diagnostic plotting.

Adapted from QM's cryoscope analysis tooling.

The forward and inverse FIR predistortion filters are obtained by regularized
optimization following Hellings et al. arXiv:2503.04610, Sec. IV and Appendix I:
the forward FIR fit (Tikhonov + exponentially-weighted trailing-tap regularizer)
implements Eq. (I1); the inverse FIR (Gaussian-delta target, sigma = 0.75 ns,
with a derivative/Sobolev smoothness regularizer) implements Eq. (I2).
NOTE: the data-driven L-curve / GCV regularization-strength selection
(Castellanos triangle method) is an in-house addition beyond Hellings.
"""

from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def conv_causal(v: ArrayLike, h: ArrayLike, N: Optional[int] = None) -> np.ndarray:
    """Causal (one-sided) convolution of *v* with filter *h*.

    Parameters
    ----------
    v : array-like
        Input sequence.
    h : array-like
        Impulse response (filter coefficients).
    N : int or None
        Output length.  Defaults to ``len(v)``.

    Returns
    -------
    np.ndarray
        Convolution truncated to *N* samples.
    """
    v = np.asarray(v, dtype=float)
    h = np.asarray(h, dtype=float)
    y = np.convolve(v, h, mode="full")
    return y[: (len(v) if N is None else N)]


def build_toeplitz_matrix(v: ArrayLike, L: int) -> np.ndarray:
    """Toeplitz matrix from *v* for FIR system identification.

    Returns shape ``(len(v), L)`` so that ``phi ≈ V @ h``.
    """
    v = np.asarray(v, float)
    return linalg.toeplitz(c=v, r=np.concatenate([[v[0]], np.zeros(L - 1)]))


def resample_to_target_rate(
    data: ArrayLike,
    original_Ts: float,
    target_Ts: float,
    kind: str = "cubic",
    t_original_ns: Optional[ArrayLike] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample time-domain data to a finer uniform grid in nanoseconds.

    When ``t_original_ns`` is omitted, sample times default to
    ``np.arange(len(data)) * original_Ts`` (legacy behaviour).

    Cryoscope pipelines should pass the actual measurement time axis (typically
    ``1 .. N`` ns from ``ds_fit.time``) so upsampled values align with the same
    physical axis used for the 1 GS/s trace.

    Parameters
    ----------
    data : array-like
        Original time-domain samples.
    original_Ts, target_Ts : float
        Sampling period of *data* when times are implicit (ns), and target
        period for the output grid (ns).
    kind : str
        Interpolation kind passed to :func:`scipy.interpolate.interp1d`.
    t_original_ns : array-like, optional
        Explicit time coordinate (ns) for each sample in *data*. Must match
        ``len(data)``.

    Returns
    -------
    y_target : np.ndarray
        Interpolated samples on ``t_target``.
    t_target : np.ndarray
        Uniform grid from first to last original time with spacing ``target_Ts``.
    """
    data = np.asarray(data)
    if t_original_ns is not None:
        t_original = np.asarray(t_original_ns, dtype=float).reshape(-1)
        if len(t_original) != len(data):
            raise ValueError(
                f"t_original_ns length {len(t_original)} != len(data) {len(data)}"
            )
    else:
        t_original = np.arange(len(data), dtype=float) * float(original_Ts)

    t_start = float(t_original[0])
    t_end = float(t_original[-1])
    span = t_end - t_start
    if span < 0:
        raise ValueError("t_original_ns must be sorted non-decreasing")

    dt = float(target_Ts)
    if dt <= 0:
        raise ValueError("target_Ts must be positive")

    n_steps = int(np.floor(span / dt)) + 1
    t_target = t_start + np.arange(n_steps, dtype=float) * dt
    t_target = t_target[t_target <= t_end + 1e-9]

    interp_fun = interp1d(t_original, data, kind=kind, fill_value="extrapolate", bounds_error=False)
    y_target = np.asarray(interp_fun(t_target), dtype=float)
    return y_target, t_target


# ---------------------------------------------------------------------------
# Forward FIR fitting
# ---------------------------------------------------------------------------

def fit_fir(
    phi: ArrayLike,
    v: ArrayLike,
    L: int,
    Ts: float = 0.5,
    lam1: float = 1e-2,
    lam2: float = 1e-2,
    tail_ns: Optional[float] = None,
) -> np.ndarray:
    """Fit a forward FIR filter of length *L* via regularised least-squares.

    Solves ``phi ≈ V @ h`` where *V* is the Toeplitz matrix of *v*.

    Parameters
    ----------
    phi : array-like
        Measured (distorted) response.
    v : array-like
        Ideal stimulus (typically a step).
    L : int
        Number of FIR taps.
    Ts : float
        Sampling period (ns).
    lam1 : float
        L2 (Tikhonov) regularisation weight.
    lam2 : float
        Exponential-tail suppression weight.
    tail_ns : float or None
        Decay constant for tail regularisation.  Defaults to ``L*Ts/3``.

    Returns
    -------
    np.ndarray
        FIR coefficients of length *L*.
    """
    phi = np.asarray(phi, float)
    v = np.asarray(v, float)
    V = build_toeplitz_matrix(v, L)

    if tail_ns is None:
        tail_ns = (L * Ts) / 3.0
    idx = np.arange(L)
    x = np.exp(idx * Ts / tail_ns)

    A = V.T @ V + lam1 * np.eye(L) + lam2 * np.diag(x)
    b = V.T @ phi
    return linalg.solve(A, b, assume_a="pos")


def optimize_fir_parameters(
    response: ArrayLike,
    Ts: float = 0.5,
    L_values: Optional[List[int]] = None,
    lam1_values: Optional[List[float]] = None,
    lam2_values: Optional[List[float]] = None,
    verbose: bool = True,
) -> Tuple[list, float, dict, np.ndarray, np.ndarray]:
    """Grid-search over ``(L, lam1, lam2)`` to minimise reconstruction NRMS.

    Returns
    -------
    results : list[dict]
        All tested combinations.
    best_error : float
        Minimum NRMS error.
    best_params : dict
        Best ``{L, lam1, lam2, error, h, reconstructed}``.
    best_h : np.ndarray
        Best forward FIR.
    best_reconstructed : np.ndarray
        Reconstructed signal from best FIR.
    """
    if L_values is None:
        L_values = [16, 20, 24, 28, 32, 40, 48]
    if lam1_values is None:
        lam1_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    if lam2_values is None:
        lam2_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    if verbose:
        total = len(L_values) * len(lam1_values) * len(lam2_values)
        print(f"FIR grid search: {total} combinations  (signal {len(response)} samples, Ts={Ts} ns)")

    ideal = np.ones(len(response))
    results: list = []
    best_error = float("inf")
    best_params = None
    best_h = None
    best_reconstructed = None

    for L in L_values:
        for lam1 in lam1_values:
            for lam2 in lam2_values:
                try:
                    h = fit_fir(response, ideal, L=L, Ts=Ts, lam1=lam1, lam2=lam2)
                    h /= np.sum(h)
                    V = build_toeplitz_matrix(ideal, L)
                    reconstructed = V @ h
                    err = np.linalg.norm(response - reconstructed) / np.linalg.norm(response)
                    result = {"L": L, "lam1": lam1, "lam2": lam2, "error": err, "h": h.copy(), "reconstructed": reconstructed.copy()}
                    results.append(result)
                    if err < best_error:
                        best_error = err
                        best_params = result
                        best_h = h.copy()
                        best_reconstructed = reconstructed.copy()
                except Exception as exc:  # noqa: BLE001
                    if verbose:
                        print(f"  FIR grid: L={L} lam1={lam1:.0e} lam2={lam2:.0e} failed: {exc}")

    if verbose and best_params is not None:
        print(f"  Best: L={best_params['L']}, lam1={best_params['lam1']:.0e}, lam2={best_params['lam2']:.0e}, NRMS={best_error:.4e}")

    return results, best_error, best_params, best_h, best_reconstructed


# ---------------------------------------------------------------------------
# Inverse FIR
# ---------------------------------------------------------------------------

def invert_fir(
    h: ArrayLike,
    Ts: float = 0.5,
    M: Optional[int] = None,
    method: Literal["optimization", "analytical"] = "optimization",
    sigma_ns: float = 0.75,
    lam_smooth: float = 5e-2,
    normalize_dc_gain: bool = False,
) -> np.ndarray:
    """Compute an approximate causal inverse of FIR *h*.

    Parameters
    ----------
    h : array-like
        Forward FIR coefficients.
    Ts : float
        Sampling period (ns).
    M : int or None
        Length of the inverse FIR.  Defaults to ``len(h)``.
    method : ``"optimization"`` | ``"analytical"``
        Inversion strategy.
    sigma_ns : float
        Gaussian sigma for the target delta approximation (optimisation method).
    lam_smooth : float
        Smoothing regularisation weight (optimisation method).
    normalize_dc_gain : bool
        Force composite DC gain to unity.

    Returns
    -------
    np.ndarray
        Inverse FIR coefficients of length *M*.
    """
    h = np.asarray(h, float)
    L = len(h)
    if M is None:
        M = L

    if method == "optimization":
        t = np.arange(M) * Ts
        d = np.exp(-0.5 * (t / sigma_ns) ** 2)
        d /= d.sum()

        h_padded = np.pad(h, (0, max(0, M - L))) if M > L else h
        H = build_toeplitz_matrix(h_padded, M)[:M, :]

        D = np.eye(M, k=0) - np.eye(M, k=1)
        D = D[:-1, :]

        A = H.T @ H + lam_smooth * (D.T @ D)
        b = H.T @ d
        h_inv = linalg.solve(A, b, assume_a="pos")

        if normalize_dc_gain:
            gain = h.sum() * h_inv.sum()
            if gain != 0:
                h_inv /= gain
    else:
        h_inv = np.zeros(M)
        h_inv[0] = 1 / h[0]
        for m in range(1, min(L, M)):
            s = sum(h_inv[m - i] * h[i] for i in range(1, min(m + 1, L)))
            h_inv[m] = -s / h[0]

    return h_inv


# ---------------------------------------------------------------------------
# High-level analysis + plotting
# ---------------------------------------------------------------------------

def analyze_and_plot_fir_fit(
    response: ArrayLike,
    time: ArrayLike,
    Ts: float = 0.5,
    L_values: Optional[List[int]] = None,
    lam1_values: Optional[List[float]] = None,
    lam2_values: Optional[List[float]] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, dict, np.ndarray, float, plt.Figure]:
    """Run forward-FIR grid search and produce a 2x2 diagnostic figure.

    Returns
    -------
    best_h, best_params, best_reconstructed, best_error, fig
    """
    results, best_error, best_params, best_h, best_reconstructed = optimize_fir_parameters(
        response, Ts=Ts, L_values=L_values, lam1_values=lam1_values, lam2_values=lam2_values, verbose=verbose,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(time, response, "r-", lw=2, alpha=0.7, label="Measured (distorted)")
    ax.plot(time, best_reconstructed, "b--", lw=2, label=f"Reconstructed (NRMS={best_error:.4e})")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Best Reconstruction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    residual = response - best_reconstructed
    ax.plot(time, residual, "m-", lw=1.5)
    ax.axhline(0, color="k", ls="--", alpha=0.3)
    ax.fill_between(time, -np.std(residual), np.std(residual), alpha=0.2, color="gray")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Residual")
    ax.set_title(f"Reconstruction Residual (σ={np.std(residual):.4e})")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(best_h, "b-o", ms=4, lw=2)
    ax.set_xlabel("Tap Index")
    ax.set_ylabel("Coefficient")
    ax.set_title(f"Forward FIR (L={best_params['L']})")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    errors_by_L: dict = {}
    for r in results:
        errors_by_L.setdefault(r["L"], []).append(r["error"])
    L_sorted = sorted(errors_by_L)
    means = [np.mean(errors_by_L[L]) for L in L_sorted]
    stds = [np.std(errors_by_L[L]) for L in L_sorted]
    ax.errorbar(L_sorted, means, yerr=stds, fmt="o-", capsize=5, lw=2, ms=6)
    ax.axhline(best_error, color="r", ls="--", alpha=0.5, label=f"Best: {best_error:.4e}")
    ax.set_xlabel("Filter Length L")
    ax.set_ylabel("Reconstruction Error")
    ax.set_title("Error vs Filter Length")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return best_h, best_params, best_reconstructed, best_error, fig


def analyze_and_plot_inverse_fir(
    response: ArrayLike,
    time: ArrayLike,
    Ts: float = 0.5,
    L_values: Optional[List[int]] = None,
    lam1_values: Optional[List[float]] = None,
    lam2_values: Optional[List[float]] = None,
    M: Optional[int] = None,
    sigma_ns: float = 0.75,
    lam_smooth: float = 5e-2,
    method: Literal["optimization", "analytical"] = "optimization",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, plt.Figure, plt.Figure, dict]:
    """Full FIR analysis: forward fit + inversion + correction simulation.

    Returns
    -------
    best_h : np.ndarray
        Forward FIR.
    h_inv : np.ndarray
        Inverse (predistortion) FIR.
    best_reconstructed : np.ndarray
        Signal reconstructed by the forward FIR model.
    fig_fir_fit : matplotlib.figure.Figure
        2x2 forward-FIR diagnostic.
    fig_inverse : matplotlib.figure.Figure
        3x2 inverse-FIR & correction diagnostic.
    """
    best_h, best_params, best_reconstructed, best_error, fig_fir_fit = analyze_and_plot_fir_fit(
        response=response, time=time, Ts=Ts,
        L_values=L_values, lam1_values=lam1_values, lam2_values=lam2_values,
        verbose=verbose,
    )

    h_inv = invert_fir(h=best_h, Ts=Ts, M=M, method=method, sigma_ns=sigma_ns, lam_smooth=lam_smooth)

    delta = conv_causal(best_h, h_inv, N=len(best_h))
    ideal_response = np.ones(len(response))

    L_guard = len(h_inv)
    guard = np.zeros(L_guard)
    ideal_padded = np.concatenate([guard, ideal_response, guard])
    predistorted_padded = conv_causal(ideal_padded, h_inv)
    start = L_guard
    end = start + len(ideal_response)
    predistorted_response = predistorted_padded[start:end]
    corrected_response = conv_causal(predistorted_response, best_h, N=len(ideal_response))
    correction_error = np.linalg.norm(corrected_response - ideal_response) / np.linalg.norm(ideal_response)

    distorted_padded = np.concatenate([guard, response, guard])
    corrected_from_measured = conv_causal(distorted_padded, h_inv)[start:end]
    correction_error_meas = np.linalg.norm(corrected_from_measured - ideal_response) / np.linalg.norm(ideal_response)

    if verbose:
        print(f"  Correction NRMS (predistort): {correction_error:.3e}")
        print(f"  Correction NRMS (measured):   {correction_error_meas:.3e}")

    # --- 3x2 inverse FIR diagnostic figure ---
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    ax = axes[0, 0]
    ax.plot(time, ideal_response, "g--", lw=2, alpha=0.7, label="Ideal")
    ax.plot(time, response, "r-", lw=2, label="Distorted")
    ax.plot(time, best_reconstructed, "b:", lw=2, alpha=0.7, label="FIR model")
    ax.axhline(1.001, color="gray", ls="--", lw=1, alpha=0.7)
    ax.axhline(0.999, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set_ylim([0.95, 1.05])
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Signals and FIR Prediction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(best_h, "b-o", ms=4, lw=2, label="Forward h")
    ax.plot(h_inv, "r-s", ms=4, lw=2, label="Inverse h_inv")
    ax.set_xlabel("Tap Index")
    ax.set_ylabel("Coefficient")
    ax.set_title("FIR Filters")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(time, ideal_response, "g--", lw=2, alpha=0.7, label="Ideal")
    ax.plot(time, predistorted_response, "c-", lw=2, alpha=0.7, label="Predistorted")
    ax.plot(time, corrected_response, "m-", lw=2, label="Corrected (sim)")
    ax.axhline(1.001, color="gray", ls="--", lw=1, alpha=0.7)
    ax.axhline(0.999, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set_ylim([0.95, 1.05])
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Predistortion and Correction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(time, ideal_response, "g--", lw=2, alpha=0.7, label="Ideal")
    ax.plot(time, response, "r-", lw=2, alpha=0.5, label="Distorted")
    ax.plot(time, corrected_response, "m-", lw=2, label="Corrected (predistort)")
    ax.plot(time, corrected_from_measured, color="orange", lw=2, label="Corrected (measured)")
    ax.axhline(1.001, color="gray", ls="--", lw=1, alpha=0.7)
    ax.axhline(0.999, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set_ylim([0.95, 1.05])
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Correction Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    res_fit = response - best_reconstructed
    res_corr = corrected_response - ideal_response
    ax.plot(time, res_fit, "b-", lw=1.5, label=f"Fit residual (σ={np.std(res_fit):.4e})")
    ax.plot(time, res_corr, "m-", lw=1.5, label=f"Correction residual (σ={np.std(res_corr):.4e})")
    ax.axhline(0, color="k", ls="--", alpha=0.3)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Residual")
    ax.set_title("Residual Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(np.arange(len(delta)) * Ts, delta, "g-o", ms=4, lw=2)
    ax.axhline(0, color="k", ls="--", alpha=0.3)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"h * h_inv ≈ δ  (peak={np.max(np.abs(delta)):.3e})")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return best_h, h_inv, best_reconstructed, fig_fir_fit, fig, best_params


# ---------------------------------------------------------------------------
# Auto-tuning helpers (GCV / L-curve based)
# ---------------------------------------------------------------------------
#
# Forward FIR fit reformulated to allow data-driven choice of the
# regularisation strength.  The original ``fit_fir`` exposes two penalties
# (``lam1`` for Tikhonov, ``lam2`` for an exponential-tail prior).  Here we
# collapse them into a single scalar ``lam`` by writing the regularisation
# matrix as ``lam * (I + alpha * diag(x))`` where ``x_i = exp(i*Ts/tail_ns)``
# (the same tail kernel as ``fit_fir``).  ``alpha`` defaults to 1.0 which
# matches the original ``lam1 == lam2`` setting and keeps the auto path a true
# generalisation of the grid path.
#
# Key trick: we transform variables via ``h = (1/sqrt(P_diag)) * g`` so the
# problem becomes a *standard* ridge regression on ``V_tilde = V / sqrt(P)``.
# A single economy SVD of ``V_tilde`` then lets us evaluate the fit at *any*
# ``lam`` in O(L) — making both GCV (1-D continuous search) and L-curve
# (logarithmic grid + corner detection) cheap enough to run inside a single
# node analysis pass without blowing the timing budget.


def _fit_fir_setup(
    phi: ArrayLike,
    v: ArrayLike,
    L: int,
    Ts: float,
    alpha: float = 1.0,
    tail_ns: Optional[float] = None,
) -> Dict:
    """Precompute SVD-based factors so that fits at any ``lam`` are O(L).

    Returns a dict with cached arrays consumed by :func:`_eval_at_lambda`.
    """
    phi = np.asarray(phi, float)
    v = np.asarray(v, float)
    V = build_toeplitz_matrix(v, L)

    if tail_ns is None:
        tail_ns = (L * Ts) / 3.0
    idx = np.arange(L)
    x_tail = np.exp(idx * Ts / tail_ns)
    p_diag = 1.0 + alpha * x_tail
    s = np.sqrt(p_diag)

    V_tilde = V / s[np.newaxis, :]
    U, sigma, Wt = linalg.svd(V_tilde, full_matrices=False)
    Uty = U.T @ phi

    return {
        "V": V,
        "U": U,
        "sigma": sigma,
        "Wt": Wt,
        "Uty": Uty,
        "s": s,
        "N": len(phi),
        "y_norm_sq": float(phi @ phi),
        "phi": phi,
        "alpha": float(alpha),
        "L": int(L),
        "Ts": float(Ts),
    }


def _eval_at_lambda(setup: Dict, lam: float) -> Tuple[np.ndarray, float, float]:
    """Compute (h, rss, hat_trace) at ``lam`` using the cached SVD.

    ``hat_trace`` is the effective degrees of freedom
    ``tr(V (V'V + lam P)^-1 V')`` and equals ``sum_i sigma_i^2/(sigma_i^2+lam)``
    in the transformed (ridge) coordinates.
    """
    sigma = setup["sigma"]
    Uty = setup["Uty"]
    Wt = setup["Wt"]
    s = setup["s"]

    sig2 = sigma * sigma
    denom = sig2 + lam
    g = Wt.T @ (sigma * Uty / denom)
    h = g / s

    pred_proj = (sig2 / denom) * Uty
    res_proj = Uty - pred_proj
    rss_in_range = float(res_proj @ res_proj)
    rss_orthogonal = setup["y_norm_sq"] - float(Uty @ Uty)
    rss = rss_in_range + max(rss_orthogonal, 0.0)

    hat_trace = float((sig2 / denom).sum())
    return h, rss, hat_trace


def _gcv_score(rss: float, hat_trace: float, N: int) -> float:
    """Standard Golub-Heath-Wahba GCV score (lower is better)."""
    denom = N - hat_trace
    if denom <= 1e-12:
        return np.inf
    return (rss / N) / ((denom / N) ** 2)


def _lcurve_corner_idx(log_res: np.ndarray, log_sol: np.ndarray) -> int:
    """Triangle-method (Castellanos et al.) corner detection on the L-curve.

    Returns the index of maximum curvature.  Robust for ~10-20 point curves.
    """
    n = len(log_res)
    if n < 4:
        return n // 2
    p = np.column_stack([log_res, log_sol])
    best_kappa = -np.inf
    best_i = n // 2
    for i in range(1, n - 1):
        a = p[i] - p[i - 1]
        b = p[i + 1] - p[i]
        c = p[i + 1] - p[i - 1]
        area = abs(a[0] * b[1] - a[1] * b[0])
        denom = (np.linalg.norm(a) * np.linalg.norm(b) * np.linalg.norm(c))
        if denom == 0:
            continue
        kappa = 2.0 * area / denom
        if kappa > best_kappa:
            best_kappa = kappa
            best_i = i
    return best_i


def _normalize_and_score(
    h: np.ndarray,
    setup: Dict,
    response: np.ndarray,
    hat_trace: float,
) -> Dict:
    """L1-normalise ``h`` (DC gain = 1) and recompute NRMS, AIC on raw signal."""
    sum_h = float(h.sum())
    if sum_h != 0:
        h_norm = h / sum_h
    else:
        h_norm = h.copy()
    y_pred = setup["V"] @ h_norm
    diff = response - y_pred
    rss = float(diff @ diff)
    nrms = float(np.linalg.norm(diff) / max(np.linalg.norm(response), 1e-12))
    N = setup["N"]
    aic = N * np.log(rss / N + 1e-30) + 2.0 * hat_trace
    return {"h": h_norm, "rss": rss, "nrms": nrms, "aic": aic, "hat_trace": hat_trace}


# ---------------------------------------------------------------------------
# Auto forward fit
# ---------------------------------------------------------------------------

def auto_fit_fir(
    response: ArrayLike,
    ideal: Optional[ArrayLike] = None,
    Ts: float = 0.5,
    max_taps: int = 48,
    min_taps: int = 8,
    L_step: int = 4,
    criterion: Literal["gcv", "lcurve", "both"] = "both",
    alpha: float = 1.0,
    lcurve_points: int = 16,
    gcv_xatol: float = 0.15,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """IIR-style auto-tuned forward FIR.

    The user specifies only ``max_taps`` (and the auto criterion).  Internally:
      1. Sweep ``L`` over ``range(min_taps, max_taps+1, L_step)``.
      2. For each L, find the optimal regularisation strength ``lam`` with
         GCV (1-D bounded minimisation on log-lambda) and / or L-curve
         (corner detection on a logarithmic grid).
      3. Select the final L (per criterion) by AIC, which trades reconstruction
         residual against effective degrees-of-freedom — preventing the
         "always pick the largest L" pathology of pure NRMS minimisation.
      4. When ``criterion="both"`` GCV and L-curve are run independently and
         the criterion with the lower AIC wins (full search history is kept
         in the returned info dict).

    Parameters
    ----------
    response : array-like
        Measured (distorted) step response.
    ideal : array-like, optional
        Stimulus.  Defaults to a unit step ``np.ones_like(response)``.
    Ts : float
        Sampling period (ns).
    max_taps : int
        Upper bound on the forward filter length L.
    min_taps : int
        Lower bound on L (default 8).
    L_step : int
        Step size for the L sweep.
    criterion : {"gcv", "lcurve", "both"}
        Lambda-selection rule.
    alpha : float
        Ratio of exponential-tail penalty to base Tikhonov penalty.
        ``alpha=1`` reproduces the ``lam1==lam2`` setting of the original grid.
    lcurve_points : int
        Number of log-spaced lambdas evaluated when computing the L-curve.
    gcv_xatol : float
        Absolute tolerance for the 1-D GCV search in log-lambda (default 0.15
        ≈ 16% relative resolution in lambda).
    verbose : bool
        Print per-L diagnostic line.

    Returns
    -------
    h : np.ndarray
        Best forward FIR (L1-normalised so DC gain = 1).
    info : dict
        ``L``, ``lam``, ``alpha``, ``criterion_used``, ``nrms``, ``aic``,
        ``hat_trace``, ``all_L_results`` (per-criterion per-L records).
    """
    response = np.asarray(response, float)
    if ideal is None:
        ideal = np.ones_like(response)
    else:
        ideal = np.asarray(ideal, float)

    L_candidates = list(range(int(min_taps), int(max_taps) + 1, int(L_step)))
    if not L_candidates:
        raise ValueError(f"empty L sweep: min={min_taps}, max={max_taps}, step={L_step}")

    crits_to_run = ["gcv", "lcurve"] if criterion == "both" else [criterion]
    per_crit_records: Dict[str, List[Dict]] = {c: [] for c in crits_to_run}

    for L in L_candidates:
        setup = _fit_fir_setup(response, ideal, L, Ts, alpha=alpha)
        sigma = setup["sigma"]
        sig_max = float(sigma.max()) if sigma.size else 1.0
        log_lam_lo = float(np.log(max(sig_max * sig_max * 1e-8, 1e-15)))
        log_lam_hi = float(np.log(max(sig_max * sig_max * 1e2, 1e-12)))

        # --- GCV ---
        if "gcv" in crits_to_run:
            def _obj_gcv(log_lam, _setup=setup):
                _h, _rss, _ht = _eval_at_lambda(_setup, float(np.exp(log_lam)))
                return _gcv_score(_rss, _ht, _setup["N"])
            res = minimize_scalar(
                _obj_gcv,
                bounds=(log_lam_lo, log_lam_hi),
                method="bounded",
                options={"xatol": gcv_xatol},
            )
            lam = float(np.exp(res.x))
            h_raw, _rss_raw, ht = _eval_at_lambda(setup, lam)
            scored = _normalize_and_score(h_raw, setup, response, ht)
            rec = {"L": L, "lam": lam, "alpha": alpha, **scored}
            per_crit_records["gcv"].append(rec)
            if verbose:
                print(f"  [gcv   ] L={L:3d}  lam={lam:.2e}  NRMS={rec['nrms']:.4e}  d_eff={ht:.2f}  AIC={rec['aic']:.2f}")

        # --- L-curve ---
        if "lcurve" in crits_to_run:
            lam_grid = np.exp(np.linspace(log_lam_lo, log_lam_hi, int(lcurve_points)))
            log_res_pts = np.empty(len(lam_grid))
            log_sol_pts = np.empty(len(lam_grid))
            for k, lam in enumerate(lam_grid):
                _h, _rss, _ = _eval_at_lambda(setup, float(lam))
                log_res_pts[k] = 0.5 * np.log(max(_rss, 1e-30))
                log_sol_pts[k] = 0.5 * np.log(max(float(_h @ _h), 1e-30))
            idx = _lcurve_corner_idx(log_res_pts, log_sol_pts)
            lam = float(lam_grid[idx])
            h_raw, _rss_raw, ht = _eval_at_lambda(setup, lam)
            scored = _normalize_and_score(h_raw, setup, response, ht)
            rec = {"L": L, "lam": lam, "alpha": alpha, **scored,
                   "lcurve_log_res": log_res_pts.tolist(),
                   "lcurve_log_sol": log_sol_pts.tolist(),
                   "lcurve_lam_grid": lam_grid.tolist(),
                   "lcurve_corner_idx": int(idx)}
            per_crit_records["lcurve"].append(rec)
            if verbose:
                print(f"  [lcurve] L={L:3d}  lam={lam:.2e}  NRMS={rec['nrms']:.4e}  d_eff={ht:.2f}  AIC={rec['aic']:.2f}")

    best_per_crit = {c: min(recs, key=lambda r: r["aic"]) for c, recs in per_crit_records.items()}

    if criterion == "both":
        used = min(best_per_crit.keys(), key=lambda c: best_per_crit[c]["aic"])
    else:
        used = criterion
    chosen = best_per_crit[used]

    info = {
        "L": chosen["L"],
        "lam": chosen["lam"],
        "alpha": chosen["alpha"],
        "lam1": chosen["lam"],
        "lam2": chosen["lam"] * chosen["alpha"],
        "criterion_used": used,
        "nrms": chosen["nrms"],
        "rss": chosen["rss"],
        "aic": chosen["aic"],
        "hat_trace": chosen["hat_trace"],
        "all_L_results": per_crit_records,
        "best_per_criterion": best_per_crit,
        "L_candidates": L_candidates,
    }
    return chosen["h"], info


# ---------------------------------------------------------------------------
# Auto inverse FIR
# ---------------------------------------------------------------------------

def auto_invert_fir(
    h: ArrayLike,
    Ts: float = 0.5,
    M: Optional[int] = None,
    sigma_ns: Optional[float] = None,
    criterion: Literal["gcv", "lcurve", "both"] = "both",
    lcurve_points: int = 16,
    gcv_xatol: float = 0.15,
    normalize_dc_gain: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """Auto-tune the smoothness regulariser of the inverse FIR.

    The optimisation-method inverse FIR (see :func:`invert_fir`) minimises
    ``||H h_inv - d||^2 + lam_smooth * ||D h_inv||^2`` where ``d`` is a
    Gaussian approximation to ``delta`` and ``D`` is the first-difference
    operator.  ``lam_smooth`` is chosen data-driven (GCV and/or L-curve).
    The ``sigma_ns`` default of ``1.3 * Ts`` is set by the cryoscope
    measurement Nyquist: cryoscope sweeps in 1 ns steps, so any spectral
    content above 500 MHz is unresolved.  A tighter target ``sigma`` would
    only amplify noise in those unresolvable bands.  Since the baking grid
    is also 1 ns, sub-ns features are physically absent from both the
    stimulus and the response, so this default needs no per-experiment
    tuning.

    Parameters
    ----------
    h : array-like
        Forward FIR coefficients.
    Ts : float
        Sampling period (ns).
    M : int, optional
        Length of the inverse FIR.  Defaults to ``len(h)``.
    sigma_ns : float, optional
        Gaussian sigma for the target delta.  Defaults to ``1.3 * Ts``.
    criterion : {"gcv", "lcurve", "both"}
        Lambda-selection rule for ``lam_smooth``.
    lcurve_points, gcv_xatol : as in :func:`auto_fit_fir`.
    normalize_dc_gain : bool
        If True, rescale so composite DC gain ``sum(h)*sum(h_inv) == 1``.

    Returns
    -------
    h_inv : np.ndarray
        Inverse FIR coefficients.
    info : dict
        ``lam_smooth``, ``sigma_ns``, ``criterion_used``, ``rss``,
        ``hat_trace``, ``all_results``.
    """
    h = np.asarray(h, float)
    L = len(h)
    if M is None:
        M = L
    if sigma_ns is None:
        sigma_ns = 1.3 * Ts

    t = np.arange(M) * Ts
    d = np.exp(-0.5 * (t / sigma_ns) ** 2)
    d = d / d.sum()

    h_padded = np.pad(h, (0, max(0, M - L))) if M > L else h
    H = build_toeplitz_matrix(h_padded, M)[:M, :]

    D = np.eye(M, k=0) - np.eye(M, k=1)
    D = D[:-1, :]
    DtD = D.T @ D

    # Use joint diagonalisation: D'D is symmetric positive semi-definite;
    # eigendecompose it once so per-lambda evaluation is O(M^2) instead of
    # O(M^3).  Then the problem reduces to ridge regression in the
    # eigenbasis of D'D after whitening through H.  Practically, since M is
    # small (~48) we just solve the M-by-M system at each lambda; total cost
    # for ~30 GCV evals stays well under 5 ms.
    HtH = H.T @ H
    Htd = H.T @ d
    N_eff = M  # the "data points" of the inverse problem

    def _eval(lam: float) -> Tuple[np.ndarray, float, float]:
        A = HtH + lam * DtD
        h_inv = linalg.solve(A, Htd, assume_a="pos")
        y_pred = H @ h_inv
        rss = float(np.sum((d - y_pred) ** 2))
        Ainv_HtH = linalg.solve(A, HtH, assume_a="pos")
        hat_trace = float(np.trace(Ainv_HtH))
        return h_inv, rss, hat_trace

    log_lam_lo, log_lam_hi = np.log(1e-6), np.log(1e2)
    crits_to_run = ["gcv", "lcurve"] if criterion == "both" else [criterion]
    results_per_crit: Dict[str, Dict] = {}

    if "gcv" in crits_to_run:
        def _obj_gcv(log_lam):
            _h, _rss, _ht = _eval(float(np.exp(log_lam)))
            return _gcv_score(_rss, _ht, N_eff)
        res = minimize_scalar(
            _obj_gcv,
            bounds=(log_lam_lo, log_lam_hi),
            method="bounded",
            options={"xatol": gcv_xatol},
        )
        lam = float(np.exp(res.x))
        h_inv_g, rss_g, ht_g = _eval(lam)
        aic = N_eff * np.log(rss_g / N_eff + 1e-30) + 2.0 * ht_g
        results_per_crit["gcv"] = {
            "lam_smooth": lam, "h_inv": h_inv_g,
            "rss": rss_g, "hat_trace": ht_g, "aic": aic,
        }
        if verbose:
            print(f"  inv [gcv   ] lam_smooth={lam:.2e}  RSS={rss_g:.4e}  d_eff={ht_g:.2f}  AIC={aic:.2f}")

    if "lcurve" in crits_to_run:
        lam_grid = np.exp(np.linspace(log_lam_lo, log_lam_hi, int(lcurve_points)))
        log_res_pts = np.empty(len(lam_grid))
        log_sol_pts = np.empty(len(lam_grid))
        cached = []
        for k, lam in enumerate(lam_grid):
            _h, _rss, _ht = _eval(float(lam))
            cached.append((_h, _rss, _ht))
            log_res_pts[k] = 0.5 * np.log(max(_rss, 1e-30))
            sol_norm_sq = float((D @ _h) @ (D @ _h))
            log_sol_pts[k] = 0.5 * np.log(max(sol_norm_sq, 1e-30))
        idx = _lcurve_corner_idx(log_res_pts, log_sol_pts)
        lam = float(lam_grid[idx])
        h_inv_l, rss_l, ht_l = cached[idx]
        aic = N_eff * np.log(rss_l / N_eff + 1e-30) + 2.0 * ht_l
        results_per_crit["lcurve"] = {
            "lam_smooth": lam, "h_inv": h_inv_l,
            "rss": rss_l, "hat_trace": ht_l, "aic": aic,
        }
        if verbose:
            print(f"  inv [lcurve] lam_smooth={lam:.2e}  RSS={rss_l:.4e}  d_eff={ht_l:.2f}  AIC={aic:.2f}")

    if criterion == "both":
        used = min(results_per_crit.keys(), key=lambda c: results_per_crit[c]["aic"])
    else:
        used = criterion
    chosen = results_per_crit[used]
    h_inv = chosen["h_inv"]

    if normalize_dc_gain:
        gain = h.sum() * h_inv.sum()
        if gain != 0:
            h_inv = h_inv / gain

    info = {
        "lam_smooth": chosen["lam_smooth"],
        "sigma_ns": float(sigma_ns),
        "criterion_used": used,
        "rss": chosen["rss"],
        "hat_trace": chosen["hat_trace"],
        "aic": chosen["aic"],
        "all_results": results_per_crit,
    }
    return h_inv, info


# ---------------------------------------------------------------------------
# Auto high-level plot + analysis
# ---------------------------------------------------------------------------

def _build_fir_fit_figure_auto(
    h: np.ndarray,
    response: np.ndarray,
    time: np.ndarray,
    info_fwd: Dict,
) -> plt.Figure:
    """Build the 2×2 forward-FIR diagnostic figure for the auto-tuned path.

    Replaces the singleton-grid replay used previously.  The bottom-right
    panel shows the GCV/L-curve optimal NRMS per L over the full search
    landscape, with a vertical marker at the AIC-chosen L.
    """
    ideal = np.ones_like(response)
    reconstructed = build_toeplitz_matrix(ideal, len(h)) @ h
    residual = response - reconstructed
    nrms = float(np.linalg.norm(residual) / max(np.linalg.norm(response), 1e-30))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(time, response, "r-", lw=2, alpha=0.7, label="Measured (distorted)")
    ax.plot(time, reconstructed, "b--", lw=2, label=f"Reconstructed (NRMS={nrms:.4e})")
    ax.set_xlabel("Time (ns)"); ax.set_ylabel("Amplitude")
    ax.set_title("Best Reconstruction"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(time, residual, "m-", lw=1.5)
    ax.axhline(0, color="k", ls="--", alpha=0.3)
    ax.fill_between(time, -np.std(residual), np.std(residual), alpha=0.2, color="gray")
    ax.set_xlabel("Time (ns)"); ax.set_ylabel("Residual")
    ax.set_title(f"Reconstruction Residual (σ={np.std(residual):.4e})")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(h, "b-o", ms=4, lw=2)
    ax.set_xlabel("Tap Index"); ax.set_ylabel("Coefficient")
    ax.set_title(f"Forward FIR (L={info_fwd['L']}, auto-chosen by AIC)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    _colors = {"gcv": "steelblue", "lcurve": "darkorange"}
    for crit, records in info_fwd["all_L_results"].items():
        Ls = [r["L"] for r in records]
        nrmss = [r["nrms"] for r in records]
        ax.plot(Ls, nrmss, "o-", lw=2, ms=6, label=crit, color=_colors.get(crit, None))
    ax.axvline(info_fwd["L"], color="r", ls="--", lw=1.5, alpha=0.7,
               label=f"Chosen L={info_fwd['L']} (AIC)")
    ax.axhline(nrms, color="r", ls=":", alpha=0.5, label=f"Best NRMS: {nrms:.4e}")
    ax.set_xlabel("Filter Length L"); ax.set_ylabel("NRMS (optimal λ per L)")
    ax.set_title("NRMS vs Filter Length"); ax.legend(); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def analyze_and_plot_inverse_fir_auto(
    response: ArrayLike,
    time: ArrayLike,
    Ts: float = 0.5,
    max_taps: int = 48,
    min_taps: int = 8,
    L_step: int = 4,
    M: Optional[int] = None,
    sigma_ns: Optional[float] = None,
    alpha: float = 1.0,
    criterion: Literal["gcv", "lcurve", "both"] = "both",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, plt.Figure, plt.Figure, Dict]:
    """Auto-tuned forward + inverse FIR with diagnostic plots.

    Drop-in replacement for :func:`analyze_and_plot_inverse_fir` returning the
    same tuple ``(h, h_inv, reconstructed, fig_fir_fit, fig_inverse, info)``.

    ``fig_fir_fit`` (2×2) is built from the full auto-search landscape so the
    "NRMS vs Filter Length" panel shows all L candidates, not just one point.
    ``fig_inverse`` (3×2 correction diagnostic) is generated by a single
    replay of the chosen (L, lam, lam_smooth) via the existing function.
    """
    response = np.asarray(response, float)
    time = np.asarray(time, float)

    h_fwd, info_fwd = auto_fit_fir(
        response=response,
        ideal=np.ones_like(response),
        Ts=Ts,
        max_taps=max_taps,
        min_taps=min_taps,
        L_step=L_step,
        criterion=criterion,
        alpha=alpha,
        verbose=verbose,
    )

    if M is None:
        M = max_taps
    if sigma_ns is None:
        sigma_ns = 1.3 * Ts

    h_inv, info_inv = auto_invert_fir(
        h_fwd, Ts=Ts, M=M, sigma_ns=sigma_ns, criterion=criterion, verbose=verbose,
    )

    fig_fir_fit = _build_fir_fit_figure_auto(h_fwd, response, time, info_fwd)

    # Replay with the chosen point to generate the 3×2 correction diagnostic.
    # The 2×2 fit figure produced internally is immediately closed — it is
    # replaced by fig_fir_fit built above with the full L-sweep landscape.
    _, h_inv_final, best_reconstructed, _ghost_fig, fig_inverse, replay_params = \
        analyze_and_plot_inverse_fir(
            response=response, time=time, Ts=Ts,
            L_values=[info_fwd["L"]],
            lam1_values=[info_fwd["lam1"]],
            lam2_values=[info_fwd["lam2"]],
            M=M, sigma_ns=info_inv["sigma_ns"], lam_smooth=info_inv["lam_smooth"],
            method="optimization", verbose=False,
        )
    plt.close(_ghost_fig)

    info = {
        "L": info_fwd["L"],
        "lam": info_fwd["lam"],
        "lam1": info_fwd["lam1"],
        "lam2": info_fwd["lam2"],
        "alpha": info_fwd["alpha"],
        "lam_smooth": info_inv["lam_smooth"],
        "sigma_ns": info_inv["sigma_ns"],
        "criterion_forward": info_fwd["criterion_used"],
        "criterion_inverse": info_inv["criterion_used"],
        "forward_nrms": info_fwd["nrms"],
        "forward_rss": info_fwd["rss"],
        "forward_aic": info_fwd["aic"],
        "forward_hat_trace": info_fwd["hat_trace"],
        "inverse_rss": info_inv["rss"],
        "inverse_hat_trace": info_inv["hat_trace"],
        "_replay_params": replay_params,
        "_forward_search": info_fwd,
        "_inverse_search": info_inv,
    }
    return h_fwd, h_inv_final, best_reconstructed, fig_fir_fit, fig_inverse, info


# ---------------------------------------------------------------------------
# Noise-floor triangulation
# ---------------------------------------------------------------------------

def estimate_noise_floor(
    signal: ArrayLike,
    Ts: float,
    tail_window_ns: float = 30.0,
    sigma_C: Optional[float] = None,
    warn_ratio: float = 1.5,
) -> Dict:
    """Triangulate measurement noise sigma from up-to-three independent estimators.

    Three estimators with progressively weaker assumptions:
      - ``sigma_A`` = std of the late-time tail (assumes tail is flat).
      - ``sigma_B`` = std(diff(tail)) / sqrt(2) (assumes tail is smooth only).
      - ``sigma_C`` = sqrt(rss / (N - tr(H))) from the chosen fit (caller
        passes it in; only available in auto mode where tr(H) is known).

    Run on the **raw measurement grid** (1 GS/s here).  At 2 GS/s the cubic
    interpolation correlates adjacent samples and biases ``sigma_B`` downward,
    producing spurious WARN flags.

    Two distinct failure modes are detected (ratio threshold = ``warn_ratio``):

    - ``"warn_tail"`` — ``sigma_A / sigma_B > warn_ratio``.  Tail itself
      still contains unresolved physics (slow drift inflates ``sigma_A`` but
      not ``sigma_B``).  Suggest extending ``cryoscope_len``.

    - ``"warn_fit"`` — tail is consistent (``sigma_A ~ sigma_B``) but
      ``sigma_C > warn_ratio * max(sigma_A, sigma_B)``.  The fit leaves
      systematic bias outside the tail (e.g., upstream IIR underfits a
      long-tau component that the short FIR window cannot absorb).
      Extending ``cryoscope_len`` does NOT help; raise ``n_exponentials`` or
      revisit the cryoscope amplitude.

    Returns
    -------
    dict
        Keys: ``sigma_A``, ``sigma_B``, ``sigma_C`` (or None), ``displayed``
        (= sigma_B), ``ratio_AB``, ``ratio_fit`` (= 1 if sigma_C unavailable),
        ``ratio_max_min`` (legacy), ``status``
        (``"ok"|"warn_tail"|"warn_fit"``), ``msg_short`` (legend label).
    """
    signal = np.asarray(signal, float)
    tail = signal[-max(int(round(tail_window_ns / Ts)), 4):]

    sigma_A = float(np.std(tail))
    sigma_B = float(np.std(np.diff(tail)) / np.sqrt(2.0))
    sigma_C_f = float(sigma_C) if (sigma_C is not None and np.isfinite(sigma_C)) else None

    sAB_max = max(sigma_A, sigma_B)
    sAB_min = max(min(sigma_A, sigma_B), 1e-30)
    ratio_AB = sAB_max / sAB_min
    ratio_fit = (sigma_C_f / sAB_max) if (sigma_C_f and sAB_max > 0) else 1.0

    if ratio_AB > warn_ratio:
        status, msg = "warn_tail", "WARN: extend cryoscope_len"
    elif ratio_fit > warn_ratio:
        status, msg = "warn_fit", "WARN: fit underfit"
    else:
        status, msg = "ok", "OK"

    all_vals = [sigma_A, sigma_B] + ([sigma_C_f] if sigma_C_f else [])
    positive = [s for s in all_vals if s > 0]
    ratio_max_min = max(all_vals) / min(positive) if positive else float("inf")

    return {
        "sigma_A": sigma_A,
        "sigma_B": sigma_B,
        "sigma_C": sigma_C_f,
        "displayed": sigma_B,
        "ratio_AB": float(ratio_AB),
        "ratio_fit": float(ratio_fit),
        "ratio_max_min": float(ratio_max_min),
        "status": status,
        "msg_short": msg,
    }
