"""Minimal FIR utilities for cryoscope flux-line predistortion (17c / short-time).

Context
-------
After the IIR exponential correction, residual *fast* flux-line distortion can
still remain. This module models that leftover as a short FIR and produces
feedforward taps for the OPX.

Typical pipeline (see ``analyze_inverse_fir`` and node ``fit_fir_data``)::

    measured flux step  (normalized, often resampled 1 ns → 0.5 ns)
            │
            ├─ fit_fir  →  h      (forward: how the *line* distorts a step)
            └─ invert_fir → h_inv (predistortion: what to play on the DAC)

Only ``h_inv`` is written to state as ``feedforward_filter``. ``h`` is an
intermediate model / diagnostic. Hyperparameters (``L``, ``lam``, …) are fixed
by the caller — there is no GCV / L-curve auto-tuning.

Math follows Hellings et al. arXiv:2503.04610, Appendix I (regularised forward
fit and smoothed inverse). ``estimate_noise_floor`` is optional and does not
enter the solve; it only flags unsettled tails before you trust / update taps.
"""

from typing import Dict, Literal, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg
from scipy.interpolate import interp1d


def conv_causal(v: ArrayLike, h: ArrayLike, N: Optional[int] = None) -> np.ndarray:
    """Causal (one-sided) convolution ``y = v * h``, truncated to length ``N``.

    Used when applying FIR taps in time. Default ``N = len(v)`` keeps the
    output on the same grid as the input (no trailing convolution tail).
    """
    v = np.asarray(v, dtype=float)
    h = np.asarray(h, dtype=float)
    y = np.convolve(v, h, mode="full")
    return y[: (len(v) if N is None else N)]


def build_toeplitz_matrix(v: ArrayLike, L: int) -> np.ndarray:
    """Build the convolution matrix ``V`` so ``phi ≈ V @ h`` for an FIR of length ``L``.

    ``V`` has shape ``(len(v), L)``. This is the linear-algebra form of
    filtering the stimulus ``v`` with taps ``h``.
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
    """Interpolate a time trace onto a finer/coarser uniform grid (ns).

    Cryoscope data are typically acquired at 1 GS/s (``original_Ts=1``) while
    OPX FIR taps are often defined at 0.5 ns (``target_Ts=0.5``). Pass the
    real measurement axis via ``t_original_ns`` (e.g. ``ds_fit.time``) so the
    upsampled samples stay aligned with physical time.

    Returns
    -------
    y_target, t_target
        Resampled amplitudes and the corresponding time grid (ns).
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
    if t_end < t_start:
        raise ValueError("t_original_ns must be sorted non-decreasing")

    dt = float(target_Ts)
    if dt <= 0:
        raise ValueError("target_Ts must be positive")

    n_steps = int(np.floor((t_end - t_start) / dt)) + 1
    t_target = t_start + np.arange(n_steps, dtype=float) * dt
    t_target = t_target[t_target <= t_end + 1e-9]

    interp_fun = interp1d(
        t_original, data, kind=kind, fill_value="extrapolate", bounds_error=False
    )
    return np.asarray(interp_fun(t_target), dtype=float), t_target


def fit_fir(
    phi: ArrayLike,
    v: ArrayLike,
    L: int,
    Ts: float = 0.5,
    lam1: float = 1e-2,
    lam2: float = 1e-2,
    tail_ns: Optional[float] = None,
) -> np.ndarray:
    """Fit the *forward* FIR ``h`` that maps stimulus ``v`` → measured ``phi``.

    Solves the regularised least-squares problem ``phi ≈ V @ h`` where ``V`` is
    the Toeplitz matrix of ``v`` (usually a unit step). Regularisation:

    - ``lam1`` — Tikhonov (``‖h‖²``) so noise does not explode the taps.
    - ``lam2`` — exponentially growing penalty on late taps (tail prior).

    Returns ``h`` of length ``L``. This is **not** what goes into QUAM state;
    it models the distortion. Predistortion taps come from :func:`invert_fir`.
    """
    phi = np.asarray(phi, float)
    v = np.asarray(v, float)
    V = build_toeplitz_matrix(v, L)

    if tail_ns is None:
        tail_ns = (L * Ts) / 3.0
    x = np.exp(np.arange(L) * Ts / tail_ns)

    A = V.T @ V + lam1 * np.eye(L) + lam2 * np.diag(x)
    b = V.T @ phi
    return linalg.solve(A, b, assume_a="pos")


def invert_fir(
    h: ArrayLike,
    Ts: float = 0.5,
    M: Optional[int] = None,
    method: Literal["optimization", "analytical"] = "optimization",
    sigma_ns: float = 0.75,
    lam_smooth: float = 5e-2,
    normalize_dc_gain: bool = False,
) -> np.ndarray:
    """Compute predistortion taps ``h_inv`` ≈ causal inverse of forward FIR ``h``.

    Goal: ``h * h_inv ≈`` a narrow pulse (smoothed δ), so playing ``h_inv`` on
    the DAC undoes the line response. These coefficients are what the node
    stores as ``z.opx_output.feedforward_filter``.

    Parameters
    ----------
    h :
        Forward FIR from :func:`fit_fir`.
    M :
        Inverse length; defaults to ``len(h)``.
    method :
        ``"optimization"`` (default) — soft Gaussian target + smoothness
        regulariser (Hellings). ``"analytical"`` — recursive exact inverse
        of a minimum-phase FIR (more brittle on noisy ``h``).
    sigma_ns :
        Width of the Gaussian δ target (ns). Wider → less noise gain.
    lam_smooth :
        Weight on first-difference smoothness of ``h_inv``.
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


def analyze_inverse_fir(
    response: ArrayLike,
    Ts: float = 0.5,
    L: int = 48,
    lam: float = 1e-2,
    M: Optional[int] = None,
    sigma_ns: Optional[float] = None,
    lam_smooth: float = 5e-2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """End-to-end FIR solve: measured step → forward ``h`` → inverse ``h_inv``.

    This is the glue used by the calibration node. Assumes ``response`` is
    already a *normalized* flux step (settled level ≈ 1), typically after
    :func:`resample_to_target_rate`.

    Steps
    -----
    1. ``fit_fir(response, ideal_step, L)`` with ``lam1 = lam2 = lam``.
    2. L1-normalise ``h`` (DC gain 1).
    3. Rebuild the model step ``reconstructed = V @ h`` and score NRMS.
    4. ``invert_fir(h)`` → ``h_inv`` (state candidate).

    Returns
    -------
    h, h_inv, reconstructed, info
        ``info`` includes ``L``, ``lam``, ``lam_smooth``, ``sigma_ns``,
        ``forward_nrms`` (‖meas − recon‖ / ‖meas‖; ~0 is good), ``forward_rss``.

    Notes
    -----
    Default ``sigma_ns = 1.3 * Ts`` matches cryoscope Nyquist: do not ask the
    inverse for sub-sample spikes the measurement cannot resolve.
    """
    response = np.asarray(response, float)
    ideal = np.ones_like(response)

    h = fit_fir(response, ideal, L=L, Ts=Ts, lam1=lam, lam2=lam)
    h = h / np.sum(h)
    reconstructed = build_toeplitz_matrix(ideal, L) @ h
    residual = response - reconstructed
    nrms = float(np.linalg.norm(residual) / max(np.linalg.norm(response), 1e-30))
    rss = float(np.sum(residual**2))

    if M is None:
        M = L
    if sigma_ns is None:
        sigma_ns = 1.3 * Ts

    h_inv = invert_fir(
        h, Ts=Ts, M=M, method="optimization", sigma_ns=sigma_ns, lam_smooth=lam_smooth
    )

    info = {
        "L": L,
        "lam": lam,
        "lam1": lam,
        "lam2": lam,
        "lam_smooth": lam_smooth,
        "sigma_ns": sigma_ns,
        "forward_nrms": nrms,
        "forward_rss": rss,
    }
    return h, h_inv, reconstructed, info


def estimate_noise_floor(
    signal: ArrayLike,
    Ts: float,
    tail_window_ns: float = 30.0,
    warn_ratio: float = 1.5,
) -> Dict:
    """Sanity-check whether the late-time step has settled (does not affect the FIR solve).

    FIR / IIR both assume the *tail* is the true DC level used for
    normalisation. If the tail still drifts, ``h_inv`` is unreliable.

    On the last ``tail_window_ns`` of ``signal`` (use the raw 1 GS/s grid):

    - ``sigma_A`` = ``std(tail)`` — inflated by slow drift or unsettled physics.
    - ``sigma_B`` = ``std(diff(tail)) / √2`` — sensitive mainly to sample noise.

    If ``max/min(sigma_A, sigma_B) > warn_ratio``, status is ``"warn_tail"``
    (suggest longer ``cryoscope_len``). Otherwise ``"ok"``.
    """
    signal = np.asarray(signal, float)
    tail = signal[-max(int(round(tail_window_ns / Ts)), 4) :]

    sigma_A = float(np.std(tail))
    sigma_B = float(np.std(np.diff(tail)) / np.sqrt(2.0))
    s_max = max(sigma_A, sigma_B)
    s_min = max(min(sigma_A, sigma_B), 1e-30)
    ratio_AB = s_max / s_min

    if ratio_AB > warn_ratio:
        status, msg = "warn_tail", "WARN: extend cryoscope_len"
    else:
        status, msg = "ok", "OK"

    return {
        "sigma_A": sigma_A,
        "sigma_B": sigma_B,
        "displayed": sigma_B,
        "ratio_AB": float(ratio_AB),
        "status": status,
        "msg_short": msg,
    }
