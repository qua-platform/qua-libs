"""Leakage randomized benchmarking: g/e/f populations, the leakage fit, per-gate leakage.

Shared by the RB nodes that can read g/e/f instead of g/e (``measure_leakage=True``). The RB
sequences are unchanged; only the final readout assigns each qubit to g, e or f, and the
analysis tracks the population outside the computational subspace against sequence length.

Model (Wood & Gambetta, PRA 97, 032306 (2018)): with per-Clifford leakage ``L1`` (computational
-> leaked) and seepage ``L2`` (leaked -> computational), the leaked population after ``m``
Cliffords is

    P_L(m) = p_inf + (p0 - p_inf) * lam**m,    lam = 1 - L1 - L2,    p_inf = L1 / (L1 + L2)

so ``L1 = p_inf * (1 - lam)`` and ``L2 = (1 - p_inf) * (1 - lam)``. Small leakage rarely
saturates within reachable depths: ``p_inf`` and ``lam`` are then individually loose while
their product, the initial slope, is not. ``L1`` is reported with its propagated error; a fit
that cannot separate them falls back to a straight line whose slope is ``L1``.

Interleaving a gate after every Clifford adds its leakage to every step, so for small rates the
gate's leakage is the interleaved minus the reference rate.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from matplotlib.axes import Axes
from scipy.optimize import OptimizeWarning, curve_fit

NUM_LEVELS = 3
"""Levels each qubit is assigned to: g, e, f."""


@dataclass
class LeakageFit:
    """Fitted leakage and seepage per step (per Clifford, or per interleaved Clifford)."""

    l1: float = float("nan")
    l1_stderr: float = float("nan")
    l2: float = float("nan")
    l2_stderr: float = float("nan")
    p_inf: float = float("nan")
    lam: float = float("nan")
    p0: float = float("nan")
    method: str = ""
    """``exponential`` (full model) or ``linear`` (slope only; ``L2`` unknown)."""
    success: bool = False
    note: str = ""

    def curve(self, depths: Iterable[float]) -> np.ndarray:
        """Return the fitted leaked population at ``depths``."""
        m = np.asarray(depths, dtype=float)
        if not self.success:
            return np.full_like(m, np.nan)
        if self.method == "linear":
            return self.p0 + self.l1 * m
        return self.p_inf + (self.p0 - self.p_inf) * self.lam**m

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-safe dict, with ``None`` in place of NaN."""
        return {key: _finite_or_none(value) for key, value in asdict(self).items()}


@dataclass
class GateLeakage:
    """Leakage of an interleaved gate, from interleaved minus reference."""

    l1: float = float("nan")
    l1_stderr: float = float("nan")
    l2: float = float("nan")
    l2_stderr: float = float("nan")

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-safe dict, with ``None`` in place of NaN."""
        return {key: _finite_or_none(value) for key, value in asdict(self).items()}


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def read_confusion_matrix(qubit: Any) -> Optional[np.ndarray]:
    """Return the qubit's g/e/f confusion matrix, rows prepared and columns measured.

    Returns ``None`` when the resonator has none, or it is not a 3x3 matrix whose rows sum to 1.
    """
    matrix = getattr(getattr(qubit, "resonator", None), "gef_confusion_matrix", None)
    if matrix is None:
        return None
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (NUM_LEVELS, NUM_LEVELS) or not np.allclose(matrix.sum(axis=1), 1.0, atol=0.02):
        return None
    return matrix


def check_gef_readout(qubits: Iterable[Any], pulse_name: str = "readout_GEF") -> None:
    """Raise if a qubit cannot be read as g, e or f.

    ``readout_state_gef`` needs the resonator's ``pulse_name`` operation and three calibrated
    ``gef_centers`` (node 15_iq_blobs_gef).

    Raises:
        ValueError: Listing every qubit that is not ready, and why.
    """
    problems = []
    for qubit in qubits:
        resonator = qubit.resonator
        if pulse_name not in resonator.operations:
            problems.append(f"{qubit.name}: no {pulse_name!r} operation")
        centers = getattr(resonator, "gef_centers", None) or []
        if len(centers) != NUM_LEVELS or any(len(center) != 2 for center in centers):
            problems.append(f"{qubit.name}: no three calibrated gef_centers")
    if problems:
        raise ValueError(
            "measure_leakage needs a calibrated g/e/f readout (run 15_iq_blobs_gef first): " + "; ".join(problems)
        )


def correct_populations(measured: np.ndarray, confusion: Optional[np.ndarray]) -> np.ndarray:
    """Undo readout misassignment on populations along the last axis.

    ``measured = true @ confusion`` for a confusion matrix with rows prepared, so
    ``true = measured @ inv(confusion)``. Negative results from shot noise are clipped and each
    distribution renormalized.

    Args:
        measured: Measured populations, last axis the outcomes.
        confusion: Square matrix matching the last axis, or ``None`` to return ``measured``.
    """
    measured = np.asarray(measured, dtype=float)
    if confusion is None:
        return measured
    corrected = np.clip(measured @ np.linalg.inv(confusion), 0.0, None)
    total = corrected.sum(axis=-1, keepdims=True)
    return np.divide(corrected, total, out=np.zeros_like(corrected), where=total > 0)


def two_qubit_leakage(
    states: np.ndarray,
    shots_axis: int,
    confusion_control: Optional[np.ndarray] = None,
    confusion_target: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Leaked populations of a qubit pair from per-shot states ``3 * control + target``.

    The joint 9-outcome distribution is corrected with the Kronecker product of the two
    confusion matrices (identity for a missing one), so correlated assignments are handled.

    Returns:
        ``pair`` (either qubit in f), ``control`` and ``target`` (that qubit in f), each with
        ``shots_axis`` averaged out.
    """
    states = np.moveaxis(np.asarray(states), shots_axis, -1)
    outcomes = NUM_LEVELS * NUM_LEVELS
    measured = np.stack([(states == k).mean(axis=-1) for k in range(outcomes)], axis=-1)
    eye = np.eye(NUM_LEVELS)
    confusion = None
    if confusion_control is not None or confusion_target is not None:
        confusion = np.kron(
            eye if confusion_control is None else confusion_control,
            eye if confusion_target is None else confusion_target,
        )
    joint = correct_populations(measured, confusion).reshape(*measured.shape[:-1], NUM_LEVELS, NUM_LEVELS)
    leaked_control = joint[..., 2, :].sum(axis=-1)
    leaked_target = joint[..., :, 2].sum(axis=-1)
    return {
        "pair": leaked_control + leaked_target - joint[..., 2, 2],
        "control": leaked_control,
        "target": leaked_target,
    }


def single_qubit_leakage(
    states: np.ndarray, shots_axis: int, confusion: Optional[np.ndarray] = None
) -> np.ndarray:
    """Leaked population of one qubit from per-shot states 0 (g), 1 (e), 2 (f)."""
    states = np.moveaxis(np.asarray(states), shots_axis, -1)
    measured = np.stack([(states == k).mean(axis=-1) for k in range(NUM_LEVELS)], axis=-1)
    return correct_populations(measured, confusion)[..., 2]


def mean_over_sequences(per_sequence: np.ndarray, sequence_axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard error across random sequences (shots are repeats, not samples)."""
    per_sequence = np.asarray(per_sequence, dtype=float)
    count = per_sequence.shape[sequence_axis]
    mean = per_sequence.mean(axis=sequence_axis)
    if count < 2:
        return mean, np.full_like(mean, np.nan)
    return mean, per_sequence.std(axis=sequence_axis, ddof=1) / np.sqrt(count)


def _leakage_model(m, p_inf, lam, p0):
    return p_inf + (p0 - p_inf) * lam**m


def fit_leakage_decay(
    depths: Sequence[float], leaked: Sequence[float], stderr: Optional[Sequence[float]] = None
) -> LeakageFit:
    """Fit leaked population against sequence length.

    Tries the full exponential model; falls back to a straight line (slope ``L1``, ``L2``
    unknown) when the exponential does not converge or cannot pin down ``L1``.
    """
    m = np.asarray(depths, dtype=float)
    p = np.asarray(leaked, dtype=float)
    sigma = None if stderr is None else np.asarray(stderr, dtype=float)
    keep = np.isfinite(m) & np.isfinite(p)
    if sigma is not None:
        keep &= np.isfinite(sigma)
    m, p = m[keep], p[keep]
    if sigma is not None:
        sigma = sigma[keep]
        floor = np.nanmax(sigma) * 1e-3 if np.any(sigma > 0) else 0.0
        sigma = np.maximum(sigma, floor) if floor > 0 else None
    if m.size < 3:
        return LeakageFit(note=f"Need at least 3 depths, got {m.size}.")

    order = np.argsort(m)
    m, p = m[order], p[order]
    sigma = None if sigma is None else sigma[order]

    fit = _fit_exponential(m, p, sigma)
    if fit.success:
        return fit
    linear = _fit_linear(m, p, sigma)
    linear.note = "; ".join(filter(None, [fit.note, linear.note]))
    return linear


def _fit_exponential(m: np.ndarray, p: np.ndarray, sigma: Optional[np.ndarray]) -> LeakageFit:
    p0_guess = float(np.clip(p[0], 0.0, 1.0))
    span = float(m[-1] - m[0]) or 1.0
    slope = max(float((p[-1] - p[0]) / span), 1e-6)
    p_inf_guess = float(np.clip(max(p[-1] * 2.0, slope * 100.0), 1e-4, 0.99))
    lam_guess = float(np.clip(1.0 - slope / p_inf_guess, 0.5, 1.0 - 1e-6))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            params, cov = curve_fit(
                _leakage_model,
                m,
                p,
                p0=[p_inf_guess, lam_guess, p0_guess],
                sigma=sigma,
                absolute_sigma=sigma is not None,
                bounds=([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
                maxfev=20000,
            )
    except (RuntimeError, ValueError) as error:
        return LeakageFit(method="exponential", note=f"Exponential fit failed: {error}")

    p_inf, lam, p0 = (float(x) for x in params)
    l1 = p_inf * (1.0 - lam)
    l2 = (1.0 - p_inf) * (1.0 - lam)
    if not np.all(np.isfinite(cov)):
        return LeakageFit(method="exponential", note="Exponential fit has no usable covariance.")
    grad_l1 = np.array([1.0 - lam, -p_inf, 0.0])
    grad_l2 = np.array([-(1.0 - lam), -(1.0 - p_inf), 0.0])
    l1_stderr = float(np.sqrt(max(grad_l1 @ cov @ grad_l1, 0.0)))
    l2_stderr = float(np.sqrt(max(grad_l2 @ cov @ grad_l2, 0.0)))
    if l1 > 0 and l1_stderr > l1:
        return LeakageFit(
            method="exponential",
            note=f"Exponential fit leaves L1 unconstrained ({l1:.2e} +/- {l1_stderr:.2e}).",
        )
    return LeakageFit(
        l1=l1,
        l1_stderr=l1_stderr,
        l2=l2,
        l2_stderr=l2_stderr,
        p_inf=p_inf,
        lam=lam,
        p0=p0,
        method="exponential",
        success=True,
    )


def _fit_linear(m: np.ndarray, p: np.ndarray, sigma: Optional[np.ndarray]) -> LeakageFit:
    weights = None if sigma is None else 1.0 / sigma
    try:
        coeffs, cov = np.polyfit(m, p, 1, w=weights, cov="unscaled" if sigma is not None else True)
    except (np.linalg.LinAlgError, ValueError) as error:
        return LeakageFit(method="linear", note=f"Linear fit failed: {error}")
    slope, intercept = (float(x) for x in coeffs)
    slope_stderr = float(np.sqrt(max(cov[0, 0], 0.0)))
    return LeakageFit(
        l1=slope,
        l1_stderr=slope_stderr,
        p0=intercept,
        method="linear",
        success=bool(np.isfinite(slope)),
        note="Linear slope only (leakage not saturated): L1 from the slope, L2 unknown.",
    )


def interleaved_gate_leakage(
    interleaved: LeakageFit,
    reference_l1: Optional[float],
    reference_l1_stderr: Optional[float] = None,
    reference_l2: Optional[float] = None,
    reference_l2_stderr: Optional[float] = None,
) -> GateLeakage:
    """Leakage of the interleaved gate: interleaved minus reference, errors in quadrature.

    ``L2`` is only given when both fits measured it.
    """

    def value(x: Optional[float]) -> float:
        return float("nan") if x is None else float(x)

    def quadrature(a: float, b: Optional[float]) -> float:
        return float(np.hypot(a, value(b))) if b is not None else a

    out = GateLeakage()
    if not interleaved.success or reference_l1 is None:
        return out
    out.l1 = interleaved.l1 - float(reference_l1)
    out.l1_stderr = quadrature(interleaved.l1_stderr, reference_l1_stderr)
    if np.isfinite(interleaved.l2) and reference_l2 is not None:
        out.l2 = interleaved.l2 - float(reference_l2)
        out.l2_stderr = quadrature(interleaved.l2_stderr, reference_l2_stderr)
    return out


def plot_leakage_decay(
    ax: Axes,
    depths: Sequence[float],
    leaked: Sequence[float],
    stderr: Optional[Sequence[float]],
    fit: LeakageFit,
    *,
    label: str,
    color: Optional[str] = None,
) -> None:
    """Plot measured leaked population with its fit on ``ax``."""
    depths = np.asarray(depths, dtype=float)
    line = ax.errorbar(
        depths,
        leaked,
        yerr=None if stderr is None else np.nan_to_num(np.asarray(stderr, dtype=float)),
        fmt="o",
        ms=4,
        capsize=2,
        color=color,
        label=label,
    )
    if fit.success:
        dense = np.linspace(depths.min(), depths.max(), 200)
        ax.plot(dense, fit.curve(dense), "-", color=line[0].get_color(), lw=1)
