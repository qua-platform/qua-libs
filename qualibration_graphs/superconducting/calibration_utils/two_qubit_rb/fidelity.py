"""Alpha -> fidelity conversion for two-qubit RB, for both Standard and
Interleaved protocols
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_FLOAT_TOLERANCE = 1e-9


@dataclass
class SRBFidelity:
    """Fidelity quantities derived from a Standard RB fit."""

    fidelity: float
    fidelity_stderr: float | None
    epc: float
    epg: float | None
    average_gate_fidelity: float | None


@dataclass
class IRBFidelity:
    """Fidelity quantities derived from an Interleaved RB fit."""

    fidelity: float
    fidelity_stderr: float | None
    epc: float
    epg: float
    epg_stderr: float | None


@dataclass
class ImpliedCZResult:
    """Result of inferring per-CZ RB decay / fidelity from 1Q + 2Q RB data
    (McKay et al. Eq. 10, https://arxiv.org/abs/1712.06550v2) — a cross-check available from Standard RB alone."""

    alpha_0: float | None
    alpha_1: float | None
    alpha_01: float | None
    alpha_01_stderr: float | None
    epg_cz: float | None
    epg_cz_stderr: float | None


def clifford_fidelity_from_alpha(alpha: float, n_qubits: int = 2) -> float:
    """Average Clifford fidelity from an RB decay constant."""
    d = 2**n_qubits
    r = 1 - alpha - (1 - alpha) / d
    return 1 - r


def interleaved_gate_fidelity_from_alpha(alpha: float, standard_rb_alpha: float, n_qubits: int = 2) -> float:
    """Interleaved gate fidelity using https://arxiv.org/pdf/1210.7011."""
    return 1 - ((2**n_qubits - 1) * (1 - alpha / standard_rb_alpha) / 2**n_qubits)


def fidelity_stderr_from_alpha(
    alpha_stderr: float | None,
    *,
    interleaved: bool,
    standard_rb_alpha: float | None = None,
    n_qubits: int = 2,
) -> float | None:
    """Propagate 1σ uncertainty on RB decay ``alpha`` to fidelity (1σ)."""
    if alpha_stderr is None or not np.isfinite(alpha_stderr) or alpha_stderr <= 0:
        return None

    d = 2**n_qubits
    if interleaved:
        if standard_rb_alpha is None or not np.isfinite(standard_rb_alpha) or standard_rb_alpha <= 0:
            return None
        return ((d - 1) / d) * alpha_stderr / standard_rb_alpha

    return ((d - 1) / d) * alpha_stderr


def compute_srb_fidelity(
    alpha: float,
    alpha_stderr: float | None,
    average_gates_per_clifford: float | None,
) -> SRBFidelity:
    """2Q Clifford fidelity, EPC, and per-gate EPG from a Standard RB fit."""
    fidelity = clifford_fidelity_from_alpha(alpha)
    epc = 1 - fidelity
    fidelity_stderr = fidelity_stderr_from_alpha(alpha_stderr, interleaved=False)

    if average_gates_per_clifford is not None and average_gates_per_clifford > 0:
        epg = epc / average_gates_per_clifford
        average_gate_fidelity = 1 - epg
    else:
        epg = None
        average_gate_fidelity = None

    return SRBFidelity(
        fidelity=fidelity,
        fidelity_stderr=fidelity_stderr,
        epc=epc,
        epg=epg,
        average_gate_fidelity=average_gate_fidelity,
    )


def compute_irb_fidelity(
    alpha: float,
    alpha_stderr: float | None,
    standard_rb_alpha: float,
    *,
    epc_reference: float | None = None,
) -> IRBFidelity:
    """Direct CZ gate fidelity from interleaved alpha vs. a reference Standard RB alpha."""
    fidelity = interleaved_gate_fidelity_from_alpha(alpha, standard_rb_alpha)
    epg = 1 - fidelity
    epc = epc_reference if epc_reference is not None else 1 - clifford_fidelity_from_alpha(standard_rb_alpha)
    fidelity_stderr = fidelity_stderr_from_alpha(alpha_stderr, interleaved=True, standard_rb_alpha=standard_rb_alpha)

    return IRBFidelity(
        fidelity=fidelity,
        fidelity_stderr=fidelity_stderr,
        epc=epc,
        epg=epg,
        epg_stderr=fidelity_stderr,
    )


def validate_fidelity_bounds(fidelity: float, *, interleaved: bool) -> list[str]:
    """Hard-fail checks on extracted gate / Clifford fidelity."""
    label = "CZ gate fidelity" if interleaved else "2Q Clifford fidelity"
    issues: list[str] = []
    if not np.isfinite(fidelity):
        issues.append(f"{label} is non-finite.")
    elif fidelity < -_FLOAT_TOLERANCE or fidelity > 1.0 + _FLOAT_TOLERANCE:
        issues.append(f"{label}={100 * fidelity:.4f}% outside physical range [0, 100%].")
    return issues


def validate_interleaved_alpha(alpha: float, standard_rb_alpha: float) -> list[str]:
    """Hard-fail checks specific to interleaved RB reference comparison."""
    issues: list[str] = []
    if not np.isfinite(standard_rb_alpha):
        issues.append("Reference StandardRB_alpha is non-finite.")
        return issues

    if standard_rb_alpha <= _FLOAT_TOLERANCE or standard_rb_alpha > 1.0 + _FLOAT_TOLERANCE:
        issues.append(f"Reference StandardRB_alpha={standard_rb_alpha:.6f} outside physical range (0, 1].")

    if alpha > standard_rb_alpha + _FLOAT_TOLERANCE:
        issues.append(
            f"Interleaved alpha={alpha:.6f} exceeds reference StandardRB_alpha="
            f"{standard_rb_alpha:.6f} (implies CZ fidelity > 100%)."
        )
    return issues


def _qubit_rb_alpha(qubit) -> float | None:
    """1Q RB decay ``alpha`` for a qubit, derived from its stored gate fidelity."""
    gf = getattr(qubit, "gate_fidelity", None) or {}
    if "averaged" not in gf:
        return None
    f = float(gf["averaged"])
    if not np.isfinite(f):
        return None
    return 2.0 * f - 1.0


def infer_alpha_01(
    alpha_2q: float,
    qubit_control,
    qubit_target,
    n_1: float,
    n_2: float,
) -> tuple[float | None, float | None, float | None]:
    """Infer per-CZ RB decay alpha_01 by inverting Eq. (10) of supplemental material of McKay et al. (https://arxiv.org/abs/1712.06550v2)."""
    if n_2 <= 0 or not np.isfinite(n_1) or n_1 < 0:
        return None, None, None

    alpha_0 = _qubit_rb_alpha(qubit_control)
    alpha_1 = _qubit_rb_alpha(qubit_target)
    if alpha_0 is None or alpha_1 is None:
        return alpha_0, alpha_1, None
    if not np.isfinite(alpha_2q) or alpha_2q <= 0 or alpha_0 <= 0 or alpha_1 <= 0:
        return alpha_0, alpha_1, None

    half = n_1 / 2.0
    a0, a1 = alpha_0**half, alpha_1**half
    factor_1q = (a0 + a1 + 3.0 * a0 * a1) / 5.0
    if factor_1q <= 0:
        return alpha_0, alpha_1, None

    ratio = alpha_2q / factor_1q
    if ratio <= 0:
        return alpha_0, alpha_1, None

    return alpha_0, alpha_1, float(ratio ** (1.0 / n_2))


def _alpha_01_stderr_from_alpha_2q(
    alpha_01: float,
    alpha_2q: float,
    alpha_2q_stderr: float | None,
    n_2: float,
) -> float | None:
    """Propagate fit uncertainty on ``alpha_2q`` to inferred ``alpha_01`` (1Q inputs held fixed)."""
    if (
        alpha_2q_stderr is None
        or not np.isfinite(alpha_2q_stderr)
        or alpha_2q_stderr <= 0
        or not np.isfinite(alpha_01)
        or not np.isfinite(alpha_2q)
        or alpha_2q <= 0
        or n_2 <= 0
    ):
        return None
    return alpha_01 * alpha_2q_stderr / (n_2 * alpha_2q)


def compute_implied_cz(
    alpha_2q: float,
    alpha_2q_stderr: float | None,
    qubit_control,
    qubit_target,
    n_1: float | None,
    n_2: float | None,
) -> ImpliedCZResult | None:
    """Full implied-CZ pipeline: infer alpha_01, propagate error, convert to EPG."""
    if n_1 is None or n_2 is None:
        return None

    alpha_0, alpha_1, alpha_01 = infer_alpha_01(alpha_2q, qubit_control, qubit_target, n_1, n_2)

    alpha_01_stderr = None
    epg_cz = None
    epg_cz_stderr = None
    if alpha_01 is not None:
        alpha_01_stderr = _alpha_01_stderr_from_alpha_2q(alpha_01, alpha_2q, alpha_2q_stderr, n_2)
        epg_cz = 1.0 - clifford_fidelity_from_alpha(alpha_01, n_qubits=2)
        if alpha_01_stderr is not None:
            epg_cz_stderr = fidelity_stderr_from_alpha(alpha_01_stderr, interleaved=False, n_qubits=2)

    return ImpliedCZResult(
        alpha_0=alpha_0,
        alpha_1=alpha_1,
        alpha_01=alpha_01,
        alpha_01_stderr=alpha_01_stderr,
        epg_cz=epg_cz,
        epg_cz_stderr=epg_cz_stderr,
    )
