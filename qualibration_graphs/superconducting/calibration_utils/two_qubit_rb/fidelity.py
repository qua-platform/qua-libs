"""Alpha -> fidelity conversion for two-qubit RB, for both Standard and
Interleaved protocols
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from uncertainties import ufloat

_FLOAT_TOLERANCE = 1e-9


@dataclass
class SRBFidelity:
    """Fidelity quantities derived from a Standard RB fit."""

    fidelity: float
    fidelity_stderr: float | None
    epc: float
    epc_stderr: float | None
    epg: float | None
    epg_stderr: float | None
    average_gate_fidelity: float | None


@dataclass
class IRBFidelity:
    """Fidelity quantities derived from an Interleaved RB fit."""

    fidelity: float
    fidelity_stderr: float | None
    epc: float
    epc_stderr: float | None
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


def _as_ufloat(value: float, stderr: float | None) -> Any:
    """Wrap ``value`` with ``stderr`` for automatic propagation, or return a plain float."""
    if stderr is not None and np.isfinite(stderr) and stderr > 0 and np.isfinite(value):
        return ufloat(value, stderr)
    return value


def _unwrap(value: Any) -> tuple[float, float | None]:
    """Return ``(nominal, stderr)`` from a ``ufloat`` or plain float."""
    if hasattr(value, "n"):
        nominal = float(value.n)
        if not hasattr(value, "s"):
            return nominal, None
        stderr = float(value.s)
        if not np.isfinite(stderr) or stderr <= 0:
            return nominal, None
        return nominal, stderr
    return float(value), None


def clifford_fidelity_from_alpha(alpha: Any, n_qubits: int = 2) -> Any:
    """Average Clifford fidelity from an RB decay constant (plain float or ufloat)."""
    d = 2**n_qubits
    r = 1 - alpha - (1 - alpha) / d
    return 1 - r


def interleaved_gate_fidelity_from_alpha(alpha: Any, standard_rb_alpha: Any, n_qubits: int = 2) -> Any:
    """Interleaved gate fidelity using https://arxiv.org/pdf/1210.7011 (plain float or ufloat)."""
    d = 2**n_qubits
    return 1 - ((d - 1) * (1 - alpha / standard_rb_alpha) / d)


def compute_srb_fidelity(
    alpha: float,
    alpha_stderr: float | None,
    average_gates_per_clifford: float | None,
) -> SRBFidelity:
    """2Q Clifford fidelity, EPC, and per-gate EPG from a Standard RB fit."""
    alpha_u = _as_ufloat(alpha, alpha_stderr)
    fidelity_u = clifford_fidelity_from_alpha(alpha_u)
    epc_u = 1 - fidelity_u

    epg = None
    epg_stderr = None
    average_gate_fidelity = None
    if average_gates_per_clifford is not None and average_gates_per_clifford > 0:
        epg, epg_stderr = _unwrap(epc_u / average_gates_per_clifford)
        average_gate_fidelity = 1 - epg

    fidelity, fidelity_stderr = _unwrap(fidelity_u)
    epc, epc_stderr = _unwrap(epc_u)
    return SRBFidelity(
        fidelity=fidelity,
        fidelity_stderr=fidelity_stderr,
        epc=epc,
        epc_stderr=epc_stderr,
        epg=epg,
        epg_stderr=epg_stderr,
        average_gate_fidelity=average_gate_fidelity,
    )


def compute_irb_fidelity(
    alpha: float,
    alpha_stderr: float | None,
    standard_rb_alpha: float,
    *,
    standard_rb_alpha_stderr: float | None = None,
    epc_reference: float | None = None,
) -> IRBFidelity:
    """Direct CZ gate fidelity from interleaved alpha vs. a reference Standard RB alpha."""
    alpha_u = _as_ufloat(alpha, alpha_stderr)
    standard_rb_alpha_u = _as_ufloat(standard_rb_alpha, standard_rb_alpha_stderr)
    fidelity_u = interleaved_gate_fidelity_from_alpha(alpha_u, standard_rb_alpha_u)
    epg_u = 1 - fidelity_u
    if epc_reference is not None:
        epc = epc_reference
        epc_stderr = None
    else:
        epc, epc_stderr = _unwrap(1 - clifford_fidelity_from_alpha(standard_rb_alpha_u))

    fidelity, fidelity_stderr = _unwrap(fidelity_u)
    epg, epg_stderr = _unwrap(epg_u)
    return IRBFidelity(
        fidelity=fidelity,
        fidelity_stderr=fidelity_stderr,
        epc=epc,
        epc_stderr=epc_stderr,
        epg=epg,
        epg_stderr=epg_stderr,
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


def validate_fidelity_uncertainty(
    fidelity: float,
    fidelity_stderr: float | None,
    *,
    interleaved: bool,
) -> list[str]:
    """Hard-fail when propagated fidelity uncertainty exceeds the point estimate."""
    issues: list[str] = []
    if fidelity_stderr is None or not np.isfinite(fidelity_stderr) or fidelity_stderr <= 0:
        return issues

    label = "CZ gate fidelity" if interleaved else "2Q Clifford fidelity"
    if fidelity_stderr > fidelity:
        issues.append(
            f"{label} uncertainty ({100 * fidelity_stderr:.3f}%) exceeds "
            f"point estimate ({100 * fidelity:.3f}%); result is not statistically significant."
        )
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


def _alpha_01_from_inputs(alpha_2q: Any, alpha_0: float, alpha_1: float, n_1: float, n_2: float) -> Any | None:
    """Infer per-CZ RB decay ``alpha_01`` from 2Q alpha and fixed 1Q reference alphas."""
    if n_2 <= 0 or not np.isfinite(n_1) or n_1 < 0:
        return None
    if not np.isfinite(alpha_0) or alpha_0 <= 0 or not np.isfinite(alpha_1) or alpha_1 <= 0:
        return None

    half = n_1 / 2.0
    a0, a1 = alpha_0**half, alpha_1**half
    factor_1q = (a0 + a1 + 3.0 * a0 * a1) / 5.0
    if factor_1q <= 0:
        return None

    ratio = alpha_2q / factor_1q
    if not hasattr(ratio, "s") and (not np.isfinite(ratio) or ratio <= 0):
        return None
    if hasattr(ratio, "n") and float(ratio.n) <= 0:
        return None

    return ratio ** (1.0 / n_2)


def infer_alpha_01(
    alpha_2q: float,
    qubit_control,
    qubit_target,
    n_1: float,
    n_2: float,
) -> tuple[float | None, float | None, float | None]:
    """Infer per-CZ RB decay alpha_01 by inverting Eq. (10) of supplemental material of McKay et al. (https://arxiv.org/abs/1712.06550v2)."""
    alpha_0 = _qubit_rb_alpha(qubit_control)
    alpha_1 = _qubit_rb_alpha(qubit_target)
    if alpha_0 is None or alpha_1 is None:
        return alpha_0, alpha_1, None
    if not np.isfinite(alpha_2q) or alpha_2q <= 0:
        return alpha_0, alpha_1, None

    alpha_01 = _alpha_01_from_inputs(alpha_2q, alpha_0, alpha_1, n_1, n_2)
    if alpha_01 is None:
        return alpha_0, alpha_1, None
    return alpha_0, alpha_1, _unwrap(alpha_01)[0]


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
    if alpha_01 is not None and alpha_0 is not None and alpha_1 is not None:
        alpha_2q_u = _as_ufloat(alpha_2q, alpha_2q_stderr)
        alpha_01_u = _alpha_01_from_inputs(alpha_2q_u, alpha_0, alpha_1, n_1, n_2)
        if alpha_01_u is not None:
            _, alpha_01_stderr = _unwrap(alpha_01_u)
            epg_cz, epg_cz_stderr = _unwrap(1 - clifford_fidelity_from_alpha(alpha_01_u, n_qubits=2))

    return ImpliedCZResult(
        alpha_0=alpha_0,
        alpha_1=alpha_1,
        alpha_01=alpha_01,
        alpha_01_stderr=alpha_01_stderr,
        epg_cz=epg_cz,
        epg_cz_stderr=epg_cz_stderr,
    )
