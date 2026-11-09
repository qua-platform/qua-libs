"""RB decay alpha -> fidelity, for Standard and Interleaved RB on ``n_qubits`` qubits (d = 2**n_qubits)."""

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


def as_ufloat(value: float, stderr: float | None) -> Any:
    """Wrap ``value`` with ``stderr`` for automatic propagation, or return a plain float."""
    if stderr is not None and np.isfinite(stderr) and stderr > 0 and np.isfinite(value):
        return ufloat(value, stderr)
    return value


def unwrap(value: Any) -> tuple[float, float | None]:
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


def clifford_fidelity_from_alpha(alpha: Any, n_qubits: int) -> Any:
    """Average Clifford fidelity from an RB decay constant (plain float or ufloat)."""
    d = 2**n_qubits
    r = 1 - alpha - (1 - alpha) / d
    return 1 - r


def interleaved_gate_fidelity_from_alpha(alpha: Any, standard_rb_alpha: Any, n_qubits: int) -> Any:
    """Interleaved gate fidelity using https://arxiv.org/pdf/1210.7011 (plain float or ufloat)."""
    d = 2**n_qubits
    return 1 - ((d - 1) * (1 - alpha / standard_rb_alpha) / d)


def compute_srb_fidelity(
    alpha: float,
    alpha_stderr: float | None,
    average_gates_per_clifford: float | None,
    *,
    n_qubits: int,
) -> SRBFidelity:
    """Clifford fidelity, EPC, and per-gate EPG from a Standard RB fit."""
    alpha_u = as_ufloat(alpha, alpha_stderr)
    fidelity_u = clifford_fidelity_from_alpha(alpha_u, n_qubits)
    epc_u = 1 - fidelity_u

    epg = None
    epg_stderr = None
    average_gate_fidelity = None
    if average_gates_per_clifford is not None and average_gates_per_clifford > 0:
        epg, epg_stderr = unwrap(epc_u / average_gates_per_clifford)
        average_gate_fidelity = 1 - epg

    fidelity, fidelity_stderr = unwrap(fidelity_u)
    epc, epc_stderr = unwrap(epc_u)
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
    n_qubits: int,
    standard_rb_alpha_stderr: float | None = None,
    epc_reference: float | None = None,
) -> IRBFidelity:
    """Direct fidelity of the interleaved gate from interleaved alpha vs. a reference Standard RB alpha."""
    alpha_u = as_ufloat(alpha, alpha_stderr)
    standard_rb_alpha_u = as_ufloat(standard_rb_alpha, standard_rb_alpha_stderr)
    fidelity_u = interleaved_gate_fidelity_from_alpha(alpha_u, standard_rb_alpha_u, n_qubits)
    epg_u = 1 - fidelity_u
    if epc_reference is not None:
        epc = epc_reference
        epc_stderr = None
    else:
        epc, epc_stderr = unwrap(1 - clifford_fidelity_from_alpha(standard_rb_alpha_u, n_qubits))

    fidelity, fidelity_stderr = unwrap(fidelity_u)
    epg, epg_stderr = unwrap(epg_u)
    return IRBFidelity(
        fidelity=fidelity,
        fidelity_stderr=fidelity_stderr,
        epc=epc,
        epc_stderr=epc_stderr,
        epg=epg,
        epg_stderr=epg_stderr,
    )


def validate_fidelity_bounds(fidelity: float, *, label: str) -> list[str]:
    """Hard-fail checks on an extracted gate or Clifford fidelity (``label`` names it in the message)."""
    issues: list[str] = []
    if not np.isfinite(fidelity):
        issues.append(f"{label} is non-finite.")
    elif fidelity < -_FLOAT_TOLERANCE or fidelity > 1.0 + _FLOAT_TOLERANCE:
        issues.append(f"{label}={100 * fidelity:.4f}% outside physical range [0, 100%].")
    return issues


def validate_fidelity_uncertainty(fidelity: float, fidelity_stderr: float | None, *, label: str) -> list[str]:
    """Hard-fail when the propagated fidelity uncertainty exceeds the point estimate."""
    issues: list[str] = []
    if fidelity_stderr is None or not np.isfinite(fidelity_stderr) or fidelity_stderr <= 0:
        return issues
    if fidelity_stderr > fidelity:
        issues.append(
            f"{label} uncertainty ({100 * fidelity_stderr:.3f}%) exceeds "
            f"point estimate ({100 * fidelity:.3f}%); result is not statistically significant."
        )
    return issues


def validate_interleaved_alpha(
    alpha: float, standard_rb_alpha: float, *, reference_label: str, gate_label: str
) -> list[str]:
    """Hard-fail checks on the interleaved alpha against its Standard RB reference.

    An interleaved alpha above the reference implies a gate fidelity above 100%: noise, or a stale
    reference.
    """
    issues: list[str] = []
    if not np.isfinite(standard_rb_alpha):
        issues.append(f"Reference {reference_label} is non-finite.")
        return issues

    if standard_rb_alpha <= _FLOAT_TOLERANCE or standard_rb_alpha > 1.0 + _FLOAT_TOLERANCE:
        issues.append(f"Reference {reference_label}={standard_rb_alpha:.6f} outside physical range (0, 1].")

    if alpha > standard_rb_alpha + _FLOAT_TOLERANCE:
        issues.append(
            f"Interleaved alpha={alpha:.6f} exceeds reference {reference_label}="
            f"{standard_rb_alpha:.6f} (implies {gate_label} fidelity > 100%)."
        )
    return issues
