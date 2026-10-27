"""RB decay alpha -> fidelity, for Standard and Interleaved RB on ``n_qubits`` qubits (d = 2**n_qubits)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from uncertainties import ufloat


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
