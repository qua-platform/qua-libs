# dynamiqs_qubits.py
from __future__ import annotations
from typing import Callable, Sequence, Union, Literal, Optional, Iterable, Tuple

import jax
import jax.numpy as jnp
import dynamiqs as dq
import matplotlib.pyplot as plt

Array = jnp.ndarray
QArrayLike = dq.QArrayLike
StateLike = QArrayLike  # ket |psi> or density matrix rho
Solver = Literal["se", "me"]  # Schrödinger or Master eq.


# --------- Utilities: tensor products & local ops ----------
def kron_n(ops: Sequence[QArrayLike]) -> dq.QArray:
    """Tensor product of a list of operators (works for qarrays)."""
    out = dq.asqarray(ops[0])
    for op in ops[1:]:
        out = dq.tensor(out, dq.asqarray(op))
    return out


def embed_single_qubit_op(op: QArrayLike, which: int, n: int) -> dq.QArray:
    """Embed single-qubit operator 'op' into an n-qubit space at position 'which' (0-based)."""
    factors = [dq.eye(2) for _ in range(n)]
    factors[which] = dq.asqarray(op)
    return kron_n(factors)


def embed_two_qubit_op(op_a, op_b, a: int, b: int, n: int):
    """Place op_a on qubit a and op_b on qubit b; identities elsewhere."""
    if a == b:
        raise ValueError("a and b must be different qubits")

    low, high = min(a, b), max(a, b)
    ops = []
    for k in range(n):
        if k == low:
            ops.append(op_a if low == a else op_b)
        elif k == high:
            ops.append(op_b if high == b else op_a)
        else:
            ops.append(I2)
    return kron_n(ops)


# Pre-make Paulis on one qubit
SX, SY, SZ, I2 = dq.sigmax(), dq.sigmay(), dq.sigmaz(), dq.eye(2)


# --------- Measurement helpers ----------
def projector(bitstring: str) -> dq.QArray:
    """Projector onto computational |bitstring> (e.g. '00','01','10','11' for n=2)."""
    n = len(bitstring)
    ket = dq.basis([2] * n, [int(b) for b in bitstring])  # |b0 b1 ...>
    return dq.proj(ket)


def expval(x: StateLike, O: QArrayLike) -> Array:
    """⟨O⟩ for ket/bra or density matrix; returns complex (take .real if needed)."""
    return dq.expect(O, x)


def sweep_circuit(
    make_circuit: Callable[[float], Circuit],
    state0: StateLike,
    *args,
    solver: Solver = "se",
    projector=None,
) -> tuple[Array, dq.QArray]:
    """
    Example: param is pulse length or amplitude; rebuild the circuit per param.
    Returns stacked final states for each param (shape (P, dim) or (P, dim, dim)).
    """

    @jax.jit
    def run_one(*args):
        circ = make_circuit(*args)
        state = circ.final_state(state0, solver=solver)
        if projector is not None:
            state = circ.project(state, projector)
        return state

    # vmap over param axis
    return jax.vmap(run_one)(*args)


# ----- Support


def drive_support(pulse):
    """
    Active window of a single-qubit drive:
      [t0, t0 + duration]
    Assumes the Pulse has a .duration; if None/missing, treats as zero-length.
    """
    t0 = jnp.asarray(getattr(pulse, "t0"))
    dur = getattr(pulse, "duration", None)
    if dur is None:
        return t0, t0
    return t0, t0 + jnp.asarray(dur)


def coupling_support(cpulse):
    """
    Active window of a two-qubit coupling (ramp–wait–ramp):
      [t0, t0 + t_rise + t_wait + t_fall]
    """
    t0 = jnp.asarray(cpulse.t0)
    total = cpulse.duration
    return t0, t0 + total


def controls_support(drives, couplings, pad: float = 0.05):
    """
    Compute a padded global window that covers *all* drives and couplings.

    Parameters
    ----------
    drives    : sequence of (which, pulse)
    couplings : sequence of ((i, j), coupling_pulse, kind)
    pad       : fractional padding added to both ends of the window

    Returns
    -------
    (t_start, t_end) : jnp.ndarray, jnp.ndarray
        Padded window. Guarantees t_end > t_start (with a tiny epsilon if needed).
    """
    # Collect intervals
    t_starts = []
    t_ends = []

    for _, p in drives:
        lo, hi = drive_support(p)
        t_starts.append(lo)
        t_ends.append(hi)

    for _, cp in couplings:
        lo, hi = coupling_support(cp)
        t_starts.append(lo)
        t_ends.append(hi)

    if not t_starts:  # no controls → fall back to a default window
        return jnp.array(0.0), jnp.array(1.0)

    t_starts = jnp.stack(t_starts)
    t_ends = jnp.stack(t_ends)

    t_start = jnp.min(t_starts)
    t_end = jnp.max(t_ends)

    # Ensure strictly positive length (avoid equal endpoints)
    width = jnp.maximum(1e-12, t_end - t_start)
    pad_abs = pad * width

    return t_start - pad_abs, t_end + pad_abs


def build_tsave_synced_fixed_n(drives, couplings, n_points: int, pad: float = 0.05):
    """
    Build a uniform tsave aligned to the union of all control windows, padded by `pad`.

    You control the number of samples via `n_points`.
    """
    t_start, t_end = controls_support(drives, couplings, pad=pad)
    return jnp.linspace(t_start, t_end, int(n_points))
