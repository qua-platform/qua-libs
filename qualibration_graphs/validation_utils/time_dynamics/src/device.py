"""
Quantum device abstractions for time dynamics simulations.

This module provides a flexible framework for defining quantum devices
with time-dependent Hamiltonians. The base class QuantumDeviceBase
implements generic drive and coupling logic that can be specialized
for specific physical systems.

Classes
-------
PulseLike : Protocol
    Protocol for objects that can be used as pulses
QuantumDeviceBase
    Generic base class for quantum devices with configurable Hamiltonian
TwoSpinDevice
    Specialized device for two-qubit Heisenberg-type interactions,
    supporting both lab and rotating frames
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence, Tuple

import dynamiqs as dq
import jax.numpy as jnp

from .pulse import GaussianPulse, CouplingPulse
from .utils import embed_two_qubit_op, embed_single_qubit_op, kron_n


# =============================================================================
# Base class (simple orchestrator; hooks implemented by child classes)
# =============================================================================
@dataclass(frozen=True)
class QuantumDeviceBase:
    n: int
    frame: Literal["lab", "rot"] = "rot"

    def construct_h(
        self,
        drives: Sequence[tuple[int, GaussianPulse]] = (),
        couplings: Sequence[tuple[Tuple[int, int], CouplingPulse]] = (),  # Heisenberg only
    ) -> dq.TimeQArray:
        """
        Compose H(t) = H_static + H_dynamic(t) + H_drives(t) + H_pulses(t)
        """
        # Static part (built at Python time; safe for jit/vmap)
        Ht = dq.constant(self._static_h())

        # Intrinsic time-dependent pieces (e.g., RW exchange for constant Jxx/Jyy)
        for f, op in self._dynamic_h():
            Ht = Ht + dq.modulated(f, op)

        # Single-qubit drives
        for which, pulse in drives:
            for f, op in self._drive_h(which, pulse):
                Ht = Ht + dq.modulated(f, op)

        # Heisenberg two-qubit pulses
        for (i, j), cpulse in couplings:
            for f, op in self._pulse_h(i, j, cpulse):
                Ht = Ht + dq.modulated(f, op)

        return Ht

    # ---------------- hooks to implement in child classes ----------------
    def _static_h(self) -> dq.QArray:
        raise NotImplementedError("_static_h not implemented")

    def _dynamic_h(self) -> Sequence[tuple[Callable[[jnp.ndarray], jnp.ndarray], dq.QArray]]:
        # Default: no intrinsic time dependence
        return ()

    def _drive_h(self, which: int, pulse: GaussianPulse) -> Sequence[tuple[Callable[[jnp.ndarray], jnp.ndarray], dq.QArray]]:
        raise NotImplementedError("_drive_h not implemented")

    def _pulse_h(self, i: int, j: int, cpulse: CouplingPulse) -> Sequence[tuple[Callable[[jnp.ndarray], jnp.ndarray], dq.QArray]]:
        raise NotImplementedError("_pulse_h not implemented")

    # --------- tiny helpers, reusable by children ---------
    def _effective_drive_phase_evolution(self, which: int, drive_freq: jnp.ndarray | None):
        # Child can override to subtract ref_omega per qubit.
        omega_eff = jnp.asarray(0.0) if drive_freq is None else drive_freq
        def g(t): return jnp.exp(1j * (omega_eff * t))
        return g

    def _xy_ops(self, which: int):
        return (
            embed_single_qubit_op(dq.sigmax(), which, self.n),
            embed_single_qubit_op(dq.sigmay(), which, self.n),
        )

    def _jump_operators(self) -> Sequence[dq.QArray]:
        """
        Pure dephasing jump operators for each qubit.

        Implements L_j = sqrt(γ_φ,j) * Z_j with γ_φ,j = 1/(2*Tφ_j).
        This yields single-qubit coherence decay exp(-t / Tφ_j).
        """
        raise NotImplementedError("_jump_operators not implemented")
