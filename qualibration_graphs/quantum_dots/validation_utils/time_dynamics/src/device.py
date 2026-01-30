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

# =============================================================================
# Two-spin Heisenberg/ZZ with rotating-wave exchange in ROT
# =============================================================================
@dataclass(frozen=True)
class TwoSpinDevice(QuantumDeviceBase):
    n: int = 2
    frame: Literal["lab", "rot"] = "lab"

    # physical qubit angular frequencies (rad/s)
    omega: Sequence[float] = (0.0, 0.0)
    ref_omega: Sequence[float] | None = None

    # constant couplings
    J0: float = 0.0

    # (optional) dephasing parameters for jump ops (not used here)
    Tphi1: float = 1e6
    Tphi2: float = 1e6

    # ---------- math helpers ----------
    def _rw_decompose(self, op: dq.QArray):
        # H(t) = op e^{iωt} + op† e^{-iωt} = cos(ωt)*(op+op†) + sin(ωt)*i(op-op†)
        op_dag = op.conj().mT
        A = op + op_dag                  # Hermitian
        B = 1j * (op - op_dag)           # Hermitian
        return A, B

    def _ref_freqs_ij(self, i: int, j: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Keep as JAX arrays (jit/vmap friendly); no float(...) casts
        if self.ref_omega is None:
            w_i = jnp.asarray(0.0)
            w_j = jnp.asarray(0.0)
        else:
            w_i = jnp.asarray(self.ref_omega[i])
            w_j = jnp.asarray(self.ref_omega[j])
        Delta = w_i - w_j
        Sigma = w_i + w_j
        return w_i, w_j, Delta, Sigma

    def _ff_ops_ij(self, i: int, j: int):
        sp = 0.5 * (dq.sigmax() + 1j * dq.sigmay())
        sm = 0.5 * (dq.sigmax() - 1j * dq.sigmay())
        S_ip_S_jm = embed_two_qubit_op(sp, sm, i, j, self.n)  # σ_i^+ σ_j^-
        S_ip_S_jp = embed_two_qubit_op(sp, sp, i, j, self.n)  # σ_i^+ σ_j^+
        return S_ip_S_jm, S_ip_S_jp

    # ---------- hooks ----------
    def _static_h(self) -> dq.QArray:
        H = 0.0 * kron_n([dq.eye(2)] * self.n)

        if self.frame == "lab":
            # Zeeman: add terms unconditionally; zeros no-op
            for q, w in enumerate(self.omega):
                H = H + 0.5 * jnp.asarray(w) * embed_single_qubit_op(dq.sigmaz(), q, self.n)
            # Static couplings in LAB
            H = H + 0.25 * self.J0 * kron_n([dq.sigmax(), dq.sigmax()] + [dq.eye(2)] * (self.n - 2))
            H = H + 0.25 * self.J0 * kron_n([dq.sigmay(), dq.sigmay()] + [dq.eye(2)] * (self.n - 2))
            H = H + 0.25 * self.J0 * kron_n([dq.sigmaz(), dq.sigmaz()] + [dq.eye(2)] * (self.n - 2))

        elif self.frame == "rot":
            # Detunings: Δ_j Z_j / 2, where Δ_j = ω_j - ω_ref,j (0 if no ref_omega)
            if self.ref_omega is None:
                ref = (0.0,) * self.n
            else:
                ref = self.ref_omega
            for q, (w, wref) in enumerate(zip(self.omega, ref)):
                delta = jnp.asarray(w) - jnp.asarray(wref)
                H = H + 0.5 * delta * embed_single_qubit_op(dq.sigmaz(), q, self.n)
            # Keep ZZ static in ROT
            H = H + 0.25 * self.J0 * kron_n([dq.sigmaz(), dq.sigmaz()] + [dq.eye(2)] * (self.n - 2))

        else:
            raise NotImplementedError

        return H

    def _dynamic_h(self):
        if self.frame != "rot":
            return ()
        i, j = 0, 1
        _, _, Delta, Sigma = self._ref_freqs_ij(i, j)
        S_ip_S_jm, S_ip_S_jp = self._ff_ops_ij(i, j)

        Jflip_scale = (self.J0 + self.J0) / 4.0

        terms = []

        A, B = self._rw_decompose(Jflip_scale * S_ip_S_jm)
        D = Delta
        terms.extend([(lambda t, D=D: jnp.cos(D * t), A),
                      (lambda t, D=D: jnp.sin(D * t), B)])

        return terms

    def _drive_h(self, which: int, pulse: GaussianPulse) -> Sequence[tuple[Callable[[jnp.ndarray], jnp.ndarray], dq.QArray]]:
        # Override to subtract per-qubit ref frequency in ROT
        def wobble_rot(which: int, drive_freq):
            if self.frame == "lab":
                omega_eff = jnp.asarray(0.0) if drive_freq is None else drive_freq
            else:
                if drive_freq is None:
                    omega_eff = jnp.asarray(0.0)
                else:
                    wref = jnp.asarray(0.0) if self.ref_omega is None else jnp.asarray(self.ref_omega[which])
                    omega_eff = drive_freq - wref
            return lambda t: jnp.exp(1j * (omega_eff * t))

        s_base = pulse.timecallable()
        wobble = wobble_rot(which, getattr(pulse, "drive_freq", None))
        Xj, Yj = self._xy_ops(which)

        def s_re(t): return jnp.real(s_base(t) * wobble(t))
        def s_im(t): return jnp.imag(s_base(t) * wobble(t))

        return ((s_re, Xj), (s_im, Yj))

    def _pulse_h(self, i: int, j: int, cpulse: CouplingPulse) -> Sequence[
        tuple[Callable[[jnp.ndarray], jnp.ndarray], dq.QArray]]:
        """
        Heisenberg coupling pulse J(t).
        We automatically subtract any DC baseline from the pulse so that
        static couplings (Jxx/Jyy/Jzz) carry the DC and pulses are pure modulation.
        """
        Jt_raw = cpulse.timecallable()

        # --- Automatic DC subtraction (JIT/vmap-safe) ---
        # We look for common baseline attributes on the pulse; if missing, they default to 0.
        # This avoids Python branching and keeps values as JAX scalars.

        def Jt(t):
            # zero-mean modulation used for both ZZ and RW XX+YY
            return Jt_raw(t) - self.J0

        out: list[tuple[Callable[[jnp.ndarray], jnp.ndarray], dq.QArray]] = []

        # ZZ always static under modulation
        ZZ = embed_two_qubit_op(dq.sigmaz(), dq.sigmaz(), i, j, self.n)
        out.append((Jt, 0.25 * ZZ))

        if self.frame == "rot":
            # XX+YY -> RW flip-flop with scale J(t)/2 on σ_i^+σ_j^- + h.c.
            S_ip_S_jm, _ = self._ff_ops_ij(i, j)
            A, B = self._rw_decompose(S_ip_S_jm)
            _, _, Delta, _ = self._ref_freqs_ij(i, j)
            D = Delta  # capture JAX value

            def c(t, D=D):
                return 0.5 * Jt(t) * jnp.cos(D * t)

            def s(t, D=D):
                return 0.5 * Jt(t) * jnp.sin(D * t)

            # Use extend for clarity when adding multiple terms
            out.extend([(c, A), (s, B)])
        else:
            # LAB: static (XX+YY)/4 * J(t)
            XX = embed_two_qubit_op(dq.sigmax(), dq.sigmax(), i, j, self.n)
            YY = embed_two_qubit_op(dq.sigmay(), dq.sigmay(), i, j, self.n)
            out.append((Jt, 0.25 * (XX + YY)))

        return out

    def _jump_operators(self) -> Sequence[dq.QArray]:
        """
        Pure dephasing jump operators for each qubit.

        Implements L_j = sqrt(γ_φ,j) * Z_j with γ_φ,j = 1/(2*Tφ_j).
        This yields single-qubit coherence decay exp(-t / Tφ_j).
        """
        # JAX-friendly scalars
        gamma_phi1 = jnp.asarray(0.5 / self.Tphi1)
        gamma_phi2 = jnp.asarray(0.5 / self.Tphi2)

        Lphi1 = jnp.sqrt(gamma_phi1) * embed_single_qubit_op(dq.sigmaz(), 0, self.n)
        Lphi2 = jnp.sqrt(gamma_phi2) * embed_single_qubit_op(dq.sigmaz(), 1, self.n)

        # If dynamiqs needs explicit subsystem dims, pass (2, 2) for two qubits.
        # (If embed_* already returns a dq.QArray with dims, you can skip asqarray.)
        return [
            dq.asqarray(Lphi1, dims=(2, 2)),
            dq.asqarray(Lphi2, dims=(2, 2)),
        ]
