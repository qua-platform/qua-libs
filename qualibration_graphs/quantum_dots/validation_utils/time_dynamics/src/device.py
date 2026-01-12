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
from typing import Literal, Sequence, Protocol, Tuple

import dynamiqs as dq
import jax.numpy as jnp

from .pulse import GaussianPulse, CouplingPulse
from .utils import embed_two_qubit_op, embed_single_qubit_op, kron_n


# --- Base protocol (for typing pulses in signatures) ---
class PulseLike(Protocol):
    """
    Protocol for pulse-like objects.

    Any object implementing this protocol can be used as a pulse in
    device Hamiltonians.
    """

    def timecallable(self):
        """Return a callable that evaluates the pulse as a function of time."""
        ...


# =======================
# Base class (generic)
# =======================
@dataclass(frozen=True)
class QuantumDeviceBase:
    """
    Generic quantum device base.
    Override `static_h()` for your model's time-independent part.
    You can also override `_drive_xy_ops()` or `_effective_drive_phase_evolution()`
    to change how drives are mapped.
    """

    n: int
    frame: Literal["lab", "rot"] = "rot"

    # ---- overridables ----
    def static_h(self) -> dq.QArray:
        """Return time-independent Hamiltonian as a dq.QArray with dims=(2,)*n."""
        return 0.0 * kron_n([dq.eye(2)] * self.n)  # default: zero

    def _drive_xy_ops(self, which: int):
        """
        Return the (X_j, Y_j) operators for a single-qubit drive.

        Parameters
        ----------
        which : int
            Index of the qubit to drive (0-based)

        Returns
        -------
        tuple[dq.QArray, dq.QArray]
            Tuple of (X_j, Y_j) operators acting on qubit `which`
        """
        return (embed_single_qubit_op(dq.sigmax(), which, self.n), embed_single_qubit_op(dq.sigmay(), which, self.n))

    def _effective_drive_phase_evolution(self, which: int, drive_freq: jnp.ndarray | None):
        """
        Return phase evolution function g(t) = exp(i * ω_eff * t).

        Default implementation assumes rotating frame with zero effective
        frequency unless a drive_freq is specified. Child classes can override
        to implement lab/rotating frame logic.

        Parameters
        ----------
        which : int
            Index of the driven qubit
        drive_freq : jnp.ndarray | None
            Drive carrier frequency in rad/s, or None for on-resonance drive

        Returns
        -------
        Callable[[jnp.ndarray], jnp.ndarray]
            Function that computes exp(i * ω_eff * t)
        """
        omega_eff = 0.0 if drive_freq is None else drive_freq

        def g(t):
            return jnp.exp(1j * (omega_eff * t))

        return g

    # ---- fixed composition logic (reused by all children) ----
    def hamiltonian_with_pulses(self, pulses: Sequence[tuple[int, GaussianPulse]]) -> dq.TimeQArray:
        """
        Build time-dependent Hamiltonian with single-qubit drive pulses.

        Constructs H(t) = H_static + Σ_j [Re(s_j(t)) X_j + Im(s_j(t)) Y_j],
        where s_j(t) = (amp * envelope * exp(i*phase)) * exp(i*ω_eff*t).

        The drive is decomposed into X and Y components to handle the
        complex-valued control in a way compatible with dynamiqs.

        Parameters
        ----------
        pulses : Sequence[tuple[int, GaussianPulse]]
            List of (qubit_index, pulse) tuples

        Returns
        -------
        dq.TimeQArray
            Time-dependent Hamiltonian operator
        """
        Ht = dq.constant(self.static_h())

        for which, pulse in pulses:
            s_base = pulse.timecallable()  # complex envelope
            wobble = self._effective_drive_phase_evolution(which, getattr(pulse, "drive_freq", None))

            def s_re(t):
                val = s_base(t) * wobble(t)
                return jnp.real(val)

            def s_im(t):
                val = s_base(t) * wobble(t)
                return jnp.imag(val)

            Xj, Yj = self._drive_xy_ops(which)
            Ht = Ht + dq.modulated(s_re, Xj) + dq.modulated(s_im, Yj)

        return Ht

    def hamiltonian_with_controls(
        self,
        drives: Sequence[tuple[int, GaussianPulse]] = (),
        couplings: Sequence[tuple[Tuple[int, int], CouplingPulse, str]] = (),
    ) -> dq.TimeQArray:
        """
        Build complete time-dependent Hamiltonian with drives and couplings.

        Combines the static Hamiltonian with:
        - Single-qubit drive pulses (X/Y control)
        - Two-qubit Heisenberg-type coupling modulation

        Parameters
        ----------
        drives : Sequence[tuple[int, GaussianPulse]], optional
            List of (qubit_index, pulse) tuples for single-qubit drives
        couplings : Sequence[tuple[tuple[int, int], CouplingPulse, str]], optional
            List of ((qubit_i, qubit_j), coupling_pulse, kind) tuples for
            time-dependent two-qubit interactions

        Returns
        -------
        dq.TimeQArray
            Full time-dependent Hamiltonian H(t)

        Notes
        -----
        The coupling term uses the Heisenberg form:
        H_coupling(t) = J(t) * 0.25 * (X_i X_j + Y_i Y_j + Z_i Z_j)
        """
        Ht = dq.constant(self.static_h())

        # --- single-qubit drives ---
        for which, pulse in drives:
            s_base = pulse.timecallable()
            wobble = self._effective_drive_phase_evolution(which, getattr(pulse, "drive_freq", None))

            def s_re(t):
                return jnp.real(s_base(t) * wobble(t))

            def s_im(t):
                return jnp.imag(s_base(t) * wobble(t))

            Xj, Yj = self._drive_xy_ops(which)
            Ht = Ht + dq.modulated(s_re, Xj) + dq.modulated(s_im, Yj)

        # --- two-qubit couplings (Heisenberg form) ---
        for (i, j), cpulse in couplings:
            Jt = cpulse.timecallable()  # time-dependent coupling strength J(t)
            XX = embed_two_qubit_op(dq.sigmax(), dq.sigmax(), i, j, self.n)
            YY = embed_two_qubit_op(dq.sigmay(), dq.sigmay(), i, j, self.n)
            ZZ = embed_two_qubit_op(dq.sigmaz(), dq.sigmaz(), i, j, self.n)
            # Heisenberg coupling: J(t) * (XX + YY + ZZ) / 4
            op = 0.25 * (XX + YY + ZZ)

            Ht = Ht + dq.modulated(Jt, op)

        return Ht

    def _jump_operators(self) -> Sequence[dq.QArray]:
        """
        Return jump operators for master equation simulations.

        Override this method in subclasses to define decoherence channels
        for use with mesolve.

        Returns
        -------
        Sequence[dq.QArray]
            List of jump (Lindblad) operators

        Raises
        ------
        NotImplementedError
            If not overridden in a subclass
        """
        raise NotImplementedError("Subclass must implement _jump_operators for master equation")


# ===================================
# Child class: Two-spin Heisenberg / ZZ
# ===================================
@dataclass(frozen=True)
class TwoSpinDevice(QuantumDeviceBase):
    """
    Two-spin device that supports both LAB and ROT frames.

    LAB frame:
      H = 0.5 * Σ_j ω_j Z_j + 0.25*(Jxx X⊗X + Jyy Y⊗Y + Jzz Z⊗Z)

    ROT frame:
      H = 0.5 * Σ_j Δ_j Z_j + 0.25*(Jzz Z⊗Z)   (Jxx/Jyy typically RWA-dropped)
      where Δ_j = ω_j - ω_ref,j  (if ref_omega provided)
    """

    n: int = 2
    frame: Literal["lab", "rot"] = "rot"

    # physical qubit angular frequencies (rad/s)
    omega: Sequence[float] = (0.0, 0.0)
    # per-qubit reference frame (only used in 'rot' to set detuning)
    ref_omega: Sequence[float] | None = None

    # Couplings
    Jxx: float = 0.0
    Jyy: float = 0.0
    Jzz: float = 0.0

    Tphi1: float = 1e6
    Tphi2: float = 1e6

    # ---- overrides ----
    def static_h(self) -> dq.QArray:
        """
        Build the static (time-independent) Hamiltonian.

        In LAB frame:
            H = 0.5 * Σ_j ω_j Z_j + 0.25 * (Jxx X⊗X + Jyy Y⊗Y + Jzz Z⊗Z)

        In ROT (rotating) frame:
            H = 0.5 * Σ_j Δ_j Z_j + 0.25 * Jzz Z⊗Z
            where Δ_j = ω_j - ω_ref,j

        Returns
        -------
        dq.QArray
            Static Hamiltonian operator
        """
        H = 0.0 * kron_n([dq.eye(2)] * self.n)

        if self.frame == "lab":
            # Zeeman terms: ω_j Z_j / 2
            for j, w in enumerate(self.omega):
                if w != 0.0:
                    H = H + 0.5 * w * embed_single_qubit_op(dq.sigmaz(), j, self.n)
        else:  # 'rot'
            # Detuning terms: Δ_j Z_j / 2, where Δ_j = ω_j - ω_ref,j
            if self.ref_omega is not None:
                for j, (w, wref) in enumerate(zip(self.omega, self.ref_omega)):
                    delta = w - wref
                    if delta != 0.0:
                        H = H + 0.5 * delta * embed_single_qubit_op(dq.sigmaz(), j, self.n)

        # Two-qubit coupling terms
        if self.n >= 2:
            if self.Jxx != 0.0:
                H = H + 0.25 * self.Jxx * kron_n([dq.sigmax(), dq.sigmax()] + [dq.eye(2)] * (self.n - 2))
            if self.Jyy != 0.0:
                H = H + 0.25 * self.Jyy * kron_n([dq.sigmay(), dq.sigmay()] + [dq.eye(2)] * (self.n - 2))
            if self.Jzz != 0.0:
                H = H + 0.25 * self.Jzz * kron_n([dq.sigmaz(), dq.sigmaz()] + [dq.eye(2)] * (self.n - 2))
        return H

    def _effective_drive_phase_evolution(self, which: int, drive_freq: jnp.ndarray | None):
        """
        Compute frame-dependent phase evolution for drives.

        Returns g(t) = exp(i * ω_eff * t) where:
        - LAB frame: ω_eff = drive_freq (or 0 if None)
        - ROT frame: ω_eff = drive_freq - ω_ref[which] (or 0 if None)

        Parameters
        ----------
        which : int
            Index of the driven qubit
        drive_freq : jnp.ndarray | None
            Drive carrier frequency in rad/s

        Returns
        -------
        Callable[[jnp.ndarray], jnp.ndarray]
            Phase evolution function g(t)
        """
        if self.frame == "lab":
            omega_eff = 0.0 if drive_freq is None else drive_freq
        else:
            if drive_freq is None:
                omega_eff = 0.0
            else:
                wref = 0.0 if self.ref_omega is None else self.ref_omega[which]
                omega_eff = drive_freq - wref

        def g(t):
            return jnp.exp(1j * (omega_eff * t))

        return g

    def _jump_operators(self) -> Sequence[dq.QArray]:
        """
        Construct pure dephasing jump operators for each qubit.

        Implements Lindblad operators L_j = sqrt(γ_φ,j) * Z_j where
        γ_φ,j = 1/(2*T_φ,j). This produces exp(-t/T_φ) decay of
        off-diagonal density matrix elements.

        Returns
        -------
        Sequence[dq.QArray]
            List of jump operators [L_φ,1, L_φ,2, ...]

        Notes
        -----
        The master equation is:
            dρ/dt = -i[H, ρ] + Σ_j γ_φ,j (Z_j ρ Z_j - ρ)
        which is equivalent to Lindblad form with L_j = sqrt(γ_φ,j) Z_j
        """
        # Dephasing rates: γ_φ = 1/(2*T_φ)
        gamma_phi1 = 0.5 / self.Tphi1
        gamma_phi2 = 0.5 / self.Tphi2

        # Jump operators: L = sqrt(γ) * σ_z
        Lphi1 = jnp.sqrt(gamma_phi1) * dq.tensor(dq.sigmaz(), dq.eye(self.n))
        Lphi2 = jnp.sqrt(gamma_phi2) * dq.tensor(dq.eye(self.n), dq.sigmaz())

        dims_2q = (self.n, self.n)

        jump_ops = [
            dq.asqarray(Lphi1, dims=dims_2q),
            dq.asqarray(Lphi2, dims=dims_2q),
        ]
        return jump_ops
