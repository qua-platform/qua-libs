"""
Quantum circuit abstraction for time dynamics simulations.

This module provides a high-level circuit interface for composing quantum gates
and pulses, then simulating their time evolution using either the Schrödinger
or master equation.

Classes
-------
Gate : dataclass
    Represents a single quantum gate with associated pulse control
Circuit : dataclass
    High-level circuit abstraction that combines gates and devices

Functions
---------
X(which, t0, pulse_class=GaussianPulse, **kwargs) -> Gate
    Create an X gate (π rotation around X axis)
Y(which, t0, pulse_class=GaussianPulse, **kwargs) -> Gate
    Create a Y gate (π rotation around Y axis)
HeisenbergRampGate(which, t0, pulse_class=CouplingPulse, **kwargs) -> Gate
    Create a two-qubit Heisenberg coupling gate with ramp profile

Type Aliases
------------
QArrayLike
    Alias for dynamiqs QArrayLike type
StateLike
    Quantum state (ket |psi> or density matrix rho)
Solver
    Solver type: "se" for Schrödinger equation or "me" for master equation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, Union

import dynamiqs as dq
import jax.numpy as jnp
from dynamiqs.method import Tsit5

from .device import QuantumDeviceBase
from .pulse import GaussianPulse, CouplingPulse, SquarePulse
from .utils import build_tsave_synced_fixed_n

QArrayLike = dq.QArrayLike
StateLike = QArrayLike  # ket |psi> or density matrix rho
Solver = Literal["se", "me"]  # Schrödinger or Master eq.

# --------- Gate library ----------
@dataclass(frozen=True)
class Gate:
    """
    Represents a quantum gate with associated pulse control.

    A Gate encapsulates both the abstract gate operation (e.g., X, Y) and the
    physical pulse that implements it. Gates can be single-qubit drives or
    two-qubit coupling operations.

    Attributes
    ----------
    name : str
        Name of the gate (e.g., "X", "Y", "HeisRamp")
    which : Union[int, tuple[int]]
        Target qubit index (int) for single-qubit gates, or tuple of indices
        (int, int) for two-qubit gates
    pulse : GaussianPulse | CouplingPulse | SquarePulse
        Pulse object defining the time-dependent control field
    type : Literal["coupling", "drive"], default="drive"
        Type of gate: "drive" for single-qubit operations, "coupling" for
        two-qubit interactions
    """
    name: str
    which: Union[int, tuple[int]]
    pulse: GaussianPulse | CouplingPulse | SquarePulse
    type: Literal["coupling", "drive"] = "drive"


def X(which: int, t0, pulse_class=GaussianPulse, **kwargs) -> Gate:
    """
    Create an X gate (π rotation around X axis).

    Parameters
    ----------
    which : int
        Index of the target qubit
    t0 : float
        Start time of the pulse in seconds
    pulse_class : type, default=GaussianPulse
        Pulse class to use for implementing the gate
    **kwargs
        Additional parameters passed to the pulse constructor (amp, phase, etc.)

    Returns
    -------
    Gate
        X gate object with configured pulse

    Examples
    --------
    >>> gate = X(which=0, t0=0.0, amp=2*np.pi, duration=50e-9)
    """
    pulse = pulse_class(t0=t0, **kwargs)
    return Gate("X", which, pulse, "drive")


def Y(which: int, t0, pulse_class=GaussianPulse, **kwargs) -> Gate:
    """
    Create a Y gate (π rotation around Y axis).

    Parameters
    ----------
    which : int
        Index of the target qubit
    t0 : float
        Start time of the pulse in seconds
    pulse_class : type, default=GaussianPulse
        Pulse class to use for implementing the gate
    **kwargs
        Additional parameters passed to the pulse constructor (amp, phase, etc.)

    Returns
    -------
    Gate
        Y gate object with configured pulse

    Examples
    --------
    >>> gate = Y(which=1, t0=100e-9, amp=2*np.pi, duration=50e-9, phase=np.pi/2)
    """
    pulse = pulse_class(t0=t0, **kwargs)
    return Gate("Y", which, pulse, "drive")


def HeisenbergRampGate(
    which: tuple[int, int], t0, pulse_class=CouplingPulse, **kwargs
) -> Gate:
    """
    Create a two-qubit Heisenberg coupling gate with ramp profile.

    Implements a time-dependent Heisenberg interaction J(t) * (XX + YY + ZZ) / 4
    with smooth ramp-up and ramp-down. Useful for adiabatic gate operations
    and controlled-phase gates.

    Parameters
    ----------
    which : tuple[int, int]
        Tuple of (qubit_i, qubit_j) indices for the coupled qubits
    t0 : float
        Start time of the coupling pulse in seconds
    pulse_class : type, default=CouplingPulse
        Pulse class to use (should be CouplingPulse or compatible)
    **kwargs
        Additional parameters passed to the pulse constructor
        (Jmax, t_ramp, duration, shape, etc.)

    Returns
    -------
    Gate
        Heisenberg coupling gate object

    Examples
    --------
    >>> gate = HeisenbergRampGate(
    ...     which=(0, 1), t0=0.0, Jmax=2*np.pi*10e6,
    ...     t_ramp=20e-9, duration=100e-9, shape="cos"
    ... )
    """
    pulse = pulse_class(t0=t0, **kwargs)
    return Gate("HeisRamp", which, pulse, "coupling")

# --------- Circuit abstraction ----------
@dataclass(frozen=True)
class Circuit:
    """
    High-level quantum circuit abstraction.

    A Circuit combines a quantum device (defining the Hamiltonian structure)
    with a sequence of gates (defining the control pulses), and provides
    methods for simulating the time evolution and extracting results.

    Attributes
    ----------
    device : QuantumDeviceBase
        The quantum device defining the system Hamiltonian and frame
    gates : list[Gate]
        Ordered sequence of gates to apply
    tsave : jnp.ndarray | None, default=None
        Time points at which to save the state. If None, automatically
        generated based on the pulse timing
    n_points : int, default=501
        Number of time points for automatic tsave generation (used only
        when tsave is None)
    k_sigma : float, default=5.0
        Window coverage parameter for Gaussian pulses (unused currently,
        kept for compatibility)
    pad : float, default=0.05
        Fractional padding added to the time window (e.g., 0.05 = 5% padding
        on each side)

    Methods
    -------
    apply(state0, solver="se")
        Simulate the circuit and return (t, states) trajectory
    final_state(state0, solver="se")
        Simulate the circuit and return only the final state
    project(state, projector)
        Compute expectation value <projector> for a given state

    Examples
    --------
    >>> from dynamiqs import basis
    >>> device = TwoSpinDevice(n=2, omega=(1e9, 1.1e9))
    >>> gates = [X(which=0, t0=0.0, amp=2*np.pi, duration=50e-9)]
    >>> circuit = Circuit(device=device, gates=gates)
    >>> state0 = basis([2, 2], [0, 0])  # |00>
    >>> t, states = circuit.apply(state0, solver="se")
    """
    device: QuantumDeviceBase
    gates: list[Gate]
    tsave: jnp.ndarray | None = None
    n_points: int = 501
    k_sigma: float = 5.0
    pad: float = 0.05

    def apply(self, state0, solver="se"):
        """
        Simulate the full circuit time evolution.

        Parameters
        ----------
        state0 : StateLike
            Initial quantum state (ket or density matrix)
        solver : Literal["se", "me"], default="se"
            Solver to use: "se" for Schrödinger equation (pure states),
            "me" for master equation (open systems with decoherence)

        Returns
        -------
        t : jnp.ndarray
            Time points at which states were saved (shape: (n_times,))
        states : jnp.ndarray
            Quantum states at each time point
            - For "se": shape (n_times, dim) for ket states
            - For "me": shape (n_times, dim, dim) for density matrices

        Raises
        ------
        ValueError
            If solver is not "se" or "me"
        """
        # Separate drives and couplings
        drives = [(g.which, g.pulse) for g in self.gates if g.type == "drive"]
        couplings = [(g.which, g.pulse) for g in self.gates if g.type == "coupling"]

        # Build time-dependent Hamiltonian
        Ht = self.device.hamiltonian_with_controls(drives=drives, couplings=couplings)

        # Determine time points
        tsave = self.tsave
        if tsave is None:
            tsave = build_tsave_synced_fixed_n(
                drives, couplings, n_points=self.n_points, pad=self.pad
            )

        # Solve dynamics
        if solver == "se":
            res = dq.sesolve(Ht, state0, tsave=tsave, method=Tsit5(max_steps=100_0000))
        elif solver == "me":
            res = dq.mesolve(Ht, self.device._jump_operators(), state0, tsave=tsave)
        else:
            raise ValueError(f"solver must be 'se' or 'me', got '{solver}'")

        t = res.tsave
        states = res.states
        if t is None or states is None:
            raise AttributeError(f"Unknown result fields: {list(res.__dict__.keys())}")

        return t, states

    def final_state(self, state0, solver="se"):
        """
        Simulate the circuit and return only the final state.

        Convenience method for when only the end result is needed.

        Parameters
        ----------
        state0 : StateLike
            Initial quantum state
        solver : Literal["se", "me"], default="se"
            Solver to use

        Returns
        -------
        jnp.ndarray
            Final quantum state after all gates have been applied
        """
        _, states = self.apply(state0, solver=solver)
        return states[-1]

    def project(self, state, projector):
        """
        Compute expectation value <projector> for a quantum state.

        Parameters
        ----------
        state : StateLike
            Quantum state (ket or density matrix)
        projector : QArrayLike
            Observable or projector operator

        Returns
        -------
        float
            Real-valued expectation value Tr(projector * state)
        """
        return jnp.real(dq.expect(projector, state))