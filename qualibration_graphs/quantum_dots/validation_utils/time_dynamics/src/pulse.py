"""
Pulse abstractions for quantum control.

This module provides a hierarchy of pulse classes for defining time-dependent
control fields in quantum simulations. Each pulse type implements an envelope
function that can be used with quantum device Hamiltonians.

Classes
-------
Pulse : ABC
    Abstract base class for all pulse types
GaussianPulse : Pulse
    Gaussian-enveloped pulse with configurable amplitude and phase
SquarePulse : Pulse
    Constant-amplitude rectangular pulse
CouplingPulse : Pulse
    Time-dependent coupling strength with smooth ramp-up/ramp-down
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import jax.numpy as jnp


# ========= Base =========
@dataclass(frozen=True)
class Pulse(ABC):
    """
    Abstract base class for time-dependent control pulses.

    All pulse subclasses implement an envelope function that defines the
    time-dependent amplitude and phase of the control field. The pulse is
    automatically windowed to be non-zero only within [t0, t0+duration].

    Attributes
    ----------
    t0 : jnp.ndarray
        Start/center time of the pulse in seconds
    duration : jnp.ndarray
        Total duration of the pulse window in seconds

    Methods
    -------
    timecallable()
        Returns a callable function that evaluates the pulse at time t
    _envelope(t)
        Abstract method to be implemented by subclasses. Returns the complex
        or real envelope as a function of time (relative to t0)
    __call__(t)
        Evaluates the windowed pulse at time(s) t
    """

    t0: jnp.ndarray
    duration: jnp.ndarray

    def timecallable(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """
        Return a callable function for this pulse.

        Returns
        -------
        Callable[[jnp.ndarray], jnp.ndarray]
            Function that takes time array and returns pulse values
        """
        return lambda t: self(t)

    @abstractmethod
    def _envelope(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the unwindowed pulse envelope.

        Subclasses must implement this method to define their specific
        envelope shape (Gaussian, square, etc.).

        Parameters
        ----------
        t : jnp.ndarray
            Time relative to pulse center/start

        Returns
        -------
        jnp.ndarray
            Complex or real envelope value(s)
        """
        ...

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the pulse at given time(s) with windowing applied.

        The pulse envelope is computed and then multiplied by a rectangular
        window function that is 1 within [t0, t0+duration] and 0 elsewhere.

        Parameters
        ----------
        t : jnp.ndarray
            Time value(s) at which to evaluate the pulse

        Returns
        -------
        jnp.ndarray
            Windowed pulse value(s), vectorized over input t
        """
        val = self._envelope(t - self.t0)
        t_start = self.t0
        t_end = self.t0 + self.duration
        mask = (t >= t_start) & (t <= t_end)
        val = val * mask.astype(val.dtype)
        return val


# ========= Gaussian =========
@dataclass(frozen=True)
class GaussianPulse(Pulse):
    """
    Gaussian-enveloped pulse with configurable amplitude and phase.

    The pulse has a Gaussian envelope centered at the midpoint of the duration
    window. The envelope shape is:
        env(t) = exp(-0.5 * ((t - duration/2) / (duration/sigma))^2)

    The full pulse including phase is:
        s(t) = amp * env(t) * exp(i * phase)

    Note: The carrier frequency exp(i * ω * t) is handled by the device
    Hamiltonian, not by this pulse class.

    Attributes
    ----------
    amp : jnp.ndarray
        Pulse amplitude in rad/s (Rabi frequency)
    phase : jnp.ndarray, default=0.0
        Pulse phase in radians
    drive_freq : Optional[jnp.ndarray], default=None
        Drive carrier frequency in rad/s (handled by device frame)
    sigma : jnp.ndarray, default=5.0
        Number of standard deviations that fit within the duration.
        Larger sigma means narrower Gaussian. sigma=5 means the duration
        spans ±2.5 standard deviations from center.
    """

    amp: jnp.ndarray
    phase: jnp.ndarray = 0.0
    drive_freq: Optional[jnp.ndarray] = None
    sigma: jnp.ndarray = 5.0

    def _envelope(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Gaussian envelope.

        Parameters
        ----------
        t : jnp.ndarray
            Time relative to pulse start (t0)

        Returns
        -------
        jnp.ndarray
            Complex envelope: amp * gaussian(t) * exp(i * phase)
        """
        # Standard deviation of the Gaussian
        st = self.duration / self.sigma
        # Center of the pulse
        mt = self.duration / 2.0
        # Gaussian envelope
        env = jnp.exp(-0.5 * ((t - mt) / st) ** 2)
        # Full complex envelope (device adds exp(+i ω_eff t) separately)
        return self.amp * env * jnp.exp(1j * self.phase)


# ========= Square =========
@dataclass(frozen=True)
class SquarePulse(Pulse):
    """
    Constant-amplitude rectangular pulse.

    The pulse has a constant complex amplitude within its duration window.
    The envelope is simply:
        s(t) = amp * exp(i * phase)

    Windowing to [t0, t0+duration] is automatically applied by the base
    Pulse.__call__ method.

    Attributes
    ----------
    amp : jnp.ndarray
        Constant pulse amplitude in rad/s (Rabi frequency)
    phase : jnp.ndarray, default=0.0
        Pulse phase in radians
    drive_freq : Optional[jnp.ndarray], default=None
        Drive carrier frequency in rad/s (handled by device frame)
    """

    amp: jnp.ndarray
    phase: jnp.ndarray = 0.0
    drive_freq: Optional[jnp.ndarray] = None

    def _envelope(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Return constant complex amplitude.

        Parameters
        ----------
        t : jnp.ndarray
            Time relative to pulse start (unused for square pulse)

        Returns
        -------
        jnp.ndarray
            Constant complex amplitude: amp * exp(i * phase)
        """
        return self.amp * jnp.exp(1j * self.phase)


@dataclass(frozen=True)
class CouplingPulse(Pulse):
    """
    Time-dependent coupling strength for two-qubit interactions.

    Implements a smooth ramp-up, constant plateau, and ramp-down profile
    for modulating the coupling strength J(t) between two qubits. The total
    duration is divided into three phases:

    1. Ramp-up (duration t_ramp): J goes from 0 to Jmax
    2. Hold (duration = total - 2*t_ramp): J remains at Jmax
    3. Ramp-down (duration t_ramp): J goes from Jmax to 0

    The ramp shape can be either linear or smooth (half-cosine).

    Attributes
    ----------
    Jmax : jnp.ndarray
        Peak coupling strength in rad/s
    t_ramp : jnp.ndarray
        Duration of each ramp (up and down) in seconds
    shape : str, default="cos"
        Ramp shape: "cos" for smooth half-cosine ramps, or "linear"

    Notes
    -----
    The total duration must be at least 2*t_ramp to accommodate both ramps.
    """

    Jmax: jnp.ndarray
    t_ramp: jnp.ndarray
    shape: str = "cos"

    def _ramp(self, x: jnp.ndarray, shape: str = "cos") -> jnp.ndarray:
        """
        Smooth ramp function from 0 to 1.

        Parameters
        ----------
        x : jnp.ndarray
            Normalized time coordinate in [0, 1]
        shape : str
            Ramp shape: "cos" for half-cosine or "linear"

        Returns
        -------
        jnp.ndarray
            Ramp value in [0, 1]
        """
        if shape == "linear":
            return jnp.clip(x, 0.0, 1.0)
        # Half-cosine (smooth) ramp: 0.5 - 0.5*cos(π*x)
        x = jnp.clip(x, 0.0, 1.0)
        return 0.5 - 0.5 * jnp.cos(jnp.pi * x)

    def _envelope(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Compute piecewise coupling strength J(t).

        Parameters
        ----------
        t : jnp.ndarray
            Time relative to pulse start

        Returns
        -------
        jnp.ndarray
            Coupling strength J(t) with smooth ramp-up, hold, ramp-down
        """
        # Time boundaries for the three phases
        t_wait = self.duration - 2 * self.t_ramp
        t2 = self.t_ramp  # end of ramp-up
        t3 = t2 + t_wait  # start of ramp-down
        t4 = t3 + self.t_ramp  # end of ramp-down

        def j_of_t(tt):
            # Ramp up from 0 to 1
            up = self._ramp(tt / jnp.maximum(self.t_ramp, 1e-30), self.shape)
            # Ramp down from 1 to 0
            down = 1.0 - self._ramp((tt - t3) / jnp.maximum(self.t_ramp, 1e-30), self.shape)

            # Piecewise combination of three regions
            j_up = jnp.where(tt < t2, up, 0.0)
            j_hold = jnp.where((tt >= t2) & (tt < t3), 1.0, 0.0)
            j_down = jnp.where((tt >= t3) & (tt <= t4), down, 0.0)

            return self.Jmax * (j_up + j_hold + j_down)

        return j_of_t(t)
