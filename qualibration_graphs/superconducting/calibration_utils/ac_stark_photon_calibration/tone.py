"""Construction of the Stark tone, shared by every node that holds a tone on a resonator.

A node that drives a resonator with the `const` operation while something happens to the qubit has to
settle three things: how long the tone is held before anything is measured, where the tone sits in
frequency, and how a length in nanoseconds becomes a whole number of clock cycles. Those three live
here so that nodes sharing this sequence cannot drift apart in how they build the same tone, which
would turn a real disagreement between their results into an artefact of their code.

Frequencies are in Hz.
"""

import numpy as np

from .analysis import kappa_tot_hz_from_extras

__all__ = [
    "CLOCK_CYCLE_NS",
    "round_up_to_clock_cycle",
    "photon_lifetime_ns",
    "steady_state_fraction",
    "steady_state_pad_ns",
    "tone_intermediate_frequency",
]

CLOCK_CYCLE_NS = 4
"""Length of one OPX clock cycle, in ns. Every pulse length and wait is a multiple of it."""


def round_up_to_clock_cycle(length_ns: float) -> int:
    """Round a length in ns up to the next whole clock cycle."""
    return int(np.ceil(length_ns / CLOCK_CYCLE_NS) * CLOCK_CYCLE_NS)


def photon_lifetime_ns(qubit) -> float:
    """The resonator photon lifetime 1/kappa in ns, with kappa in rad/s.

    This is the time constant of the photon *number*, which is the quantity every other formula in
    these nodes is written in. The field amplitude decays twice as slowly, with time constant
    2/kappa, and that is what makes the pad longer than a first guess suggests. See
    `steady_state_fraction`.
    """
    return 1e9 / (2 * np.pi * kappa_tot_hz_from_extras(qubit))


def steady_state_fraction(lifetimes: float) -> float:
    """Fraction of the steady-state photon number reached after a pad of `lifetimes` 1/kappa.

    The field fills as `1 - exp(-kappa t / 2)`, so after `L` photon lifetimes the amplitude has
    reached `1 - exp(-L/2)` and the photon number, which goes as the square, has reached
    `(1 - exp(-L/2))**2`. Two lifetimes therefore reach 63% of the amplitude but only 40% of the photon
    number. Reaching 96% of the steady-state photon number takes 7.8 lifetimes, which is why the
    default pad is 8.
    """
    if lifetimes <= 0:
        return 0.0
    return float((1 - np.exp(-0.5 * lifetimes)) ** 2)


def steady_state_pad_ns(qubit, lifetimes: float) -> int:
    """How long to hold the tone before the measurement starts, in ns.

    The resonator fills with a field time constant of 2/kappa, so a measurement that starts at the
    same instant as the tone sees a rising photon number rather than the steady-state one. Holding
    the tone for a pad first removes that.

    Parameters
    ----------
    qubit
        Must carry a resonator whose extras hold the linewidth written by node 23a.
    lifetimes : float
        Number of photon lifetimes 1/kappa to wait. Use `steady_state_fraction` to turn that into
        the fraction of the steady-state photon number it reaches: 8 lifetimes give 96%. Zero or
        less disables the pad, which is how the ring-up systematic is reproduced deliberately.

    Returns
    -------
    int
        The pad in ns, rounded up to the clock cycle.
    """
    if lifetimes <= 0:
        return 0
    return round_up_to_clock_cycle(lifetimes * photon_lifetime_ns(qubit))


def tone_intermediate_frequency(qubit, tone_frequency_in_ghz: float = None) -> int:
    """Intermediate frequency at which to play the tone, in Hz.

    With no override the tone sits at the qubit's current readout frequency, so that whatever is
    measured applies to the conditions the qubit is actually read out in. An override is given as an
    absolute RF frequency in GHz and is converted using the resonator's own upconversion.
    """
    resonator = qubit.resonator
    if tone_frequency_in_ghz is None:
        return int(resonator.intermediate_frequency)
    tone_rf_hz = float(tone_frequency_in_ghz) * 1e9
    return int(resonator.intermediate_frequency + (tone_rf_hz - resonator.RF_frequency))
