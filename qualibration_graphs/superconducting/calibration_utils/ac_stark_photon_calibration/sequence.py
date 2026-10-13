"""Helpers that build the Stark-Ramsey sequence of node 25 and check it before the run."""

import numpy as np
from qualibrate import QualibrationNode

from .analysis import (
    MIN_CONTRAST_FRACTION,
    MIN_TAU_IN_LIFETIMES,
    PHASE_STEP_WARNING_FRACTION_OF_PI,
    gamma_per_photon,
    kappa_tot_hz_from_extras,
    photon_number_response,
    stark_factor,
)
from .tone import CLOCK_CYCLE_NS, round_up_to_clock_cycle

__all__ = ["MIN_WAIT_CYCLES", "assign_separate_core", "free_evolution_times_ns", "warn_on_sequence_feasibility"]

MIN_WAIT_CYCLES = 4
"""The shortest wait the OPX accepts, in clock cycles."""


def _element_core(channel) -> str:
    """The pulse-processor core a channel is pinned to, whichever field carries it.

    QUAM accepts the name under `core` and, deprecated since qm-qua 1.2.2, under `thread`. They are
    mutually exclusive, so the field already in use is the one to write back to.
    """
    core = getattr(channel, "core", None)
    return core if core is not None else getattr(channel, "thread", None)


def assign_separate_core(tracked_resonator, qubit, node: QualibrationNode) -> None:
    """Move the resonator to its own core so the tone can overlap the Ramsey.

    Elements sharing a core cannot play overlapping pulses: the resonator tone is issued first and
    the pi/2 that follows is pushed until it finishes, so the qubit only ever sees the ring-down.
    The change is tracked and reverted with everything else.
    """
    xy_core = _element_core(qubit.xy)
    resonator_core = _element_core(qubit.resonator)
    if xy_core is None or xy_core != resonator_core:
        return
    if not node.parameters.separate_resonator_core:
        node.log(
            f"WARNING {qubit.name}: its drive and resonator share core '{xy_core}', so the tone cannot "
            f"overlap the Ramsey sequence and the measured phase will not grow with the free evolution "
            f"time. separate_resonator_core is off, so this is left as it is."
        )
        return
    new_core = f"{qubit.name}_tone"
    if getattr(qubit.resonator, "core", None) is not None:
        tracked_resonator.core = new_core
    else:
        tracked_resonator.thread = new_core
    node.log(f"{qubit.name}: resonator moved from core '{xy_core}' to '{new_core}' for the duration of the run.")


def free_evolution_times_ns(node: QualibrationNode) -> list:
    """The free evolution times, rounded to the clock cycle and long enough for the OPX to wait."""
    minimum_ns = MIN_WAIT_CYCLES * CLOCK_CYCLE_NS
    times = sorted({max(round_up_to_clock_cycle(t), minimum_ns) for t in node.parameters.free_evolution_times_in_ns})
    if len(times) < 3:
        raise ValueError(
            f"at least 3 distinct free evolution times are needed to fit a slope with a free intercept, "
            f"got {times}. The Stark shift is that slope."
        )
    return times


def warn_on_sequence_feasibility(node, qubit, amps: np.ndarray, taus_ns: list, tone_hz: float) -> None:
    """Warn before the run when too few times pass the transient, the fringe will be dead there, or the
    phase will alias. The last two need a photon calibration from an earlier run."""
    kappa_hz, chi_hz = kappa_tot_hz_from_extras(qubit), float(qubit.chi)
    min_tau_ns = 1e9 * MIN_TAU_IN_LIFETIMES / (2 * np.pi * kappa_hz)
    if sum(t >= min_tau_ns for t in taus_ns) < 3:
        node.log(
            f"WARNING {qubit.name}: fewer than 3 free evolution times reach {MIN_TAU_IN_LIFETIMES:g}/κ = "
            f"{min_tau_ns:.0f} ns, so the fit will fail. Add longer times."
        )
    extras = getattr(qubit.resonator, "extras", None) or {}
    photons_per_mw = extras.get("photons_per_mw")
    if not photons_per_mw:
        node.log(f"{qubit.name}: no earlier photon calibration, so the sweep is only checked after the fit.")
        return
    # The stored calibration is at the readout frequency; scale it to the tone frequency.
    f_ground, f_excited = float(extras["f_r_ground_hz"]), float(extras["f_r_excited_hz"])
    to_tone = photon_number_response(tone_hz, f_ground, f_excited, kappa_hz) / photon_number_response(
        float(qubit.resonator.RF_frequency), f_ground, f_excited, kappa_hz
    )
    chi_rad, kappa_rad = 2 * np.pi * chi_hz, 2 * np.pi * kappa_hz
    detuning_rad = 2 * np.pi * (tone_hz - 0.5 * (f_ground + f_excited))
    max_power_mw = 10 ** ((qubit.resonator.get_output_power("readout") + 20 * np.log10(max(amps))) / 10)
    n_bar_max = float(photons_per_mw) * to_tone * max_power_mw

    surviving = np.exp(-gamma_per_photon(chi_rad, kappa_rad, detuning_rad) * n_bar_max * min_tau_ns * 1e-9)
    if surviving < MIN_CONTRAST_FRACTION:
        node.log(
            f"WARNING {qubit.name}: at amplitude scale {max(amps):.2f} the earlier calibration predicts "
            f"n̄ = {n_bar_max:.2f} at the tone, leaving {100 * surviving:.1f}% of the fringe by {min_tau_ns:.0f} ns. "
            f"Lower max_amplitude_scale or increase tone_detuning_in_linewidths."
        )
    steps = np.diff([0.0] + list(taus_ns)) * 1e-9
    stark_rad = abs(2 * chi_rad * stark_factor(chi_rad, kappa_rad, detuning_rad) * n_bar_max)
    largest_step = stark_rad * float(np.max(steps)) / np.pi
    if largest_step >= PHASE_STEP_WARNING_FRACTION_OF_PI:
        node.log(
            f"WARNING {qubit.name}: the fringe phase is expected to grow by {largest_step:.2f} π between "
            f"adjacent free evolution times. The phase is unwrapped along that axis, which only holds below "
            f"π, so add closer-spaced times or lower max_amplitude_scale."
        )
