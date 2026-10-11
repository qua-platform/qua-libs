from typing import List, Optional

from qualibrate import NodeParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
):
    """Combined parameter class for the AC Stark photon calibration node."""

    num_shots: int = 1000
    """Number of averages to perform. Default is 1000."""

    free_evolution_times_in_ns: List[int] = [20, 40, 60, 90, 120, 160, 240, 400]
    """Ramsey free evolution times in ns. The Stark shift is the slope of the fringe phase against them, so
    they should reach past a few photon lifetimes 1/kappa. Times where 2 chi tau exceeds 1 rad are dropped
    per qubit. Default is 8 times from 20 to 400 ns."""

    num_amplitude_points: int = 11
    """Number of Stark tone amplitudes, including the zero-amplitude reference. Default is 11."""

    max_amplitude_scale: float = 0.3
    """Highest tone amplitude as a scale on the operating readout amplitude, below 2. Measurement-induced
    dephasing kills the fringe long before scale 1, so the operating point is reached by extrapolating
    n_bar against power. Default is 0.3."""

    num_phase_points: int = 21
    """Number of phases of the second pi/2 pulse, spanning one full turn. Default is 21."""

    tone_frequency_in_ghz: Optional[float] = None
    """Absolute Stark tone frequency in GHz. None uses the qubit's readout frequency. Default is None."""

    separate_resonator_core: bool = True
    """Give the resonator its own pulse-processor core for the run, so the tone can overlap the Ramsey.
    Reverted when the node finishes. Default is True."""
