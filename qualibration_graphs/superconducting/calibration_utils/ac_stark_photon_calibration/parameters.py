from typing import List

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

    free_evolution_times_in_ns: List[int] = [200, 500, 800, 1200, 1600, 2000, 2500, 3000, 3500, 4000]
    """Ramsey free evolution times in ns. Only times past 4 / kappa (after the field transient) are fitted,
    and at least 3 are needed there. Default is 10 times from 0.2 to 4 us."""

    num_amplitude_points: int = 11
    """Number of Stark tone amplitudes, including the zero-amplitude reference. Default is 11."""

    max_amplitude_scale: float = 0.3
    """Highest tone amplitude as a scale on the operating readout amplitude, below 2. Measurement-induced
    dephasing kills the fringe long before scale 1, so the operating point is reached by extrapolating
    n_bar against power. Default is 0.3."""

    num_phase_points: int = 21
    """Number of phases of the second pi/2 pulse, spanning one full turn. Default is 21."""

    tone_detuning_in_linewidths: float = 4.0
    """Stark tone detuning above the midpoint of the two dressed resonances (node 24), in units of
    max(|chi|, kappa) per qubit. Far from both resonances the |0> and |1> fields are nearly equal, so the
    fringe survives and F is close to 1. n_bar is converted back to the readout frequency. Default is 4."""

    separate_resonator_core: bool = True
    """Give the resonator its own pulse-processor core for the run, so the tone can overlap the Ramsey.
    Reverted when the node finishes. Default is True."""
