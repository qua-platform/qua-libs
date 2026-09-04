"""Parameter definitions for Ramsey-based coupler flux long distortion calibration.

This module defines the parameters used in the Ramsey-based coupler flux
distortion calibration experiment using qubit pairs.
"""

from typing import ClassVar, Literal

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Specific parameters for Ramsey-based coupler flux long distortion characterization."""

    num_shots: int = 30
    """Number of shots to acquire."""
    duration_in_ns: int = 8000
    """Maximum duration of the coupler flux pulse in nanoseconds."""
    time_axis: Literal["linear", "log"] = "log"
    """Time axis scale for the delay sweep."""
    time_step_in_ns: int = 48
    """Time step in nanoseconds (used for linear time axis)."""
    time_step_num: int = 100
    """Number of time steps (used for log time axis)."""
    min_wait_time_in_ns: int = 32
    """Minimum delay time in nanoseconds."""
    ramsey_wait_time_in_ns: int = 50
    """Duration of the Ramsey wait window in nanoseconds."""
    ramsey_flux_amplitude_in_v: float = 0.1
    """Amplitude of the short coupler flux probe pulse during the Ramsey window (V)."""
    ramsey_flux_sweep_range_in_v: float = 0.02
    """Half-width in V of the reference amplitude sweep around ramsey_flux_amplitude_in_v (spans [centre - range, centre + range]). Default is 0.02."""
    num_ramsey_flux_points: int = 101
    """Number of amplitude points in the reference Ramsey amplitude sweep."""
    num_frame_rotations: int = 10
    """Number of frame rotation points in the Ramsey sweep."""
    n_exponentials: int = 3
    """Upper bound on the number of exponential components in the flux step
    response fit. The shared `multi_exp_fit_global` automatically drops
    near-degenerate or sub-noise components, so this is a ceiling."""
    update_state: bool = False
    """Whether to update the state. CAUTION: assumes fitting will be acceptable."""
    update_state_from_GUI: bool = False
    """Whether to update the state from the GUI, select when fitting is successful."""
    debug_plots: bool = False
    """If True, also show Ramsey fringe heatmap (time × frame), extracted Ramsey
    phase vs time, and the reference phase-vs-amplitude calibration curve.

    Default figure (always): flux response vs time with the IIR fit overlay.
    """
    coupler_flux_amplitude_in_v: float = 0.1
    """Amplitude of the long coupler flux pulse in Volts."""
    measure_qubit: Literal["control", "target"] = "target"
    """Which qubit to measure: 'control' or 'target'."""
    flux_settle_time_in_ns: int = 100_000
    """Duration of the long coupler flux pulse in nanoseconds.

    Recommended >= 5 x the longest expected tau (in-house heuristic; no paper
    specifies a fixed 5-tau pulse-settle rule). For reference, Hellings et al.
    arXiv:2503.04610 Sec. III samples its drive time up to ~100 us ~ 5 x tau_HP
    (a measurement-window choice, not a pulse-duration rule). With a shorter pulse,
    long-tau components are attenuated by a factor (1 - exp(-T_pulse/tau)),
    which the finite-pulse fit branch in `multi_exp_fit_global` partially
    deconvolves but at the cost of SNR (the fit caps tau at TAU_PULSE_FACTOR=20
    x T_pulse for stability). Default 100 us therefore covers tau up to ~2 ms
    with usable SNR."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for Ramsey-based coupler flux long distortion calibration node."""

    targets_name: ClassVar[str] = "qubit_pairs"
