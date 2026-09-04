"""Parameter definitions for Ramsey-based qubit flux long distortion calibration.

This module defines the parameters used in the Ramsey-based qubit flux
distortion calibration experiment.
"""

from typing import Literal

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Specific parameters for Ramsey-based qubit flux long distortion characterization."""

    num_shots: int = 50
    """Number of shots to acquire."""
    duration_in_ns: int = 5000000
    """Maximum duration of the qubit flux pulse in nanoseconds."""
    time_axis: Literal["linear", "log"] = "log"
    """Time axis scale for the delay sweep."""
    time_step_in_ns: int = 48
    """Time step in nanoseconds (used for linear time axis)."""
    time_step_num: int = 60
    """Number of time steps (used for log time axis)."""
    min_wait_time_in_ns: int = 32
    """Minimum delay time in nanoseconds."""
    ramsey_wait_time_in_ns: int = 1000
    """Duration of the Ramsey wait window in nanoseconds."""
    ramsey_flux_amplitude_in_v: float = 0.012
    """Amplitude of the short qubit flux probe pulse during the Ramsey window (V)."""
    ramsey_flux_sweep_range_in_v: float = 0.003
    """Half-width in V of the reference amplitude sweep around ramsey_flux_amplitude_in_v (spans [centre - range, centre + range]). Default is 0.003."""
    num_ramsey_flux_points: int = 101
    """Number of amplitude points in the reference Ramsey amplitude sweep."""
    num_frame_rotations: int = 20
    """Number of frame rotation points in the Ramsey sweep."""
    qubit_flux_amplitude_in_v: float = 0.1
    """Amplitude of the long qubit flux pulse in Volts."""
    flux_settle_time_in_ns: int = 100_000
    """Duration of the long qubit flux pulse in nanoseconds. Recommended >= 5 x the longest expected tau so that the finite-pulse charging factor (1 - exp(-T_pulse/tau)) >= ~0.99 (in-house heuristic; no paper specifies a fixed 5-tau settle rule)."""
    n_exponentials: int = 2
    """Upper bound on the number of exponential components in the flux step response fit."""
    update_state: bool = False
    """Master gate for writing fitted filters into QUAM state."""
    update_state_from_GUI: bool = False
    """When re-analysing via ``load_data_id``, enable ``update_state`` from the GUI."""
    update_iir: bool = True
    """When ``update_state`` is set, append IIR taps to ``exponential_filter``."""
    debug_plots: bool = False
    """If True, also show Ramsey fringe heatmap (time × frame), extracted Ramsey
    phase vs time, and the reference phase-vs-amplitude calibration curve.

    Default figure (always): flux response vs time with the IIR fit overlay.
    """


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Combined parameters for Ramsey-based qubit flux long distortion calibration node."""

    pass
