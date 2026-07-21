"""Parameter definitions for T1 ADE tracking experiments."""

from typing import List, Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for T1 ADE experiments."""

    qubits: Optional[List[str]] = ["qA1"]
    """Qubits to measure. Default is ``["qA1"]``."""
    num_repetitions: int = 500
    """Number of ADE repetitions (lab-time tracking length). Default is 500."""
    t0_ns: int = 16
    """Shortest wait time t0 in nanoseconds. Default is 16 ns."""
    t1_guess_us: float = 40.0
    """Initial T1 guess in microseconds; sets initial dt = alpha * t1_guess_us. Default is 40 us."""
    alpha: float = 1.0
    """Wait-scale factor when adaptive_dt is True. Default is 1.0."""
    n_avg_per_point: int = 50
    """Shots averaged per delay; enters sigma_P = sqrt(P(1-P)/n_avg). Default is 50."""
    n_bootstrap: int = 300
    """Host bootstrap resamples per repetition for sigma_boot. Default is 300."""
    adaptive_dt: bool = False
    """If True, adapt dt for the next repetition from the running gamma estimate. Default is False."""
    min_dt_ns: int = 16
    """Minimum adaptive wait dt in nanoseconds. Default is 16 ns."""
    max_dt_ns: int = 150_000
    """Maximum adaptive wait dt in nanoseconds. Default is 150_000 ns."""
    reset_max_attempts: int = 15
    """Maximum active-reset attempts before giving up. Default is 15."""
    simulation_duration_ns: int = 2500
    """Simulation duration in nanoseconds. Default is 2500 ns."""
    timeout: int = 100
    """QOP session timeout in seconds. Default is 100 s."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
    NodeSpecificParameters,
):
    """Combined parameters for T1 ADE tracking experiments."""
