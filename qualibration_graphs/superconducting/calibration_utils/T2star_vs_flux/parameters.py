"""Parameters for T2*-versus-flux (Ramsey dephasing) characterization."""

from typing import Literal

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for the T2*-versus-flux sweep."""

    num_shots: int = 100
    """Number of averages to perform per (flux, idle-time) point. Default is 100."""
    frequency_detuning_in_mhz: float = 2.0
    """Artificial detuning applied via a virtual-Z (frame) rotation so the Ramsey
    fringes oscillate; the oscillation envelope gives T2*. The extracted T2* does
    not depend on this value (it only needs to be large enough to resolve a few
    fringes within the idle-time window). Default is 2.0 MHz."""

    # --- idle-time sweep ---
    # NOTE: unlike T1/echo (monotonic decays, which use a log sweep), Ramsey fringes
    # must be sampled roughly uniformly (>= ~2 points per fringe), so the default sweep
    # is LINEAR. These fields are owned here (not inherited from IdleTimeNodeParameters)
    # so the linear default is unambiguous; get_idle_times_in_clock_cycles only needs
    # these four attributes to be present.
    min_wait_time_in_ns: int = 16
    """Minimum idle (free-evolution) time in ns. Default is 16."""
    max_wait_time_in_ns: int = 15000
    """Maximum idle (free-evolution) time in ns. Default is 15000."""
    wait_time_num_points: int = 150
    """Number of idle-time points. Default is 150."""
    log_or_linear_sweep: Literal["log", "linear"] = "linear"
    """Idle-time sweep spacing. Default 'linear' (required to sample Ramsey fringes
    without aliasing); 'log' is available but only sensible for very low detuning."""

    # --- flux sweep ---
    flux_span: float = 0.02
    """Full span of the flux-bias sweep in volts, centered on the qubit flux
    point. The sweep runs over ``[-flux_span/2, +flux_span/2]``. Default is 0.02 V."""
    flux_num: int = 11
    """Number of flux-bias points to sample. Default is 11."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Combined parameters for the T2*-versus-flux characterization node."""

    pass
