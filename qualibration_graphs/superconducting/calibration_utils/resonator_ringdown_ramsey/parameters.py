import numpy as np
from pydantic import Field
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 1000
    """Number of averages to perform. Default is 1000."""
    operation: str = "readout"
    """Resonator operation that rings up the resonator. Default is 'readout'."""
    min_wait_time_in_ns: int = Field(default=16, ge=16, multiple_of=4)
    """Minimum delay between the end of the readout pulse and the Ramsey probe, in ns. Default is 16 ns."""
    max_wait_time_in_ns: int = Field(default=3000, gt=16, multiple_of=4)
    """Maximum delay between the end of the readout pulse and the Ramsey probe, in ns. Default is 3000 ns."""
    wait_time_step_in_ns: int = Field(default=80, ge=4, multiple_of=4)
    """Step of the delay sweep, in ns. Default is 80 ns."""
    ramsey_idle_time_in_ns: int = Field(default=200, ge=16, multiple_of=4)
    """Fixed idle time between the two x90 pulses of the Ramsey probe, in ns. Default is 200 ns."""
    num_phase_points: int = Field(default=12, ge=5)
    """Number of points of the phase sweep of the second x90 pulse over one full turn. Default is 12."""
    residual_fraction: float = Field(default=0.01, gt=0, lt=1)
    """The depletion time is the delay after which the residual AC-Stark phase has decayed to this fraction of its
    amplitude, i.e. tau * ln(1 / residual_fraction). Default is 0.01."""
    min_depletion_time_in_ns: int = Field(default=16, ge=16)
    """Lower bound of the depletion time written to the state. Default is 16 ns."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass


def get_wait_times_in_ns(parameters: NodeSpecificParameters) -> np.ndarray:
    """Return the inclusive delay sweep, in ns, on the 4 ns grid."""
    return np.arange(
        parameters.min_wait_time_in_ns,
        parameters.max_wait_time_in_ns + parameters.wait_time_step_in_ns // 2,
        parameters.wait_time_step_in_ns,
        dtype=int,
    )


def get_phases(parameters: NodeSpecificParameters) -> np.ndarray:
    """Return the phases of the second x90 pulse, in turns, without the duplicate end point."""
    return np.arange(parameters.num_phase_points) / parameters.num_phase_points
