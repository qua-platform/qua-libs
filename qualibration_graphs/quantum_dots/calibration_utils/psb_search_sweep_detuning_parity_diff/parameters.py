from typing import Optional, List
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    detuning_min: float = -0.1
    """Minimum detuning value for the sweep in volts. Default is -0.1 V."""
    detuning_max: float = 0.1
    """Maximum detuning value for the sweep in volts. Default is 0.1 V."""
    detuning_points: int = 21
    """Number of detuning points to sweep. Default is 21."""
    quantum_dot_pair_names: Optional[List[str]] = None
    """List of quantum dot pair names."""
    ramp_duration: int = 40
    """Ramp duration to ramp to the measurement point."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    pass
