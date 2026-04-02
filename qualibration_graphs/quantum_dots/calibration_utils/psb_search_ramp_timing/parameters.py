from typing import Optional, List
from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    detuning: Optional[float] = None
    """Fixed detuning value. If None, will use default from qubit configuration. Default is None."""
    ramp_duration_min: int = 16
    """Minimum ramp duration. Must be an integer multiple of 4."""
    ramp_duration_max: int = 1000
    """Maximum ramp duration. Must be an integer multiple of 4."""
    ramp_duration_step: int = 4
    """Ramp duration step. Must be an integer multiple of 4."""
    quantum_dot_pair_names: Optional[List[str]] = None
    """List of quantum dot pair names."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    pass
