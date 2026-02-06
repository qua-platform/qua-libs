from typing import Optional, List
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    detuning: Optional[float] = None
    """Fixed detuning value. If None, will use default from qubit configuration. Default is None."""
    quantum_dot_pair_names: Optional[List[str]] = None
    """List of quantum dot pair names."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    pass
