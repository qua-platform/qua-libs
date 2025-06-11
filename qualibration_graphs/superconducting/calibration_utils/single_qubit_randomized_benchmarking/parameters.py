from typing import Optional
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    use_state_discrimination: bool = False
    """Perform qubit state discrimination. Default is True."""
    use_strict_timing: bool = False
    """Use strict timing in the QUA program. Default is False."""
    num_random_sequences: int = 100
    """Number of random RB sequences. Default is 100."""
    num_shots: int = 20
    """Number of averages. Default is 20."""
    max_circuit_depth: int = 1000
    """Maximum circuit depth (number of Clifford gates). Default is 1000."""
    delta_clifford: int = 20
    """Delta clifford (number of Clifford gates between the RB sequences). Default is 20."""
    seed: Optional[int] = None
    """Seed for the random number generator. Default is None."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
