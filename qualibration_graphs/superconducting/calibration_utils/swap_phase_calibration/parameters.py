from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters, IdleTimeNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 1000
    """Number of averages to perform. Default is 1000."""
    phase_min: float = 0.0
    """Phase min for the operation. Default is 0.0."""
    phase_max: float = 1.0
    """Phase max for the operation. Default is 0.0."""
    phase_steps_number: int = 51
    """Number of phase steps for the operation. Default is 51."""
    use_state_discrimination: bool = False
    """Perform qubit state discrimination. Default is False."""



class Parameters(
    NodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass