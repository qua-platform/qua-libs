from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters, IdleTimeNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 1000
    """Number of averages to perform. Default is 1000."""
    control_amp_range: float = 0.4
    """Amp range for the operation. Default is 0.6."""
    control_amp_step: float = 0.02
    """Amp step for the operation. Default is 0.02."""
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
