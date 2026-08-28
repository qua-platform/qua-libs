from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import QuantumDotExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""


class Parameters(
    NodeParameters,
    QuantumDotExperimentNodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    pass
