from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import QuantumDotExperimentNodeParameters

from typing import List


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    QuantumDotExperimentNodeParameters,
    NodeSpecificParameters,
):
    pass
