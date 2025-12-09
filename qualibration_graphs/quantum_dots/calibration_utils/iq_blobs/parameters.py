from typing import Literal
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
# from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters, QubitPairExperimentNodeParameters
from calibration_utils import QuantumDotExperimentNodeParameters, QubitsExperimentNodeParameters, QubitPairExperimentNodeParameters
from qualibration_libs.parameters import CommonNodeParameters

class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 2000
    """Number of runs to perform. Default is 2000."""
    operation: Literal["readout", "readout_QND"] = "readout"
    """Type of operation to perform. Default is "readout"."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
    QubitPairExperimentNodeParameters
):
    pass
