from typing import Literal
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from calibration_utils.heralded_initialization_utils import HeraldedInitializeParameters
from calibration_utils.common_utils.experiment import QuantumDotExperimentNodeParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 2000
    """Number of runs to perform. Default is 2000."""
    operation: Literal["readout", "readout_QND"] = "readout"
    """Type of operation to perform. Default is "readout"."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    HeraldedInitializeParameters,
    QubitsExperimentNodeParameters,
    QubitPairExperimentNodeParameters,
):
    pass
