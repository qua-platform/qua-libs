from typing import ClassVar, Literal, Optional

from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import (
    CommonNodeParameters,
    QubitPairExperimentNodeParameters,
    QubitsExperimentNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_averages: int = 100
    """Number of averages to perform. Default is 100."""
    coupler_pulse_amp: float = 0.1
    delay_range: int = 100
    coupler: str = "qB1-B2"


class Parameters(NodeParameters, CommonNodeParameters, NodeSpecificParameters, QubitsExperimentNodeParameters):
    pass
