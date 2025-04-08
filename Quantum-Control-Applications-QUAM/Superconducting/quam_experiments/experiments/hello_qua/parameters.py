from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from quam_experiments.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_averages: int = 100
    """Number of averages to perform. Default is 100."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
