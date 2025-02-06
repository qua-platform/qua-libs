from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters

from quam_libs.experiments.node_parameters import (
    QubitsExperimentNodeParameters,
    SimulatableNodeParameters,
    FluxControlledNodeParameters,
    MultiplexableNodeParameters,
    DataLoadableNodeParameters,
    QmSessionNodeParameters
)


class IQBlobsParameters(RunnableParameters):
    num_runs: int = 2000
    operation_name: str = "readout"

class Parameters(
    NodeParameters,
    SimulatableNodeParameters,
    DataLoadableNodeParameters,
    QmSessionNodeParameters,
    IQBlobsParameters,
    FluxControlledNodeParameters,
    MultiplexableNodeParameters,
    QubitsExperimentNodeParameters,
):
    pass
