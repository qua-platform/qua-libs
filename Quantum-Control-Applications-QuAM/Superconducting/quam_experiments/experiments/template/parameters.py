from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from quam_experiments.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    """
    Parameters for configuring a # todo ... experiment.

    Attributes:
        # todo: param (type): description. Default
        num_averages (int): Number of averages to perform. Default is 100.
    """

    #  todo:
    #  param: type = default_value
    num_averages: int = 100


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
