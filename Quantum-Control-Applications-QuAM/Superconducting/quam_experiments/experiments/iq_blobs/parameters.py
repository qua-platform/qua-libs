from typing import Literal
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from quam_experiments.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    """
    Parameters for configuring a qubit spectroscopy experiment.

    Attributes:
        num_runs (int): Number of runs to perform. Default is 2000.
        operation (str): Type of operation to perform. Default is "readout".
    """

    num_runs: int = 2000
    operation: Literal["readout", "readout_QND"] = "readout"


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
