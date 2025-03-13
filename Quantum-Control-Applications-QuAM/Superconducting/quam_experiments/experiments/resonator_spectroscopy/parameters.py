from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from quam_experiments.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class T1Parameters(RunnableParameters):
    """
    Parameters for configuring a T1 relaxation time experiment.

    Attributes:
        num_averages (int): Number of averages to perform. Default is 100.
    """

    num_averages: int = 100
    frequency_span_in_mhz: float = 30.0
    frequency_step_in_mhz: float = 0.1


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    T1Parameters,
    QubitsExperimentNodeParameters,
):
    pass
