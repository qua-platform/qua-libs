from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from quam_experiments.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    """
    Parameters for configuring a resonator spectroscopy experiment.

    Attributes:
        num_averages (int): Number of averages to perform. Default is 100.
        frequency_span_in_mhz (float): Span of frequencies to sweep in MHz. Default is 30 MHz.
        frequency_step_in_mhz (float): Step size for frequency sweep in MHz. Default is 0.1 MHz.
    """

    num_averages: int = 100
    frequency_span_in_mhz: float = 30.0
    frequency_step_in_mhz: float = 0.1


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
