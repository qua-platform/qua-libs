from typing import Optional, Literal
import numpy as np
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
        num_averages (int): Number of averages to perform. Default is 100.
        frequency_span_in_mhz (float): Span of frequencies to sweep in MHz. Default is 100 MHz.
        frequency_step_in_mhz (float): Step size for frequency sweep in MHz. Default is 0.25 MHz.
        operation (str): Type of operation to perform. Default is "saturation".
        operation_amplitude_factor (Optional[float]): Amplitude pre-factor for the operation. Default is 1.0.
        operation_len_in_ns (Optional[int]): Length of the operation in nanoseconds. Default is None.
        target_peak_width (Optional[float]): Target peak width in Hz. Default is 3e6 Hz.
    """

    num_averages: int = 100
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 250
    time_step_in_ns: int = 4
    frequency_step_in_mhz: float = 4
    frequency_span_in_mhz: float = 100


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
