from typing import Optional
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
        operation_amplitude_prefactor (Optional[float]): Amplitude pre-factor for the operation. Default is 1.0.
        operation_len_in_ns (Optional[int]): Length of the operation in nanoseconds. Default is None.
        target_peak_width (Optional[float]): Target peak width in Hz. Default is 3e6 Hz.
    """

    num_averages: int = 100
    time_of_flight_in_ns: Optional[int] = 28
    readout_amplitude_in_dBm: Optional[float] = -12
    readout_length_in_ns: Optional[int] = 1000


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
