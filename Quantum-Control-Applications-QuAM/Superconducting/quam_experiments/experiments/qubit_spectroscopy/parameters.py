from typing import Optional
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from quam_experiments.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class QubitSpectroscopyParameters(RunnableParameters):
    """
    Parameters for configuring a qubit spectroscopy experiment.

    Attributes:
        num_averages (int): Number of averages to perform. Default is 500.
        frequency_span_in_mhz (float): Span of frequencies to sweep in MHz. Default is 100 MHz.
        frequency_step_in_mhz (float): Step size for frequency sweep in MHz. Default is 0.25 MHz.
        operation (str): Type of operation to perform. Default is "saturation".
        operation_amplitude_prefactor (Optional[float]): Amplitude pre-factor for the operation. Default is 1.0.
        operation_len_in_ns (Optional[int]): Length of the operation in nanoseconds. Default is None.
        target_peak_width (Optional[float]): Target peak width in Hz. Default is 3e6 Hz.
    """

    num_averages: int = 500
    frequency_span_in_mhz: float = 100
    frequency_step_in_mhz: float = 0.25
    operation: str = "saturation"
    operation_amplitude_prefactor: float = 1.0
    operation_len_in_ns: Optional[int] = None
    target_peak_width: Optional[float] = 3e6


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    QubitSpectroscopyParameters,
    QubitsExperimentNodeParameters,
):
    pass
