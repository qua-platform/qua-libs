from typing import Optional, Literal, Union, List
import numpy as np
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, TwoQubitExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """
    Parameters for configuring a qubit spectroscopy experiment.

    Attributes:
        num_shots (int): Number of averages to perform. Default is 100.
        frequency_span_in_mhz (float): Span of frequencies to sweep in MHz. Default is 100 MHz.
        frequency_step_in_mhz (float): Step size for frequency sweep in MHz. Default is 0.25 MHz.
        operation (str): Type of operation to perform. Default is "saturation".
        operation_amplitude_factor (Optional[float]): Amplitude pre-factor for the operation. Default is 1.0.
        operation_len_in_ns (Optional[int]): Length of the operation in nanoseconds. Default is None.
        target_peak_width (Optional[float]): Target peak width in Hz. Default is 3e6 Hz.
    """

    num_shots: int = 100
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 1600
    time_step_in_ns: int = 16
    cr_type: Literal["direct", "direct+cancel", "direct+echo", "direct+cancel+echo"] = "direct"
    cr_drive_amp_scaling: Union[float, List[float]] = 1.0
    cr_drive_phase: Union[float, List[float]] = 1.0
    cr_cancel_amp_scaling: Union[float, List[float]] = 0.0
    cr_cancel_phase: Union[float, List[float]] = 0.0
    wf_type: Literal["square", "cosine", "gauss", "flattop"] = "square"


class TwoQubitExperimentNodeParametersCustom(TwoQubitExperimentNodeParameters):
    use_state_discrimination: bool = True


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    # QubitsExperimentNodeParameters,
    TwoQubitExperimentNodeParametersCustom,
):
    pass
