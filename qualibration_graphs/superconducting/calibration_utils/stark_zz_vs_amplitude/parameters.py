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

    num_shots: int = 3
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 250
    time_step_in_ns: int = 16
    # zz_control_amp_scalings: Union[float, List[float]] = 1.0
    # zz_target_amp_scalings: Union[float, List[float]] = 1.0
    # zz_control_phases: Union[float, List[float]] = 0.0
    # zz_target_phases: Union[float, List[float]] = 0.0
    min_zz_drive_amp_scaling: float = 0.0
    max_zz_drive_amp_scaling: float = 1.0
    step_zz_drive_amp_scaling: float = 0.1
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
