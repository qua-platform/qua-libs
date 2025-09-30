from typing import List

from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 5000
    """Number of averages to perform. Default is 50."""
    detuning_target_in_MHz: int = 300
    """Target detuning from sweetspot for the cryoscope pulse in MHz. Default is 350."""
    cryoscope_len: int = 240
    """Length of the cryoscope operation in microseconds. Default is 240."""
    num_frames: int = 17
    """Number of frames to use in the cryoscope experiment. Default is 17."""
    exponential_fit_time_fractions: List[float] = [0.5, 0.01]
    """List of time fractions for the exponential fit. Default is [0.5, 0.01]."""
    update_state_from_GUI: bool = False
    """Whether to update the state from the GUI. Default is False."""
    update_state: bool = False
    """Whether to update the state. Default is False."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
