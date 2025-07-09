from typing import Literal, Optional

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
    number_of_exponents: Literal[1, 2] = 1
    """Number of exponents to use in the cryoscope experiment. One or two, default is 1."""
    exp_1_tau_guess: Optional[float] = None
    """Initial guess for the time constant of the first exponential decay. Default is None."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
