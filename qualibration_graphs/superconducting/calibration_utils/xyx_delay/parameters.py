from typing import Literal

import numpy as np
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 50
    """Number of averages to perform. Default is 50."""
    zeros_before_after_pulse: int = 60
    """Number of zeros before and after the flux pulse to see the rising time."""
    z_pulse_amplitude: float = 0.1
    """Amplitude of the Z pulse to detune the qubit in frequency."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
