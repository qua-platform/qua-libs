from typing import Literal

import numpy as np
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 50
    """Number of averages to perform. Default is 50."""
    amp_factor: float = 0.2
    """Minimum amplitude factor for the operation. Default is 0.001."""
    cryoscope_len: int = 240
    """Maximum amplitude factor for the operation. Default is 1.99."""
    reset_filters: bool = True
    buffer: int = 10
    num_frames: int = 17


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
