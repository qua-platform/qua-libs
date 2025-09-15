from typing import Literal, Optional

from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, TwoQubitExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_averages: int = 100
    """Number of averages to perform. Default is 50."""
    amp_range: float = 0.030
    """Range of amplitude variation around the nominal value. Default is 0.030."""
    amp_step: float = 0.001
    """Step size for amplitude scanning. Default is 0.001."""
    num_frames: int = 10
    """Number of frame rotation points for phase measurement. Default is 10."""
    operation: Literal["cz_flattop", "cz_unipolar"] = "cz_unipolar"
    """Type of CZ operation to perform. Options are 'cz_flattop' or 'cz_unipolar'. Default is 'cz_unipolar'."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    TwoQubitExperimentNodeParameters,
):
    pass
