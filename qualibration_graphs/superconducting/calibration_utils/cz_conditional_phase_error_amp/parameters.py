from typing import Literal, Optional

from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_averages: int = 100
    """Number of averages to perform. Default is 100."""
    amp_range: float = 0.010
    """Range of amplitude variation around the nominal value. Default is 0.010."""
    amp_step: float = 0.001
    """Step size for amplitude scanning. Default is 0.001."""
    num_frame_rotations: int = 10
    """Number of frame rotation points for phase measurement. Default is 10."""
    operation: Literal["cz_flattop", "cz_unipolar", "cz_bipolar"] = "cz_unipolar"
    """Type of CZ operation to perform. Options are 'cz_flattop', 'cz_unipolar', or 'cz_bipolar'. Default is 'cz_unipolar'."""
    number_of_operations: int = 10
    """Number of operations to perform for each amplitude. Default is 10."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    pass
