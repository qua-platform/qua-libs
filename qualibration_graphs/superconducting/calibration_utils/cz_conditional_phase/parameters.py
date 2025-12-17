from typing import ClassVar, Literal, Optional

from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_averages: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_range: float = 10
    """Range of amplitude variation around the nominal value. Default is 0.030."""
    frequency_step: float = 0.1
    """Step size for amplitude scanning. Default is 0.001."""
    num_frame_rotations: int = 10
    """Number of frame rotation points for phase measurement. Default is 10."""
    operation: Literal["cz_flattop", "cz_unipolar", "cz"] = "cz"
    """Type of CZ operation to perform. Options are 'cz_flattop' or 'cz_unipolar'. Default is 'cz_unipolar'."""
    use_state_discrimination: bool = True
    """Whether to use state discrimination for readout. Default is True."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    targets_name: ClassVar[str] = "qubit_pairs"
    targets_name: ClassVar[str] = "qubit_pairs"
    targets_name: ClassVar[str] = "qubit_pairs"
