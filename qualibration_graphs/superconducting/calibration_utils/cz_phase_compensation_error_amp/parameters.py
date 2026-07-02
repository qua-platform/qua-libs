"""Parameters for CZ phase compensation with error amplification calibration."""

# pylint: disable=too-few-public-methods

from typing import ClassVar, Literal

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for CZ phase compensation with error amplification."""

    num_shots: int = 100
    """Number of shots to perform. Default is 100."""
    frame_range: float = 0.1
    """Range of frame rotation to sweep. Default is 0.1."""
    num_frames: int = 17
    """Number of phase frames to sweep. Default is 17."""
    number_of_operations: int = 8
    """Number of CZ operations applied per amplification step. Default is 8."""
    operation: Literal["cz_flattop", "cz_unipolar", "cz_bipolar", "cz_flattop_erf", "cz_SNZ"] = "cz_unipolar"
    """Type of CZ operation to perform. Options are 'cz_flattop', 'cz_unipolar', 'cz_bipolar', 'cz_flattop_erf', or 'cz_SNZ'. Default is 'cz_unipolar'."""
    use_state_discrimination: bool = True
    """Whether to use state discrimination for readout. Default is True."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters class for CZ phase compensation with error amplification."""

    targets_name: ClassVar[str] = "qubit_pairs"
