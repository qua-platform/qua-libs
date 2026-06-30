"""Parameters module for CZ leakage amplification calibration."""

# pylint: disable=too-few-public-methods

from typing import ClassVar, Literal

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for CZ leakage amplification."""

    num_shots: int = 100
    """Number of shots to perform. Default is 100."""
    amp_range: float = 0.010
    """Range of amplitude variation around the nominal value (scans center +/- range). Default is 0.010."""
    amp_step: float = 0.001
    """Step size for amplitude scanning. Default is 0.001."""
    operation: Literal["cz_flattop", "cz_unipolar", "cz_bipolar", "cz_flattop_erf", "cz_SNZ"] = "cz_unipolar"
    """Type of CZ operation to perform; one of 'cz_flattop', 'cz_unipolar', 'cz_bipolar', 'cz_flattop_erf', or 'cz_SNZ'. Default is 'cz_unipolar'."""
    number_of_operations: int = 10
    """Number of operations to perform for each amplitude. Default is 10."""
    use_state_discrimination: bool = True
    """Whether to use state discrimination for readout. Default is True because the CZ leakage amplification is only possible with state discrimination."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for CZ leakage amplification calibration."""

    targets_name: ClassVar[str] = "qubit_pairs"
