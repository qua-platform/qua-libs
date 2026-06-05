"""Parameters for JAZZ_ZZ calibration node."""

from typing import ClassVar, Literal

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Parameters specific to JAZZ_ZZ calibration."""

    num_shots: int = 100
    """Number of shots to perform. Default is 100."""
    amp_min: float = -0.1
    """Minimum amplitude for the coupler scan. Default is -0.1."""
    amp_max: float = 0.1
    """Maximum amplitude for the coupler scan. Default is 0.1."""
    amp_step: float = 0.001
    """Step size for amplitude scanning. Default is 0.001."""
    time_min_in_ns: float = 16
    """Minimum free-evolution time in nanoseconds. Default is 16."""
    time_max_in_ns: float = 1000
    """Maximum free-evolution time in nanoseconds. Default is 1000."""
    time_step_in_ns: float = 16
    """Step size for the free-evolution time sweep in nanoseconds. Default is 16."""
    artificial_detuning_in_mhz: float = 1.0
    """Artificial detuning applied during the Ramsey-like sequence in MHz. Default is 1.0."""
    use_state_discrimination: bool = True
    """Whether to use state discrimination for readout. Default is True."""
    measure_qubit: Literal["control", "target"] = "target"
    """Which qubit in the pair to measure: 'control' or 'target'. Default is 'target'."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for JAZZ_ZZ calibration node."""

    targets_name: ClassVar[str] = "qubit_pairs"
