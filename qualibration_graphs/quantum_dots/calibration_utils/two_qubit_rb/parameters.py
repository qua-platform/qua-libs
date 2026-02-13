"""Parameter definitions for two-qubit randomized benchmarking experiments.

This module defines the parameters used for configuring RB experiments,
including circuit lengths, number of shots, and operation types.
"""

from typing import ClassVar, Literal

from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for two-qubit RB experiments."""

    num_shots: int = 100
    """Number of averages to perform. Default is 50."""
    operation: Literal["cz_flattop", "cz_unipolar", "cz_bipolar"] = "cz_unipolar"
    """Type of CZ operation to perform. Options: 'cz_flattop', 'cz_unipolar', 'cz_bipolar'. Default: 'cz_unipolar'."""
    use_state_discrimination: bool = True
    """Whether to use state discrimination for readout. Default is True."""
    circuit_lengths: list[int] = [1, 4, 16, 32, 64]
    """Circuit lengths (number of Cliffords) to benchmark. Default is (1, 4, 16, 32, 64)."""
    num_circuits_per_length: int = 5
    """Number of random circuits sampled per circuit length. Default is 5."""
    seed: int = 0
    """Random seed for circuit generation to ensure reproducibility. Default is 0."""
    use_input_stream: bool = False
    """Whether to use input streams for circuit execution. Default is False."""
    reset_type: Literal["active", "thermal"] = "active"


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for two-qubit randomized benchmarking experiments."""

    targets_name: ClassVar[str] = "qubit_pairs"
