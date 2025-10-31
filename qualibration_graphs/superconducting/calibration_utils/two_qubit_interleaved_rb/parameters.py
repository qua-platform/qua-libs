from typing import ClassVar, Literal

from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 300
    """Number of averages to perform. Default is 50."""
    operation: Literal["cz_flattop", "cz_unipolar"] = "cz_unipolar"
    """Type of CZ operation to perform. Options are 'cz_flattop' or 'cz_unipolar'. Default is 'cz_unipolar'."""
    use_state_discrimination: bool = True
    """Whether to use state discrimination for readout. Default is True."""
    circuit_lengths: tuple[int] = (0, 16, 32, 64, 100)
    """Circuit lengths (number of Cliffords) to benchmark. Default is (0, 1, 4, 16, 32, 50)."""
    num_circuits_per_length: int = 1
    """Number of random circuits sampled per circuit length. Default is 5."""
    basis_gates: list[str] = ["rz", "sx", "x", "cz"]
    """Basis gate set used for circuit compilation. Default is ['rz', 'sx', 'x', 'cz']."""
    reduce_to_1q_cliffords: bool = False
    """If True, reduce interleaved circuits to single-qubit Cliffords for reference runs. Default is False."""
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
    targets_name: ClassVar[str] = "qubit_pairs"
