from typing import Optional, Literal
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    interleaved_gate_operation: Literal["I", "x180", "y180", "x90", "-x90", "y90", "-y90"] = "x180"
    """The single qubit gate to interleave. Default is 'x180'."""
    use_state_discrimination: bool = False
    """Perform qubit state discrimination. Default is True."""
    use_strict_timing: bool = False
    """Use strict timing in the QUA program. Default is False."""
    num_random_sequences: int = 100
    """Number of random RB sequences. Default is 100."""
    num_shots: int = 20
    """Number of averages. Default is 20."""
    max_circuit_depth: int = 1000
    """Maximum circuit depth (number of Clifford gates). Default is 1000."""
    delta_clifford: int = 20
    """Delta clifford (number of Clifford gates between the RB sequences). Default is 20."""
    seed: Optional[int] = None
    """Seed for the random number generator. Default is None."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass


def get_interleaved_gate_name(gate_index: int) -> str:
    """Return the name of the gate based on its Clifford index."""
    if gate_index == 0:
        return "I"
    elif gate_index == 1:
        return "x180"
    elif gate_index == 2:
        return "y180"
    elif gate_index == 12:
        return "x90"
    elif gate_index == 13:
        return "-x90"
    elif gate_index == 14:
        return "y90"
    elif gate_index == 15:
        return "-y90"
    else:
        raise ValueError(f"Interleaved gate index {gate_index} doesn't correspond to a single operation")


def get_interleaved_gate_index(gate_operation) -> int:
    """Return the Clifford gate index corresponding to the specified gate name."""
    if gate_operation == "I":
        return 0
    elif gate_operation == "x180":
        return 1
    elif gate_operation == "y180":
        return 2
    elif gate_operation == "x90":
        return 12
    elif gate_operation == "-x90":
        return 13
    elif gate_operation == "y90":
        return 14
    elif gate_operation == "-y90":
        return 15
    else:
        raise ValueError(f"Gate operation {gate_operation} not recognized")
