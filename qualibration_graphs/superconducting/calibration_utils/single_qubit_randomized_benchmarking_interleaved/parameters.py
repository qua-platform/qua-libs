from typing import Optional, Literal
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters
from quam_config.lru_components import LRU_OPERATION, validate_lru_qubits  # noqa: F401 (validator re-exported)

#: Interleaved "gate" that plays the qubit's f-LRU pump (``qubit.lru``, calibrated by node 26a).
LRU_GATE = LRU_OPERATION

#: Code of the LRU in the QUA gate sequence. The 24 Clifford gates use 0 to 23, so the LRU gets its own
#: code and node 11b can tell it from the Clifford identity, which only waits. Node 11b branches on this code
#: before the Clifford switch, which is ``unsafe`` (a jump table indexed by the code, so codes 0 to 23 only).
LRU_SEQUENCE_CODE = 24

#: Control "gate": the LRU slot with the pump off (same alignments and duration, no virtual Z). Interleaving it
#: gives the cost of the time the LRU takes, so the LRU result minus this one is the cost of the pump itself.
LRU_IDLE_GATE = "lru_idle"
#: The control takes the LRU's slot code (a run interleaves one or the other).
LRU_IDLE_SEQUENCE_CODE = LRU_SEQUENCE_CODE

#: Interleaved gates that are an f-LRU or its control. A run interleaves only one of them, so both play under
#: ``LRU_SEQUENCE_CODE``.
LRU_GATES = (LRU_GATE, LRU_IDLE_GATE)
#: Controls among ``LRU_GATES``: their slot is the LRU's, with nothing played, and nothing is written to the state.
LRU_CONTROL_GATES = (LRU_IDLE_GATE,)


class NodeSpecificParameters(RunnableParameters):
    interleaved_gate_operation: Literal["I", "x180", "y180", "x90", "-x90", "y90", "-y90", "lru", "lru_idle"] = "x180"
    """The single qubit gate to interleave. "lru" plays the qubit's microwave f-LRU pump (``qubit.lru``,
    calibrated and stored by node 26a), which counts as the identity in the Clifford sequence: an ideal
    LRU leaves g and e unchanged and only empties f. "lru_idle" is a control: the same slot with the pump off
    (nothing is written to the state for it). Default is 'x180'."""
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
    measure_leakage: bool = False
    """Read the qubits as g/e/f and fit the leaked population P(f) against depth (leakage RB);
    subtracting the 11a leakage reference gives the leakage of the interleaved gate. Needs
    15_iq_blobs_gef and a thermal or active_gef reset, and reads g/e/f states whatever use_state_discrimination
    says. Writes only
    ``qubit.extras["leakage_<gate>"]`` to the state, next to the gate fidelity. Default is False."""


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
    """Return the Clifford gate index corresponding to the specified gate name.

    The LRU is the identity on the qubit (it only acts on f), so its Clifford index is 0.
    """
    if gate_operation == "I" or gate_operation in LRU_GATES:
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


def get_interleaved_sequence_code(gate_operation) -> int:
    """Return the code stored in the QUA gate sequence for the interleaved gate.

    This is the gate's Clifford index, except for the LRU, which gets ``LRU_SEQUENCE_CODE`` so that it
    is played as the LRU pump rather than as the identity's wait.
    """
    if gate_operation in LRU_GATES:
        return LRU_SEQUENCE_CODE
    return get_interleaved_gate_index(gate_operation)
