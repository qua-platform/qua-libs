from typing import Dict

from qualibrate.core import QualibrationNode
from qualibration_libs.parameters import get_qubits

from qualang_tools.units import unit

__all__ = ["resolve_operation_name", "get_durations_and_chirp_rates"]


def resolve_operation_name(node: QualibrationNode, operation: str = "x180") -> str:
    """
    08a_qubit_spectroscopy_chirp.py requires the use of a chirped XY drive.

    Currently, chirped XY drive is not supported with the macros infrastructure. Therefore, this
    function helps to resolve the operation name for the machine's pulse_family.
    """

    # Extract the pulse family from the node's machine
    pulse_family = node.machine.pulse_family
    operation_name = f"{pulse_family}_{operation}"

    active_qubits = get_qubits(node)

    # Quick check through the actively measured qubits to ensure that the operation exists in the dict of the XY component.
    for qubit in active_qubits:
        operations_dict = qubit.xy.operations
        if operation_name not in operations_dict:
            raise ValueError(
                f"Qubit {qubit}'s XY drive has not operation named {operation_name}. Please double check the pulse family and/or the Quam state"
            )

    return operation_name


def get_durations_and_chirp_rates(node: QualibrationNode, operation: str = "x180"):
    """
    If a operation_len_in_ns is given, then standardises the qubit operation lengths in a dictionary. Otherwise,
    populates this dictionary with the default lengths.

    For each qubit's desired pulse length, calculates the required chirp rate in Hz/ns and returns this in a dict. These
    values should be directly entered into the .play() command (remember to divide the operation len by //4 to convert to clock cycles).


    Args:
        - node: QualibrationNode, the active node (08a)
        - operation: The string name of the desired operation.

    Returns:
        - A dict of qubit_name : operation length in ns
        - A dict of qubit_name : necessary chirp rates to cover the required frequency span in the desired length

    """

    u = unit(coerce_to_integer=True)

    operation_lengths_by_qubit = {}
    chirp_rates_by_qubit = {}
    frequency_step_size = node.parameters.frequency_step_in_mhz * u.MHz

    qubits = get_qubits(node)
    operation_len = node.parameters.operation_len_in_ns

    for q in qubits:
        # If no operation_len is given, get a dict of the operation lengths
        if operation_len is None:
            pulse_name = resolve_operation_name(node, operation)
            op_len = q.xy.operations[pulse_name].length
        # If an operation_len is given, standardise them in the same dict
        else:
            op_len = operation_len

        if op_len % 4 != 0:
            raise ValueError(f"Operation length for qubit {q.name} must be a multiple of 4 ns, got {op_len} ns.")
        operation_lengths_by_qubit[q.name] = op_len

    # Once we have the desired durations per qubit, work out the required chirp rates based on each duration
    for q in qubits:
        desired_len = operation_lengths_by_qubit[q.name]
        chirp_rate_in_hz_per_ns = frequency_step_size / desired_len
        chirp_rates_by_qubit[q.name] = chirp_rate_in_hz_per_ns

    return (operation_lengths_by_qubit, chirp_rates_by_qubit)
