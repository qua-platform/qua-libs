from typing import List

from quam_builder.architecture.quantum_dots.qubit import LDQubit

__all__ = ["elements_list"]

def _find_gates(
    qubit: LDQubit
): 
    """Returns a list of names of all the physical elements in the gate set"""
    qd = qubit.quantum_dot
    gate_set = qd.voltage_sequence.gate_set

    channel_ids = list(gate_set.channels.keys())
    return channel_ids

def _find_readouts(
    qubit: LDQubit
): 
    """Returns the readout resonator object associated with the qubit's readout macro"""
    machine = qubit.machine

    qd_pair_name = machine.find_quantum_dot_pair(
        qubit.quantum_dot.name, qubit.preferred_readout_quantum_dot
    )

    qd_pair = machine.quantum_dot_pairs[qd_pair_name]
    rr = qd_pair.sensor_dots[0].readout_resonator

    return rr

def elements_list(
    qubit: LDQubit
) -> List[str]:
    """Returns a list of all element names in the gate set, along with the first readout_resonator elements associated to the qubit's readout macro"""
    gates = _find_gates(qubit)
    rr = _find_readouts(qubit)
    list_total = [*gates, rr.name]
    return list_total
