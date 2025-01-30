from typing import List

from quam_libs.components import Transmon
from quam_libs.experiments.time_of_flight.parameters import Parameters


def get_optional_pulse_duration(qubits: List[Transmon], node_parameters: Parameters):
    if node_parameters.operation_len_in_ns is not None:
        return {q.name: node_parameters.operation_len_in_ns for q in qubits}
    else:
        return {q.name: q.xy.operations[node_parameters.operation].length for q in qubits}
