import numpy as np
from typing import List

from quam_libs.components import Transmon
from quam_libs.experiments.time_of_flight.parameters import Parameters


def _get_flux_bias_offset_and_detuning(qubits: List[Transmon], node_parameters: Parameters):
    if node_parameters.arbitrary_flux_bias is not None:
        arb_flux_bias_offset = {q.name: node_parameters.arbitrary_flux_bias for q in node_parameters.qubits}
        detuning = {q.name: q.freq_vs_flux_01_quad_term * arb_flux_bias_offset[q.name] ** 2 for q in qubits}
    elif node_parameters.arbitrary_qubit_frequency_in_ghz is not None:
        detuning = {q.name: 1e9 * node_parameters.arbitrary_qubit_frequency_in_ghz - q.xy.RF_frequency for q in qubits}
        arb_flux_bias_offset = {q.name: np.sqrt(detuning[q.name] / q.freq_vs_flux_01_quad_term) for q in qubits}
    else:
        arb_flux_bias_offset = {q.name: 0.0 for q in qubits}
        detuning = {q.name: 0.0 for q in qubits}
    return arb_flux_bias_offset, detuning


def _get_pulse_duration(qubits: List[Transmon], node_parameters: Parameters):
    if node_parameters.operation_len_in_ns is not None:
        return {q.name: node_parameters.operation_len_in_ns for q in qubits}
    else:
        return {q.name: q.xy.operations[node_parameters.operation].length + q.z.settle_time for q in qubits}


def _get_pulse_amplitude_scale_factor(node_parameters: Parameters):
    if node_parameters.operation_amplitude_factor is not None:
        return node_parameters.operation_amplitude_factor
    else:
        return 1.0


def get_optional_parameters(qubits: List[Transmon], node_parameters: Parameters):
    arb_flux_bias_offset, detuning = _get_flux_bias_offset_and_detuning(qubits, node_parameters)
    qubit_pulse_duration = _get_pulse_duration(qubits, node_parameters)
    return qubit_pulse_duration, arb_flux_bias_offset, detuning
