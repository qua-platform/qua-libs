from typing import Dict

from qualibration_libs.parameters.experiment import BaseExperimentNodeParameters, _make_batchable_list_from_multiplexed
from qualibrate.core import QualibrationNode
from qualibration_libs.core import BatchableList

from quam_builder.architecture.quantum_dots.qpu import BaseQuamQD
from quam_builder.architecture.quantum_dots.components import QuantumDotPair, SensorDot

__all__ = [
    "get_dot_pairs",
    "get_dot_pair_sensors",
]

def _get_dot_pairs(machine: BaseQuamQD, node_parameters : BaseExperimentNodeParameters):
    if node_parameters.quantum_dot_pairs is None or node_parameters.quantum_dot_pairs == "":
        dot_pairs = list(machine.quantum_dot_pairs.values())
    else:
        dot_pairs = [machine.quantum_dot_pairs[qdp] for qdp in node_parameters.quantum_dot_pairs]
    return dot_pairs


def get_dot_pairs(node: QualibrationNode) -> BatchableList[QuantumDotPair]:
    dots = _get_dot_pairs(node.machine, node.parameters)
    dots_batchable_list = _make_batchable_list_from_multiplexed(dots, False ) # node.parameters.multiplexed)
    return dots_batchable_list

def get_dot_pair_sensors(node: QualibrationNode) -> Dict[str, BatchableList[SensorDot]]: 
    dot_pairs = get_dot_pairs(node)

    all_sensors = {
        pair.name: _make_batchable_list_from_multiplexed(pair.sensor_dots, True) for pair in dot_pairs
    }

    return all_sensors
