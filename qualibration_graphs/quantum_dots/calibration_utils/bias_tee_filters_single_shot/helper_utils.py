from typing import List

from qualibrate.core import QualibrationNode
from qualibrate.core.parameters import RunnableParameters

from quam_builder.architecture.quantum_dots.qpu import BaseQuamQD
from quam.components import Channel


__all__ = ["get_elements"]


def _resolve_gate_set_id(machine: BaseQuamQD, comp) -> str:
    if isinstance(comp, Channel):
        return machine._get_virtual_gate_set(comp).id
    return comp.voltage_sequence.gate_set.id


def _get_elements(machine: BaseQuamQD, node_parameters: RunnableParameters):
    """
    Extract the elements (quantum_dots, barriers, sensor_dots) from the Quam state based on the string names given as node parameters.

    Possible to perform this experiment with physical_channels.
    """
    elements_list = node_parameters.elements
    if elements_list is None:
        elements_list = list(machine.quantum_dots.keys())

    resolved = [machine.get_component(el) for el in elements_list]

    vgs_ids = {_resolve_gate_set_id(machine, comp) for comp in resolved}

    if len(vgs_ids) > 1:
        raise ValueError(f"Elements from more than one gate set found: {vgs_ids}. Use a single gate set.")

    return resolved, next(iter(vgs_ids))


def get_elements(node: QualibrationNode) -> List:
    return _get_elements(node.machine, node.parameters)
