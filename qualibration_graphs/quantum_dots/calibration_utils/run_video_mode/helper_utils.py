from pathlib import Path

from qualibrate.core import QualibrationNode

__all__ = [
    "get_axis_names_and_validate",
    "get_quam_state_path",
]


def get_quam_state_path(node: QualibrationNode) -> str:
    return str(Path(node.machine.serialiser._get_state_path()).resolve())


def get_axis_names_and_validate(node: QualibrationNode):
    """In the case that x_axis_name or y_axis_name is None, assign the first and second elements of QDs."""

    quantum_dots = list(node.machine.quantum_dots.keys())
    x_axis_name = node.parameters.x_axis_name
    y_axis_name = node.parameters.y_axis_name

    if node.parameters.x_axis_name is None:
        x_axis_name = quantum_dots[0]

    if node.parameters.y_axis_name is None:
        y_axis_name = quantum_dots[1]

    x_obj = node.machine.get_component(x_axis_name)
    y_obj = node.machine.get_component(y_axis_name)

    if x_obj.voltage_sequence.gate_set.id != y_obj.voltage_sequence.gate_set.id:
        raise ValueError(
            f"X axis and Y axis elements belong to different VirtualGateSet. x: {x_obj.voltage_sequence.gate_set.id}, y: {y_obj.voltage_sequence.gate_set.id}"
        )
    vgs_id = x_obj.voltage_sequence.gate_set.id

    node.namespace["axes_names"] = {"x_axis": x_axis_name, "y_axis": y_axis_name, "gate_set_id": vgs_id}

    return x_axis_name, y_axis_name, vgs_id
