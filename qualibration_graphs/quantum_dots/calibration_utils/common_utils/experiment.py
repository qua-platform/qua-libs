from typing import List, Sequence, Optional

from qualibrate.core import QualibrationNode
from qualibration_libs.core import BatchableList
from qualibration_libs.parameters.experiment import _make_batchable_list_from_multiplexed, BaseExperimentNodeParameters

from quam_builder.architecture.quantum_dots.qpu import BaseQuamQD
from quam_builder.architecture.quantum_dots.qubit import AnySpinQubit
from quam_builder.architecture.quantum_dots.components import SensorDot, QuantumDot
from quam_builder.architecture.quantum_dots.operations.names import SingleQubitMacroName

__all__ = [
    "QuantumDotExperimentNodeParameters",
    "get_dots",
    "get_sensors",
    "get_xy_reference_pulse_name",
    "quantize_pulse_length_ns",
    "ensure_single_gate_set",
]


class QuantumDotExperimentNodeParameters(BaseExperimentNodeParameters):
    quantum_dots: Optional[List[str]] = None
    """The virtualised names of the QuantumDots in your VirtualGateSet."""


def _get_dots(machine: BaseQuamQD, node_parameters: QuantumDotExperimentNodeParameters):
    if node_parameters.quantum_dots is None or node_parameters.quantum_dots == "":
        dots = list(machine.quantum_dots.values())
    else:
        dots = [machine.quantum_dots[s] for s in node_parameters.quantum_dots]
    return dots


def get_dots(node: QualibrationNode) -> BatchableList[QuantumDot]:
    dots = _get_dots(node.machine, node.parameters)
    dots_batchable_list = _make_batchable_list_from_multiplexed(dots, True)
    return dots_batchable_list


def _get_sensors(machine: BaseQuamQD, node_parameters: BaseExperimentNodeParameters):
    if node_parameters.sensor_names is None or node_parameters.sensor_names == "":
        sensors = list(machine.sensor_dots.values())
    else:
        sensors = [machine.sensor_dots[s] for s in node_parameters.sensor_names]
    return sensors


def get_sensors(node: QualibrationNode) -> BatchableList[SensorDot]:
    sensors = _get_sensors(node.machine, node.parameters)

    if isinstance(node.parameters, BaseExperimentNodeParameters):
        multiplexed = node.parameters.multiplexed
    else:
        multiplexed = False

    sensors_batchable_list = _make_batchable_list_from_multiplexed(sensors, multiplexed)

    return sensors_batchable_list


def get_xy_reference_pulse_name(qubit: AnySpinQubit) -> str:
    """Resolve the pulse name backing the qubit's default XY macros."""
    if qubit.xy is None:
        raise ValueError(f"Qubit '{qubit.id}' has no XY drive configured.")

    xy_drive_macro = qubit.macros.get(SingleQubitMacroName.XY_DRIVE)
    if xy_drive_macro is None:
        raise KeyError(f"Qubit '{qubit.id}' is missing the '{SingleQubitMacroName.XY_DRIVE}' macro.")

    pulse_name = getattr(xy_drive_macro, "reference_pulse_name", None)
    if pulse_name is None:
        raise ValueError(f"Qubit '{qubit.id}' XY-drive macro has no reference_pulse_name configured.")
    if pulse_name not in qubit.xy.operations:
        raise KeyError(f"Reference pulse '{pulse_name}' is not defined on qubit '{qubit.id}' XY drive.")

    return pulse_name


def quantize_pulse_length_ns(pulse_length_ns: int | float) -> int:
    """Round a pulse length to the nearest hardware-valid 4 ns multiple."""
    requested_length_ns = float(pulse_length_ns)
    rounded_length_ns = int(round(requested_length_ns / 4.0)) * 4

    if rounded_length_ns < 4:
        raise ValueError(f"Pulse length must be at least 4 ns, got {pulse_length_ns}.")

    return rounded_length_ns


def ensure_single_gate_set(machine: BaseQuamQD, elements: Sequence, reset_with_voltage_tracking: bool = False) -> str:
    """
    Given a list of elements (SensorDots, QuantumDots, Qubits, QubitPairs, QuantumDotPairs, BarrierGate),
    find the associated VirtualGateSet, and validate that there is only one. For multiple VirtualGateSets,
    raises an error. Multi-VirtualGateSet functionality will be added in a future update.

    Optionally also reset the VoltageSequence associated with this VirtualGateSet with voltage tracking.
    """
    set_of_gate_sets = set()
    for el in elements:
        # Assume that the element is SensorDot, QuantumDot, Qubit, QubitPair, QuantumDotPair, or BarrierGate
        gate_set_id = el.voltage_sequence.gate_set.name

        # If the element list is a list of physical_channel objects, this will be implemented in a future version
        set_of_gate_sets.add(gate_set_id)

    number_of_gate_sets = len(set_of_gate_sets)

    if number_of_gate_sets > 1:
        raise NotImplementedError(
            f"Recieved elements from {number_of_gate_sets} VirtualGateSets. Please run this node with a elements from a single VirtualGateSet."
        )
    elif number_of_gate_sets == 0:
        raise ValueError("Zero VirtualGateSets found. Please identify some elements")

    vgs_id = next(iter(set_of_gate_sets))

    # Optional VoltageSequence reset
    if reset_with_voltage_tracking:
        machine.reset_voltage_sequence(
            gate_set_id=vgs_id,
            track_integrated_voltage=True,
        )

    return vgs_id
