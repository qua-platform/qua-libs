import logging
import time
from datetime import datetime
import json
from pathlib import Path
from typing import Any, List, Literal, Optional

from qualibrate.core import QualibrationNode
from qualibrate.core.parameters import RunnableParameters
from qualang_tools.results import progress_counter as _progress_counter
from qualibration_libs.core import BatchableList
from qualibration_libs.parameters.experiment import _make_batchable_list_from_multiplexed, BaseExperimentNodeParameters

from quam_builder.architecture.quantum_dots.components import SensorDot, QuantumDot
from quam_builder.architecture.quantum_dots.operations.names import SingleQubitMacroName
from quam_builder.architecture.quantum_dots.qpu import BaseQuamQD
from quam_builder.architecture.quantum_dots.qubit import AnySpinQubit
from quam_builder.architecture.quantum_dots.qubit_pair import AnySpinQubitPair

__all__ = [
    "QuantumDotExperimentNodeParameters", 
    "VideoModeCommonParameters",
    "get_dots", 
    "get_sensors",
    "get_xy_reference_pulse_name", 
    "quantize_pulse_length_ns",
]

class QuantumDotExperimentNodeParameters(BaseExperimentNodeParameters):
    quantum_dots: Optional[List[str]] = None
    """The virtualised names of the QuantumDots in your VirtualGateSet."""


class VideoModeCommonParameters(RunnableParameters):
    run_in_video_mode: bool = True
    """Optionally open Video Mode with the qualibration node."""
    virtual_gate_set_id: Optional[str] = None
    """Name of the associated VirtualGateSet in your QPU. """
    video_mode_port: int = 8002
    """Localhost port to open VideoMode with"""
    dc_control: bool = False
    """If an associated external DC offset exists."""
    result_type: Literal["I", "Q", "Amplitude", "Phase"] = "I"


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
        raise KeyError(
            f"Qubit '{qubit.id}' is missing the '{SingleQubitMacroName.XY_DRIVE}' macro."
        )

    pulse_name = getattr(xy_drive_macro, "reference_pulse_name", None)
    if pulse_name is None:
        raise ValueError(
            f"Qubit '{qubit.id}' XY-drive macro has no reference_pulse_name configured."
        )
    if pulse_name not in qubit.xy.operations:
        raise KeyError(
            f"Reference pulse '{pulse_name}' is not defined on qubit '{qubit.id}' XY drive."
        )

    return pulse_name


def quantize_pulse_length_ns(pulse_length_ns: int | float) -> int:
    """Round a pulse length to the nearest hardware-valid 4 ns multiple."""
    requested_length_ns = float(pulse_length_ns)
    rounded_length_ns = int(round(requested_length_ns / 4.0)) * 4

    if rounded_length_ns < 4:
        raise ValueError(f"Pulse length must be at least 4 ns, got {pulse_length_ns}.")

    return rounded_length_ns
