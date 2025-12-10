from typing import List, Literal, Optional

from qualibrate import QualibrationNode
from qualibrate.parameters import RunnableParameters
from qualibration_libs.core import BatchableList

from quam_builder.architecture.quantum_dots.components import SensorDot, QuantumDot
from quam_builder.architecture.quantum_dots.qpu import BaseQuamQD
from quam_builder.architecture.quantum_dots.qubit import AnySpinQubit
from quam_builder.architecture.quantum_dots.qubit_pair import AnySpinQubitPair


class BaseExperimentNodeParameters(RunnableParameters):
    multiplexed: bool = False
    """Whether to play control pulses, readout pulses and active/thermal reset at the same time for all qubits (True)
    or to play the experiment sequentially for each qubit (False). Default is False."""
    use_state_discrimination: bool = False
    """Whether to use on-the-fly state discrimination and return the qubit 'state', or simply return the demodulated
    quadratures 'I' and 'Q'. Default is False."""
    reset_wait_time: int = 5000
    """The wait time for qubit reset."""
    sensor_names: Optional[List[str]] = None
    """The list of sensor dot names to be included in the measurement. """

class QuantumDotExperimentNodeParameters(BaseExperimentNodeParameters):
    quantum_dots: Optional[List[str]] = None
    """The virtualised names of the QuantumDots in your VirtualGateSet."""

class QubitsExperimentNodeParameters(BaseExperimentNodeParameters):
    qubits: Optional[List[str]] = None
    """A list of qubit names which should participate in the execution of the node. Default is None."""


class QubitPairExperimentNodeParameters(BaseExperimentNodeParameters):
    qubit_pairs: Optional[List[str]] = None
    """A list of qubit pair names which should participate in the execution of the node. Default is None."""


def _make_batchable_list_from_multiplexed(items: List, multiplexed: bool) -> BatchableList:
    if multiplexed:
        batched_groups = [[i for i in range(len(items))]]
    else:
        batched_groups = [[i] for i in range(len(items))]

    return BatchableList(items, batched_groups)

def _get_dots(machine:BaseQuamQD, node_parameters: QuantumDotExperimentNodeParameters):
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


def get_qubits(node: QualibrationNode) -> BatchableList[AnySpinQubit]:
    qubits = _get_qubits(node.machine, node.parameters)

    if isinstance(node.parameters, QubitsExperimentNodeParameters):
        multiplexed = node.parameters.multiplexed
    else:
        multiplexed = False

    qubits_batchable_list = _make_batchable_list_from_multiplexed(qubits, multiplexed)

    return qubits_batchable_list


def _get_qubits(machine: BaseQuamQD, node_parameters: QubitsExperimentNodeParameters) -> List[AnySpinQubit]:
    if node_parameters.qubits is None or node_parameters.qubits == "":
        qubits = machine.active_qubits
    else:
        qubits = [machine.qubits[q] for q in node_parameters.qubits]

    return qubits


def get_qubit_pairs(node: QualibrationNode) -> BatchableList[AnySpinQubitPair]:
    qubit_pairs = _get_qubit_pairs(node.machine, node.parameters)

    if isinstance(node.parameters, QubitPairExperimentNodeParameters):
        multiplexed = node.parameters.multiplexed
    else:
        multiplexed = False

    qubit_pairs_batchable_list = _make_batchable_list_from_multiplexed(qubit_pairs, multiplexed)

    return qubit_pairs_batchable_list


def _get_qubit_pairs(machine: BaseQuamQD, node_parameters: QubitPairExperimentNodeParameters) -> List[AnySpinQubitPair]:
    if node_parameters.qubit_pairs is None or node_parameters.qubit_pairs == "":
        qubit_pairs = machine.active_qubit_pairs
    else:
        qubit_pairs = [machine.qubit_pairs[q] for q in node_parameters.qubit_pairs]

    return qubit_pairs


import xarray as xr

def convert_IQ_to_V(
    da: xr.Dataset,
    qubits: list[AnySpinQubit] = None,
    qubit_pairs: list[AnySpinQubit] = None,
    IQ_list: list[str] = ("I", "Q"),
    single_demod: bool = False,
) -> xr.Dataset:
    """
    return data array with the 'I' and 'Q' quadratures converted to Volts.

    :param da: the array on which to calculate angle. Assumed to have complex data
    :type da: xr.DataArray
    :param qubits: the list of qubits components.
    :type qubits: list[AnyTransmon]
    :param qubit_pairs: the list of qubit pair components.
    :type qubit_pairs: list[AnyTransmonPair]
    :param IQ_list: the list of data to convert to V e.g. ["I", "Q"].
    :type IQ_list: list[str]
    :param single_demod: Flag to add the additional factor of 2 needed for single demod.
    :type single_demod: bool
    :return: a data array with the same dimensions and coordinates, with a 'I' and 'Q' in Volts
    :type: xr.DataArray

    """
    # Create a xarray with a coordinate 'qubit' and the value is q.resonator.operations["readout"].length
    if qubits is not None:
        readout_lengths = xr.DataArray(
            [q.sensor_dots[0].readout_resonator.operations["readout"].length for q in qubits],
            coords=[("qubit", [q.name for q in qubits])],
        )
    elif qubit_pairs is not None:
        control_target = ["c", "t"]
        readout_lengths = xr.DataArray(
            data=[
                [
                    qp.sensor_dots[0].readout_resonator.operations["readout"].length,
                    qp.sensor_dots[0].readout_resonator.operations["readout"].length,
                ]
                for qp in qubit_pairs
            ],
            dims=["qubit_pair", "control_target"],
            coords={
                "qubit_pair": [qp.name for qp in qubit_pairs],
                "control_target": control_target,
            },
        )
    else:
        raise ValueError("Either qubits or qubit_pairs must be provided!")

    demod_factor = 2 if single_demod else 1
    return da.assign(
        {key: da[key] * demod_factor * 2**12 / readout_lengths for key in IQ_list}
    )
