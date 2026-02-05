from typing import List, Set, Dict, Optional, Tuple, Literal
from dataclasses import dataclass

from qualibrate import QualibrationNode
from qualibration_libs.core import BatchableList

from quam_builder.architecture.quantum_dots.components import SensorDot
from quam_builder.architecture.quantum_dots.qpu import BaseQuamQD
from quam_builder.architecture.quantum_dots.qubit import AnySpinQubit

from calibration_utils.common_utils.experiment import _get_qubits, get_sensors


@dataclass
class QubitReadoutFootprint:
    qubit_id: str
    own_dot_id: str
    readout_pair_dot_id: str
    quantum_dot_pair_id: str
    sensor_ids: Set[str]

    def conflicts_with(self, other_footprint: "QubitReadoutFootprint") -> bool:
        my_dots = {self.own_dot_id, self.readout_pair_dot_id}
        other_dots = {other_footprint.own_dot_id, other_footprint.readout_pair_dot_id}
        if my_dots & other_dots:
            return True
        if self.quantum_dot_pair_id == other_footprint.quantum_dot_pair_id:
            return True
        return False


def _get_qubit_readout_footprint(machine: BaseQuamQD, qubit: AnySpinQubit) -> QubitReadoutFootprint:
    own_dot_id = qubit.quantum_dot.id
    readout_dot_id = qubit.preferred_readout_quantum_dot

    if readout_dot_id is None:
        raise ValueError(f"Pairwise readout dot not chosen for {qubit.id}")

    pair_id = machine.find_quantum_dot_pair(own_dot_id, readout_dot_id)

    if pair_id is None:
        raise ValueError(f"QuantumDotPair of {own_dot_id} and {readout_dot_id} not defined, for qubit {qubit.id}")

    sensor_ids = {s.id for s in qubit.sensor_dots}

    return QubitReadoutFootprint(
        qubit_id=qubit.id,
        own_dot_id=own_dot_id,
        readout_pair_dot_id=readout_dot_id,
        quantum_dot_pair_id=pair_id,
        sensor_ids=sensor_ids,
    )


def _get_compatible_sensor_groups(sensor_batches: BatchableList[SensorDot]) -> List[Set[str]]:
    groups = []
    for batch in sensor_batches.batch():
        sensor_ids = {s.id for s in batch.values()}
        groups.append(sensor_ids)

    return groups


def _sensors_are_compatible(sensors_a: Set[str], sensors_b: Set[str], compatible_groups: List[Set[str]]) -> bool:
    combined = sensors_a | sensors_b
    for group in compatible_groups:
        if combined <= group:
            return True

    return False


def _build_readout_batches(
    footprints: List[QubitReadoutFootprint], compatible_sensor_groups: List[Set[str]]
) -> List[List[int]]:

    batches: List[List[int]] = []

    for idx, footprint in enumerate(footprints):
        placed = False
        for batch in batches:
            can_join = True

            for existing_idx in batch:
                existing = footprints[existing_idx]

                if footprint.conflicts_with(existing):
                    can_join = False
                    break

                if not _sensors_are_compatible(footprint.sensor_ids, existing.sensor_ids, compatible_sensor_groups):
                    can_join = False
                    break

            if can_join:
                batch.append(idx)
                placed = True
                break
        if not placed:
            batches.append([idx])

    return batches


def get_qubits_batched_by_readout(
    node: QualibrationNode,
    sensor_batches: BatchableList[SensorDot] = None,
    execution_mode: Literal["parallel", "sequential"] = "sequential",
) -> BatchableList[AnySpinQubit]:
    """
    Get qubits into batches that can be read out simultaneously.
    Args:
        - node: QualibrationNode, the QualibrationNode which is used in the Qualibrate script.
        - sensor_batches: BatchableList[SensorDot], a BatchableList of SensorDot objects. This will be broken down into the constituent batches with no footprint overlap.
        - execution_mode: Literal["parallel", "sequential"]: The order in which the qubits are operated.
            - "sequential": Each qubit is operated one at a time (default).
            - "parallel": Qubits are batched to maximize parallelism while avoiding footprint conflicts.
    """

    machine = node.machine
    qubits = _get_qubits(machine, node.parameters)

    if not qubits:
        return BatchableList([], [])

    if execution_mode == "sequential":
        batched_groups = [[i] for i in range(len(qubits))]
        return BatchableList(qubits, batched_groups)

    if sensor_batches is None:
        sensors = list(machine.sensor_dots.values())
        batched_groups = [[i for i in range(len(sensors))]]
        sensor_batches = BatchableList(sensors, batched_groups)

    compatible_sensor_groups = _get_compatible_sensor_groups(sensor_batches)

    footprints = [_get_qubit_readout_footprint(machine, q) for q in qubits]

    batched_groups = _build_readout_batches(footprints, compatible_sensor_groups)

    return BatchableList(qubits, batched_groups)
