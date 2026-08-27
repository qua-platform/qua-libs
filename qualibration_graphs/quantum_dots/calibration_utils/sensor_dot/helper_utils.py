from typing import Sequence, Dict, Any

from qualibrate.core import QualibrationNode

__all__ = [
    "apply_compensation_pulse",
    "refresh_voltage_sequences",
]


def apply_compensation_pulse(
    multiplexed_sensors: Sequence,
    max_voltage: float,
) -> None:

    # Extract the VoltageSequence objects in this batch
    sequences_in_batch = _extract_batch_sequences(multiplexed_sensors)

    # Play a compensation pulse per sequence in the batch
    for seq in sequences_in_batch.values():
        seq.apply_compensation_pulse(max_voltage=max_voltage, go_to_zero=True, return_to_zero=True)


def _extract_batch_sequences(
    multiplexed_sensors: Sequence,
) -> Dict[str, Any]:
    sequences_in_batch = {
        sensor.voltage_sequence.gate_set.id: sensor.voltage_sequence for sensor in multiplexed_sensors.values()
    }
    return sequences_in_batch


def refresh_voltage_sequences(
    node: QualibrationNode,
    multiplexed_sensors: Sequence,
) -> None:
    # Extract the VoltageSequence objects in this batch
    sequences_in_batch = _extract_batch_sequences(multiplexed_sensors)
    machine = node.machine

    for gate_set_id in sequences_in_batch.keys():
        machine.reset_voltage_sequence(gate_set_id, track_integrated_voltage=True)
