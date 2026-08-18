from typing import Sequence

def apply_compensation_pulse(
    multiplexed_sensors: Sequence, max_voltage: float,
) -> None: 

    # Extract the VoltageSequence objects in this batch
    sequences_in_batch = {
        sensor.voltage_sequence.gate_set.id: sensor.voltage_sequence for sensor in multiplexed_sensors.values()
    }

    # Play a compensation pulse per sequence in the batch
    for seq in sequences_in_batch.values():
        seq.apply_compensation_pulse(
            max_voltage=max_voltage, go_to_zero=True, return_to_zero=True
        )
