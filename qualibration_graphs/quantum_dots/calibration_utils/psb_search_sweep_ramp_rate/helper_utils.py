import numpy as np
from typing import Dict, List

from qualibration_libs.core import tracked_updates
from qualibration_libs.parameters.experiment import QualibrationNode

__all__ = [
    "build_ramp_duration_sweep",
    "prepare_dot_pairs",
    "modify_and_track_point",
    "validate_and_build_ramp_sweep",
]

def validate_and_build_ramp_sweep(
    node: QualibrationNode
): 
    """
    Build a simple linear array of ramp durations. 

    Ensures that: 
        - The ramp_min, ramp_max, and ramp_step are all multiples of 4, matching the QUA clock cycle
        - The resulting ramp_duration_array is not empty
    """
    ramp_min = int(node.parameters.ramp_duration_min)
    ramp_max = int(node.parameters.ramp_duration_max)
    ramp_step = int(node.parameters.ramp_duration_step)

    if ramp_min % 4 != 0 or ramp_max % 4 != 0 or ramp_step % 4 != 0:
        raise ValueError(
            "Ramp settings must be divisible by 4. Received "
            f"ramp_duration_min={ramp_min}, ramp_duration_max={ramp_max}, ramp_duration_step={ramp_step}"
        )
    ramp_duration_array = np.arange(ramp_min, ramp_max, ramp_step, dtype=int)
    if len(ramp_duration_array) == 0:
        raise ValueError("Empty ramp duration sweep: require ramp_duration_min < ramp_duration_max with positive step.")
    
    return ramp_duration_array



def prepare_dot_pairs(node: QualibrationNode):
    """
    Prepares the QuantumDotPair objects of the QubitPair:
        - Resets the voltage sequence. Done for consecutive runs
        - If readout_length_max is None in parameters, ensure that either the readout lengths are the same, or
            raise an error.
    """
    qubit_pairs = node.namespace["qubit_pairs"]
    dot_pair_objects = [qp.quantum_dot_pair for qp in qubit_pairs]

    # TODO: Verify that this is strictly necessary
    for gate_set_id in {dot_pair.voltage_sequence.gate_set.id for dot_pair in dot_pair_objects}:
        node.machine.reset_voltage_sequence(gate_set_id)
    


def build_ramp_duration_sweep(ramp_duration_min: int, ramp_duration_max: int, ramp_duration_step: int) -> np.ndarray:
    """Build ramp duration grid (ns), same rules as 06d (multiples of 4 validated by caller)."""
    r_min = int(ramp_duration_min)
    r_max = int(ramp_duration_max)
    step = int(ramp_duration_step)
    return np.arange(r_min, r_max, step, dtype=int)

def modify_and_track_point(
    qubit_pair,
    detuning_value: float | None,
    tracked_dict: Dict,
):
    """If a detuning value is given, then this will be added to the tracked changes dict and the point will be mutated for now."""
    # If not value is given, skip
    if detuning_value is None:
        return

    # First extract the dot pair and the correspoding gate_set
    dot_pair = qubit_pair.quantum_dot_pair
    dot_pair_gate_set = dot_pair.voltage_sequence.gate_set

    # Build the point name. It will be f"{dot_pair.id}_measure" and get the point object
    point_name = dot_pair._create_point_name("measure")
    point = dot_pair_gate_set.get_macros()[point_name]

    # Store the tracked change in the dict, and mutate the point voltages
    tracked_dict[dot_pair.name] = point.voltages.get(dot_pair.name)
    point.voltages[dot_pair.name] = detuning_value
