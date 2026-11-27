import numpy as np
import xarray as xr
from typing import Dict, List

from qualibration_libs.core import tracked_updates
from qualibration_libs.parameters.experiment import QualibrationNode

__all__ = [
    "build_ramp_duration_sweep",
    "modify_and_track_point",
    "validate_and_build_ramp_sweep",
    "extract_vgs_id",
    "assemble_ds_raw",
]


def extract_vgs_id(qubit_pairs):
    vgs_id = next(iter({pair.quantum_dot_pair.voltage_sequence.gate_set.name for pair in qubit_pairs}))
    return vgs_id


def assemble_ds_raw(dataset: xr.Dataset, pair_names: List[str]) -> xr.Dataset:
    """Convert fetched per-pair streams into the canonical 06c ``ds_raw`` layout."""
    i_arr = xr.concat([dataset[f"I_{pair_name}"] for pair_name in pair_names], dim="qubit_pair")
    q_arr = xr.concat([dataset[f"Q_{pair_name}"] for pair_name in pair_names], dim="qubit_pair")
    i_arr = i_arr.assign_coords(qubit_pair=pair_names)
    q_arr = q_arr.assign_coords(qubit_pair=pair_names)
    return xr.Dataset({"I": i_arr, "Q": q_arr})


def validate_and_build_ramp_sweep(node: QualibrationNode):
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
