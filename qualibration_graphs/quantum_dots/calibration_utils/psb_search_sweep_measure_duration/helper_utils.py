import numpy as np
from typing import Dict, List

from qualibration_libs.core import tracked_updates

from qualibration_libs.parameters.experiment import QualibrationNode
from quam_builder.architecture.quantum_dots.components.readout_resonator import (
    ReadoutResonatorSingle,
    ReadoutResonatorIQ,
)

__all__ = [
    "build_psb_readout_sweep",
    "modify_and_track_point",
    "modify_and_track_readout_pulse",
    "validate_readout",
    "validate_dot_pairs",
    "prepare_dot_pairs",
]


def prepare_dot_pairs(node: QualibrationNode):
    """
    Prepares the QuantumDotPair objects of the QubitPair:
        - Resets the voltage sequence. Done for consecutive runs
        - Ensures that no quantum_dot_pair has more than one SensorDot
        - If readout_length_max is None in parameters, ensure that either the readout lengths are the same, or
            raise an error.
    """
    qubit_pairs = node.namespace["qubit_pairs"]
    dot_pair_objects = [qp.quantum_dot_pair for qp in qubit_pairs]

    # TODO: Verify that this is strictly necessary
    for gate_set_id in {dot_pair.voltage_sequence.gate_set.id for dot_pair in dot_pair_objects}:
        node.machine.reset_voltage_sequence(gate_set_id)
    for dot_pair in dot_pair_objects:
        if len(dot_pair.sensor_dots) != 1:
            raise ValueError(
                f"06e expects exactly one sensor dot per pair; {dot_pair.id!r} has {len(dot_pair.sensor_dots)}"
            )

    readout_max = node.parameters.readout_length_max
    if readout_max is None:
        lengths = {}

        for dot_pair in dot_pair_objects:
            operation_name = f"readout_{dot_pair.name}"
            resonator = dot_pair.sensor_dots[0].readout_resonator
            lengths[dot_pair.name] = resonator.operations[operation_name].length

        unique_lengths = set(lengths.values())

        if len(unique_lengths) != 1:
            raise ValueError(
                "Dot pairs have different configured readout lengths. "
                "Set readout_length_max explicitly. "
                f"Configured lengths: {lengths}"
            )

        readout_max = unique_lengths.pop()
    return readout_max


def _validate_array_size(sweep_dict: Dict):
    """Ensures that the array size of the built sweep is larger than the number of segments for a given pulse length.

    Built on the sweep_dict returned in `build_psb_readout_sweep`"""
    array_size = sweep_dict["array_size"]
    num_segments = sweep_dict["num_segments"]
    if array_size > num_segments:
        raise ValueError(
            f"Sweep has {array_size} save points but pulse allows {num_segments} segments "
            f"(pulse_length={sweep_dict["pulse_length"]})."
        )


def _validate_resonator_types(qubit_pairs):
    """Ensures that all ReadoutResonators of all SensorDot objects are consistent"""
    kinds = {type(qp.quantum_dot_pair.sensor_dots[0].readout_resonator) for qp in qubit_pairs}
    if len(kinds) != 1:
        raise TypeError(f"06b expects all qubit pairs to use the same readout resonator class; got {kinds}.")
    (readout_cls,) = tuple(kinds)
    if readout_cls not in (ReadoutResonatorSingle, ReadoutResonatorIQ):
        raise TypeError(f"06b supports ReadoutResonatorSingle and ReadoutResonatorIQ; got {readout_cls}.")
    return readout_cls


def validate_readout(qubit_pairs, sweep_dict: Dict):
    """
    Ensures that:
        - The array size of the built sweep is larger than the number of segments for a given pulse length
        - All ReadoutResonators of all SensorDot objects are consistent
    """
    _validate_array_size(sweep_dict)
    return _validate_resonator_types(qubit_pairs)


def modify_and_track_point(
    qubit_pair,
    detuning_value: float,
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


def modify_and_track_readout_pulse(
    qubit_pair,
    readout_length: int,
    tracked_list: List,
):
    """Update the readout pulse length for a given sensor's readout resonator, and the corresponding pulse"""

    # Extract the readout_resonator for the qubit_pair's quantum_dot_pair, and create the readout operation name.
    rr = qubit_pair.quantum_dot_pair.sensor_dots[0].readout_resonator
    op_name = "readout" + f"_{qubit_pair.quantum_dot_pair.name}"

    # Track this update and add to dict, so that we can revert later
    with tracked_updates(rr, auto_revert=False, dont_assign_to_none=True) as resonator:
        resonator.operations[op_name].length = readout_length
        tracked_list.append(resonator)


def build_psb_readout_sweep(readout_length_min: int, readout_length_max: int, readout_length_points: int) -> dict:
    """Build sweep grid consistent with 05c charge-state readout time (arange + chunk step).

    QM requires the readout pulse length to equal an integer number of accumulated segments:
    ``pulse_length == num_segments * 4 * segment_length`` (see ``measure_accumulated``).
    ``readout_length_max`` is therefore rounded **down** to the nearest valid pulse length.

    Returns keys: ``array_size``, ``step_ns``, ``samples_per_chunk``, ``sweep_coord``,
    ``pulse_length`` (effective ns), ``segment_length`` (QUA ``segment_length`` arg),
    ``num_segments``.
    """
    r_min = max(4, int(readout_length_min) // 4 * 4)
    r_max = max(r_min + 4, int(readout_length_max) // 4 * 4)
    n_pts = max(1, int(readout_length_points))
    if n_pts < 2:
        step_ns = max(4, r_max - r_min)
    else:
        step_ns = max(4, ((r_max - r_min) // (n_pts - 1)) // 4 * 4)
    segment_length = max(1, step_ns // 4)
    chunk_ns = 4 * segment_length
    num_segments = r_max // chunk_ns
    if num_segments < 1:
        num_segments = 1
    pulse_length = num_segments * chunk_ns
    if pulse_length < r_min:
        num_segments = int(np.ceil(r_min / chunk_ns))
        pulse_length = num_segments * chunk_ns

    integrations_times = np.arange(r_min, pulse_length, step_ns, dtype=int)
    if len(integrations_times) == 0:
        integrations_times = np.array([min(r_min, pulse_length)], dtype=int)
    array_size = len(integrations_times)
    if array_size > num_segments:
        array_size = num_segments
        integrations_times = integrations_times[:array_size]
    sweep_coord = np.arange(1, array_size + 1, dtype=int) * chunk_ns
    return {
        "array_size": array_size,
        "step_ns": step_ns,
        "samples_per_chunk": segment_length,
        "sweep_coord": sweep_coord,
        "pulse_length": pulse_length,
        "segment_length": segment_length,
        "num_segments": num_segments,
    }
