from typing import Dict, Sequence

import xarray as xr

from qualibration_libs.parameters.experiment import get_qubits

__all__ = [
    "assemble_labeled_ds_raw",
    "modify_and_track_point",
    "resolve_qubits_and_dot_pairs",
]


def resolve_qubits_and_dot_pairs(node):
    """Return selected qubits and their preferred PSB readout dot pairs."""
    machine = node.machine
    qubits = get_qubits(node)
    pairs = []
    for qubit in qubits:
        preferred_dot_id = getattr(qubit, "preferred_readout_quantum_dot", None)
        if preferred_dot_id is None:
            raise ValueError(
                f"Qubit {qubit.id!r} has no preferred_readout_quantum_dot set; "
                "configure it to the partner dot used for PSB readout."
            )
        pair_name = machine.find_quantum_dot_pair(qubit.quantum_dot.id, preferred_dot_id)
        if pair_name is None:
            raise ValueError(
                f"No QuantumDotPair registered for dots {qubit.quantum_dot.id!r} and "
                f"{preferred_dot_id!r} (qubit {qubit.id!r})."
            )
        pairs.append((qubit, machine.quantum_dot_pairs[pair_name]))
    return qubits, pairs


def assemble_labeled_ds_raw(fetched_dataset: xr.Dataset, qubits: Sequence) -> xr.Dataset:
    """Assemble the standard 06d ``ds_raw`` from fetched per-qubit stream variables."""

    qnames = [q.name for q in qubits]

    def _concat(prefix: str) -> xr.DataArray:
        arrays = [fetched_dataset[f"{prefix}_{name}"] for name in qnames]
        return xr.concat(arrays, dim="qubit").assign_coords(qubit=qnames)

    return xr.Dataset(
        {
            "I_no_pi": _concat("I_no_pi"),
            "Q_no_pi": _concat("Q_no_pi"),
            "I_pi": _concat("I_pi"),
            "Q_pi": _concat("Q_pi"),
        }
    )


def modify_and_track_point(
    quantum_dot_pair,
    detuning_value: float | None,
    tracked_dict: Dict,
):
    """Temporarily override the measure-point detuning and remember the original value."""
    if detuning_value is None:
        return

    dot_pair = quantum_dot_pair
    dot_pair_gate_set = dot_pair.voltage_sequence.gate_set

    point_name = dot_pair._create_point_name("measure")
    point = dot_pair_gate_set.get_macros()[point_name]

    tracked_dict[dot_pair.name] = point.voltages.get(dot_pair.name)
    point.voltages[dot_pair.name] = detuning_value
