"""Runtime QM config helpers for PALEA leakage amplification."""

from __future__ import annotations

import copy
import dataclasses
from typing import Dict, Iterable, List


def unique_high_qubits(qubit_roles_map, qubit_pairs) -> List:
    """Return the distinct high-frequency qubits across active pairs."""
    seen = set()
    high_qubits = []
    for qp in qubit_pairs:
        high_q = qubit_roles_map[qp.name].high
        if high_q.name in seen:
            continue
        seen.add(high_q.name)
        high_qubits.append(high_q)
    return high_qubits


def ensure_ef_x180_operation(qubit) -> None:
    """Ensure ``EF_x180`` exists on the qubit XY channel (same pattern as node 13)."""
    if hasattr(qubit.xy.operations, "EF_x180"):
        return
    x180 = qubit.xy.operations["x180"]
    qubit.xy.operations["EF_x180"] = (
        dataclasses.replace(x180, alpha=0.0) if hasattr(x180, "alpha") else dataclasses.replace(x180)
    )


def add_palea_ef_elements(config: dict, high_qubits: Iterable) -> Dict[str, str]:
    """Add temporary EF elements at IF = xy_IF - anharmonicity for PALEA DD.

    For each high-frequency qubit, creates ``{xy_element}.ef`` sharing the XY port and
    exposing a single ``x180`` operation aliased to the existing ``EF_x180`` pulse.

    Returns
    -------
    dict
        Mapping from qubit name to EF element name.
    """
    ef_element_names: Dict[str, str] = {}
    for qubit in high_qubits:
        xy_name = qubit.xy.name
        ef_name = f"{xy_name}.ef"
        try:
            xy_element = config["elements"][xy_name]
        except KeyError as exc:
            raise KeyError(f"XY element {xy_name!r} not found in generated config.") from exc

        try:
            ef_x180_pulse = xy_element["operations"]["EF_x180"]
        except KeyError as exc:
            raise ValueError(
                f"Qubit {qubit.name} is missing EF_x180 on {xy_name}; calibrate EF gates first."
            ) from exc

        ef_element = copy.deepcopy(xy_element)
        ef_element["intermediate_frequency"] = qubit.xy.intermediate_frequency - qubit.anharmonicity
        ef_element["operations"] = {"EF_x180": ef_x180_pulse}
        config["elements"][ef_name] = ef_element
        ef_element_names[qubit.name] = ef_name
    return ef_element_names


def build_palea_qm_config(machine, qubit_pairs, qubit_roles_map) -> tuple[dict, Dict[str, str]]:
    """Generate QM config with PALEA EF elements injected for all high qubits in the pairs."""
    high_qubits = unique_high_qubits(qubit_roles_map, qubit_pairs)
    for qubit in high_qubits:
        ensure_ef_x180_operation(qubit)
    config = machine.generate_config()
    ef_element_names = add_palea_ef_elements(config, high_qubits)
    return config, ef_element_names
