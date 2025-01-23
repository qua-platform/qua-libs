from typing import Dict

from quam_libs.components import TransmonPair, TunableCoupler, QuAM


def add_transmon_pair_component(machine: QuAM, wiring_path: str, ports: Dict[str, str]) -> TransmonPair:
    if "opx_output" in ports:
        qubit_control_name = ports["control_qubit"].name
        qubit_target_name = ports["target_qubit"].name
        qubit_pair_name = f"{qubit_control_name}_{qubit_target_name}"
        coupler_name = f"coupler_{qubit_pair_name}"

        transmon_pair = TransmonPair(
            id=coupler_name,
            qubit_control=f"{wiring_path}/control_qubit",
            qubit_target=f"{wiring_path}/target_qubit",
            coupler=TunableCoupler(id=coupler_name, opx_output=f"{wiring_path}/opx_output"),
        )

    else:
        raise ValueError(f"Unimplemented mapping of port keys to channel for ports: {ports}")

    return transmon_pair
