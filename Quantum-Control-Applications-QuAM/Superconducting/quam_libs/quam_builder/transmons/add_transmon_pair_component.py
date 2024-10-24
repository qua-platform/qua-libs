from typing import Dict

from quam_libs.quam_builder.transmons.channel_ports import iq_in_out_channel_ports, mw_in_out_channel_ports
from quam_libs.components import TransmonPair, TunableCoupler, CrossDrive, CrossDriveMW QuAM


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
            coupler=TunableCoupler(id=coupler_name, opx_output=f"{wiring_path}/opx_output")
        )

    else:
        raise ValueError(f"Unimplemented mapping of port keys to channel for ports: {ports}")

    return transmon_pair


def add_transmon_pair_cross_drive_component(machine: QuAM, wiring_path: str, ports: Dict[str, str]) -> TransmonPair:
    qubit_control_name = ports["control_qubit"].name
    qubit_target_name = ports["target_qubit"].name
    qubit_pair_name = f"{qubit_control_name}_{qubit_target_name}"
    cross_drive_name = f"cross_drive_{qubit_pair_name}"

    if all(key in ports for key in iq_in_out_channel_ports):
        transmon_pair = TransmonPair(
            id=cross_drive_name,
            qubit_control=f"{wiring_path}/control_qubit",
            qubit_target=f"{wiring_path}/target_qubit",
            cross_drive=CrossDrive(id=cross_drive_name, opx_output=f"{wiring_path}/opx_output")
        )

    elif all(key in ports for key in mw_in_out_channel_ports):
        transmon_pair = TransmonPair(
            id=cross_drive_name,
            qubit_control=f"{wiring_path}/control_qubit",
            qubit_target=f"{wiring_path}/target_qubit",
            cross_drive=CrossDriveMW(id=cross_drive_name, opx_output=f"{wiring_path}/opx_output")
        )

    else:
        raise ValueError(f"Unimplemented mapping of port keys to channel for ports: {ports}")

    return transmon_pair