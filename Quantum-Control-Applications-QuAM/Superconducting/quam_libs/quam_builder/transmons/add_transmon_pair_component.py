from typing import Dict

from quam_libs.quam_builder.transmons.channel_ports import iq_in_out_channel_ports, mw_in_out_channel_ports
from quam_libs.components import TransmonPair, TunableCoupler, CrossResonance, CrossResonanceMW, QuAM


def add_transmon_pair_tunable_coupler_component(machine: QuAM, wiring_path: str, ports: Dict[str, str]) -> TransmonPair:
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


def add_transmon_pair_cross_resonance_component(machine: QuAM, wiring_path: str, ports: Dict[str, str]) -> TransmonPair:
    qubit_control_name = ports["control_qubit"].name
    qubit_target_name = ports["target_qubit"].name
    qubit_pair_name = f"{qubit_control_name}_{qubit_target_name}"
    cross_resonance_name = f"cross_resonance_{qubit_pair_name}"

    if all(key in iq_in_out_channel_ports for key in ports):
        transmon_pair = TransmonPair(
            id=cross_resonance_name,
            qubit_control=f"{wiring_path}/control_qubit",
            qubit_target=f"{wiring_path}/target_qubit",
            cross_resonance=CrossResonance(
                id=cross_resonance_name,
                opx_output_I=f"{wiring_path}/opx_output_I",
                opx_output_Q=f"{wiring_path}/opx_output_Q",
                intermediate_frequency="#./inferred_intermediate_frequency",
                # todo: change this to be upconverter frequency
                frequency_converter_up=ports.data["control_qubit"] + "/xy/frequency_converter_up",
                target_qubit_LO_frequency=ports.data["target_qubit"] + "/xy/LO_frequency",
                target_qubit_IF_frequency=ports.data["target_qubit"] + "/xy/intermediate_frequency"
            )
        )

    elif all(key in mw_in_out_channel_ports for key in ports):
        transmon_pair = TransmonPair(
            id=cross_resonance_name,
            qubit_control=f"{wiring_path}/control_qubit",
            qubit_target=f"{wiring_path}/target_qubit",
            cross_resonance=CrossResonanceMW(id=cross_resonance_name, opx_output=f"{wiring_path}/opx_output")
        )

    else:
        raise ValueError(f"Unimplemented mapping of port keys to channel for ports: {ports}")

    return transmon_pair


# TODO: potentially add zz-inducing drive
#       this depends on zz drive as new elements or detuning to cross reosonace. preferably the former.
def add_transmon_pair_zz_drive_component(machine: QuAM, wiring_path: str, ports: Dict[str, str]) -> TransmonPair:
   pass