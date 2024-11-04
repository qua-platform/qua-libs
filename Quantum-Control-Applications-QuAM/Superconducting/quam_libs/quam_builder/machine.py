import os
from pathlib import Path
from typing import Union, Dict

from quam.components import Octave

from qualang_tools.wirer import Connectivity
from quam_libs.quam_builder.pulses import add_default_transmon_pulses, add_default_transmon_pair_pulses
from quam_libs.quam_builder.transmons.add_transmon_drive_component import add_transmon_drive_component
from quam_libs.quam_builder.transmons.add_transmon_flux_component import add_transmon_flux_component
from quam_libs.quam_builder.transmons.add_transmon_pair_component import (
    add_transmon_pair_tunable_coupler_component, add_transmon_pair_cross_resonance_component, add_transmon_pair_zz_drive_component
)
from quam_libs.quam_builder.transmons.add_transmon_resonator_component import add_transmon_resonator_component
from qualang_tools.wirer.connectivity.wiring_spec import WiringLineType
from quam_libs.components import OPXPlusQuAM, FEMQuAM, QuAM, Transmon, TransmonPair
from quam_libs.quam_builder.wiring.create_wiring import create_wiring


def build_quam(machine: QuAM, quam_state_path: Union[Path, str], octaves_settings: Dict = {}) -> QuAM:
    add_octaves(machine, octaves_settings, quam_state_path)
    add_ports(machine)
    add_transmons(machine)
    add_pulses(machine)

    save_machine(machine, quam_state_path)

    return machine


def build_quam_wiring(connectivity: Connectivity, host_ip: str, cluster_name: str,
                      quam_state_path: Union[Path, str]) -> QuAM:
    if os.path.exists(quam_state_path) and 'state.json' in os.listdir(quam_state_path):
        # if there is a non-empty QuAM state already
        machine = QuAM.load(quam_state_path)
    else:
        machine = create_base_machine(connectivity)

    add_name_and_ip(machine, host_ip, cluster_name)
    machine.wiring = create_wiring(connectivity)
    save_machine(machine, quam_state_path)

    return machine

def create_base_machine(connectivity: Connectivity):
    """
    Detects whether the `connectivity` is using OPX+ or OPX1000 and returns
    the corresponding base object. Otherwise, raises a TypeError.
    """
    for element in connectivity.elements.values():
        for channels in element.channels.values():
            for channel in channels:
                if channel.instrument_id in ["lf-fem", "mw-fem"]:
                    return FEMQuAM()
                elif channel.instrument_id in ["opx+"]:
                    return OPXPlusQuAM()

    raise TypeError("Couldn't identify connectivity as OPX+ or LF-FEM. "
                    "Are channels were allocated for the connectivity?")


def add_name_and_ip(machine: QuAM, host_ip: str, cluster_name: str):
    """ Stores the minimal information to connect to a QuantumMachinesManager. """
    machine.network = {
        "host": host_ip,
        "port": 9510,
        "cluster_name": cluster_name
    }


def add_ports(machine: QuAM):
    """
    Creates and stores all input/output ports according to what has been
    allocated to each element in the machine's wiring.
    """
    for wiring_by_element in machine.wiring.values():
        for wiring_by_line_type in wiring_by_element.values():
            for ports in wiring_by_line_type.values():
                for port in ports:
                    if "ports" in ports.get_unreferenced_value(port):
                        machine.ports.reference_to_port(
                            ports.get_unreferenced_value(port),
                            create=True
                        )

def add_transmons(machine: QuAM):
    for element_type, wiring_by_element in machine.wiring.items():
        if element_type == 'qubits':
            for qubit_id, wiring_by_line_type in wiring_by_element.items():
                transmon = Transmon(id=qubit_id)
                machine.qubits[qubit_id] = transmon
                for line_type, ports in wiring_by_line_type.items():
                    wiring_path = f"#/wiring/{element_type}/{qubit_id}/{line_type}"
                    if line_type == WiringLineType.RESONATOR.value:
                        add_transmon_resonator_component(transmon, wiring_path, ports, machine)
                    elif line_type == WiringLineType.DRIVE.value:
                        add_transmon_drive_component(transmon, wiring_path, ports)
                    elif line_type == WiringLineType.FLUX.value:
                        add_transmon_flux_component(transmon, wiring_path, ports)
                    else:
                        raise ValueError(f'Unknown line type: {line_type}')
                machine.active_qubit_names.append(transmon.name)

        elif element_type == 'qubit_pairs':
            for qubit_pair_id, wiring_by_line_type in wiring_by_element.items():
                qc, qt = qubit_pair_id.split("-")
                qt = f"q{qt}"
                transmon_pair = TransmonPair(id=qubit_pair_id, qubit_control=f"#/qubits/{qc}", qubit_target=f"#/qubits/{qt}")
                for line_type, ports in wiring_by_line_type.items():
                    wiring_path = f"#/wiring/{element_type}/{qubit_pair_id}/{line_type}"
                    if line_type == WiringLineType.COUPLER.value:
                        add_transmon_pair_tunable_coupler_component(transmon_pair, wiring_path, ports)
                    elif line_type == WiringLineType.CROSS_RESONANCE.value:
                        add_transmon_pair_cross_resonance_component(transmon_pair, wiring_path, ports)
                    elif line_type == WiringLineType.ZZ_DRIVE.value:
                        add_transmon_pair_zz_drive_component(transmon_pair, wiring_path, ports)
                    else:
                        raise ValueError(f'Unknown line type: {line_type}')
                machine.qubit_pairs[transmon_pair.name] = transmon_pair
                machine.active_qubit_pair_names.append(transmon_pair.name)


def add_pulses(machine: QuAM):
    if hasattr(machine, 'qubits'):
        for transmon in machine.qubits.values():
            add_default_transmon_pulses(transmon)

    if hasattr(machine, 'qubit_pairs'):
        for qubit_pair in machine.qubit_pairs.values():
            add_default_transmon_pair_pulses(qubit_pair)

def add_octaves(machine: QuAM, octaves_settings: Dict, quam_state_path: Union[Path, str]):
    octave_ips, octave_ports = [], []
    for octave_settings in octaves_settings.values():
        octave_ips.append(octave_settings.get("ip", machine.network.host))
        octave_ports.append(octave_settings.get("port", 80))
    machine.network["octave_ips"] = octave_ips
    machine.network["octave_ports"] = octave_ports

    if isinstance(quam_state_path, str):
        quam_state_path = Path(quam_state_path)
    quam_state_path = str(quam_state_path.parent.resolve())
    for i, octave_name in enumerate(octaves_settings):
        octave = Octave(
            name=octave_name,
            ip=machine.network["octave_ips"][i],
            port=machine.network["octave_ports"][i],
            calibration_db_path=quam_state_path
        )
        machine.octaves[octave_name] = octave
        octave.initialize_frequency_converters()

    return machine


def save_machine(machine: QuAM, quam_state_path: Union[Path, str]):
    machine.save(
        path=quam_state_path,
        content_mapping={
            "wiring.json": ["network", "wiring"],
        }
    )


