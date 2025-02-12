from pathlib import Path
from typing import Union
from numpy import sqrt, ceil
from quam.components import Octave, LocalOscillator
from quam.components import FrequencyConverter

from quam_libs.components.superconducting.architectural_elements.mixer import StandaloneMixer
from quam_libs.quam_builder.superconducting.pulses import add_default_transmon_pulses, add_default_transmon_pair_pulses
from quam_libs.quam_builder.superconducting.add_transmon_drive_component import add_transmon_drive_component
from quam_libs.quam_builder.superconducting.add_transmon_flux_component import add_transmon_flux_component
from quam_libs.quam_builder.superconducting.add_transmon_pair_component import (
    add_transmon_pair_tunable_coupler_component, add_transmon_pair_cross_resonance_component, add_transmon_pair_zz_drive_component
)
from quam_libs.quam_builder.superconducting.add_transmon_resonator_component import add_transmon_resonator_component
from qualang_tools.wirer.connectivity.wiring_spec import WiringLineType
from quam_libs.components.superconducting.qpu import AnyQuAM
from quam_libs.quam_builder.qop_connectivity.build_quam_wiring import save_machine


def build_quam(machine: Union[AnyQuAM], quam_state_path: Union[Path, str]) -> Union[AnyQuAM]:
    add_octaves(machine, quam_state_path)
    add_external_mixers(machine)
    add_ports(machine)
    add_transmons(machine)
    add_pulses(machine)

    save_machine(machine, quam_state_path)

    return machine

def add_ports(machine: Union[AnyQuAM]):
    """
    Creates and stores all input/output ports according to what has been
    allocated to each element in the machine's wiring.
    """
    for wiring_by_element in machine.wiring.values():
        for wiring_by_line_type in wiring_by_element.values():
            for ports in wiring_by_line_type.values():
                for port in ports:
                    if "ports" in ports.get_unreferenced_value(port):
                        machine.ports.reference_to_port(ports.get_unreferenced_value(port), create=True)


def _set_default_grid_location(qubit_number: int, total_nuber_of_qubits: int):
    number_of_rows = int(ceil(sqrt(total_nuber_of_qubits)))
    y = qubit_number % number_of_rows
    x = qubit_number // number_of_rows
    return f"{x},{y}"


def add_transmons(machine: Union[AnyQuAM]):
    for element_type, wiring_by_element in machine.wiring.items():
        if element_type == "qubits":
            machine.active_qubit_names = []
            number_of_qubits = len(wiring_by_element.items())
            qubit_number = 0
            for qubit_id, wiring_by_line_type in wiring_by_element.items():
                transmon = machine.qubit_type(id=qubit_id)
                machine.qubits[qubit_id] = transmon
                machine.qubits[qubit_id].grid_location = _set_default_grid_location(qubit_number, number_of_qubits)
                qubit_number += 1
                for line_type, ports in wiring_by_line_type.items():
                    wiring_path = f"#/wiring/{element_type}/{qubit_id}/{line_type}"
                    if line_type == WiringLineType.RESONATOR.value:
                        add_transmon_resonator_component(transmon, wiring_path, ports)
                    elif line_type == WiringLineType.DRIVE.value:
                        add_transmon_drive_component(transmon, wiring_path, ports)
                    elif line_type == WiringLineType.FLUX.value:
                        add_transmon_flux_component(transmon, wiring_path, ports)
                    else:
                        raise ValueError(f"Unknown line type: {line_type}")
                machine.active_qubit_names.append(transmon.name)

        elif element_type == "qubit_pairs":
            machine.active_qubit_pair_names = []
            for qubit_pair_id, wiring_by_line_type in wiring_by_element.items():
                qc, qt = qubit_pair_id.split("-")
                qt = f"q{qt}"
                transmon_pair = machine.qubit_pair_type(id=qubit_pair_id, qubit_control=f"#/qubits/{qc}",qubit_target=f"#/qubits/{qt}")
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


def add_pulses(machine: Union[AnyQuAM]):
    if hasattr(machine, "qubits"):
        for transmon in machine.qubits.values():
            add_default_transmon_pulses(transmon)

    if hasattr(machine, "qubit_pairs"):
        for qubit_pair in machine.qubit_pairs.values():
            add_default_transmon_pair_pulses(qubit_pair)


def add_octaves(machine: Union[AnyQuAM], quam_state_path: Union[Path, str]):
    if isinstance(quam_state_path, str):
        quam_state_path = Path(quam_state_path)
    quam_state_path = str(quam_state_path.parent.resolve())
    for wiring_by_element in machine.wiring.values():
        for qubit, wiring_by_line_type in wiring_by_element.items():
            for line_type, references in wiring_by_line_type.items():
                for reference in references:
                    if "octaves" in references.get_unreferenced_value(reference):
                        octave_name = references.get_unreferenced_value(reference).split('/')[2]
                        octave = Octave(
                            name=octave_name,
                            calibration_db_path=quam_state_path,
                        )
                        machine.octaves[octave_name] = octave
                        octave.initialize_frequency_converters()

    return machine

def add_external_mixers(machine: Union[AnyQuAM]):
    for wiring_by_element in machine.wiring.values():
        for qubit, wiring_by_line_type in wiring_by_element.items():
            for line_type, references in wiring_by_line_type.items():
                for reference in references:
                    if "mixers" in references.get_unreferenced_value(reference):
                        mixer_name = references.get_unreferenced_value(reference).split('/')[2]
                        transmon_channel = {
                            WiringLineType.DRIVE.value: "xy",
                            WiringLineType.RESONATOR.value: "resonator"
                        }
                        frequency_converter = FrequencyConverter(
                            local_oscillator=LocalOscillator(),
                            mixer=StandaloneMixer(
                                intermediate_frequency=f"#/qubits/{qubit}/{transmon_channel[line_type]}/intermediate_frequency",
                            )
                        )
                        machine.mixers[mixer_name] = frequency_converter

    return machine
