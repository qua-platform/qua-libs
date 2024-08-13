import os.path
from typing import Any, Dict, List, Optional, Type, TypeVar
from pathlib import Path
import json

from qualang_tools.units import unit
from quam.components import Octave, IQChannel, DigitalOutputChannel
from quam.core import QuamRoot
from quam_libs.components import (
    Transmon,
    ReadoutResonator,
    FluxLine,
    TunableCoupler,
    TransmonPair,
    QuAM,
    FEMQuAM,
    OPXPlusQuAM,
)

from make_pulses import add_default_transmon_pulses, add_default_transmon_pair_pulses
from make_wiring import default_port_allocation, custom_port_allocation, create_wiring

# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)

# Type hint for QuAM types
QuamTypes = TypeVar("QuamTypes", OPXPlusQuAM, FEMQuAM)


def create_ports_from_wiring(machine: QuamRoot, wiring: Dict[str, Any]):
    for qubit_wiring in wiring["qubits"].values():
        # Create all necessary ports
        machine.ports.reference_to_port(qubit_wiring["xy"]["digital_port"], create=True)
        machine.ports.reference_to_port(qubit_wiring["xy"]["opx_output_I"], create=True)
        machine.ports.reference_to_port(qubit_wiring["xy"]["opx_output_Q"], create=True)
        machine.ports.reference_to_port(qubit_wiring["z"]["opx_output"], create=True)
        machine.ports.reference_to_port(
            qubit_wiring["resonator"]["opx_output_I"], create=True
        )
        machine.ports.reference_to_port(
            qubit_wiring["resonator"]["opx_output_Q"], create=True
        )
        machine.ports.reference_to_port(
            qubit_wiring["resonator"]["opx_input_I"], create=True
        )
        machine.ports.reference_to_port(
            qubit_wiring["resonator"]["opx_input_Q"], create=True
        )
        machine.ports.reference_to_port(
            qubit_wiring["resonator"]["digital_port"], create=True
        )

    for qubit_pair_wiring in wiring.get("qubit_pairs", {}).values():
        machine.ports.reference_to_port(
            qubit_pair_wiring["coupler"]["opx_output"], create=True
        )


def create_octaves(octaves_network) -> Dict[str, Octave]:
    octaves = {}
    for octave_name, octave_network in octaves_network.items():
        octave = Octave(
            name=octave_name,
            ip=octave_network["ip"],
            port=octave_network["port"],
            calibration_db_path=os.path.dirname(__file__),
        )
        # TODO: make a utility function to get the octaves from the wiring
        octave.initialize_frequency_converters()
        octaves[octave_name] = octave
    return octaves


def create_quam_superconducting(
    quam_class: Type[QuamTypes],
    wiring: Dict[str, Any],
    network: Dict[str, Any],
    octaves: Optional[Dict[str, Octave]] = None,
) -> QuamTypes:
    # TODO: docstring
    """Create a QuAM with a number of qubits.

    Args:
        wiring (dict): Wiring of the qubits.

    Returns:
        QuamRoot: A QuAM with the specified number of qubits.
    """
    # Initiate the QuAM class
    machine = quam_class(wiring=wiring, network=network)

    # This function should assume there are octaves, this should be inferred from the wiring because there is also mw fem
    if octaves is None:
        machine.octaves = create_octaves(network["octaves"])
    else:
        machine.octaves = octaves

    create_ports_from_wiring(machine, wiring)

    # Add the transmon components (xy, z and resonator) to the quam
    for qubit_name, qubit_wiring in machine.wiring["qubits"].items():
        # Create qubit components
        transmon = Transmon(
            id=qubit_name,
            xy=IQChannel(
                opx_output_I=qubit_wiring.xy.get_reference("opx_output_I"),
                opx_output_Q=qubit_wiring.xy.get_reference("opx_output_Q"),
                frequency_converter_up=qubit_wiring.xy.get_reference(
                    "frequency_converter_up"
                ),
                intermediate_frequency=-200 * u.MHz,
                digital_outputs={
                    "octave_switch": DigitalOutputChannel(
                        opx_output=qubit_wiring.xy.get_reference("digital_port"),
                        delay=87,  # 57ns for QOP222 and above
                        buffer=15,  # 18ns for QOP222 and above
                    )
                },
            ),
            z=FluxLine(opx_output=qubit_wiring.z.get_reference("opx_output")),
            resonator=ReadoutResonator(
                opx_output_I=qubit_wiring.resonator.get_reference("opx_output_I"),
                opx_output_Q=qubit_wiring.resonator.get_reference("opx_output_Q"),
                opx_input_I=qubit_wiring.resonator.get_reference("opx_input_I"),
                opx_input_Q=qubit_wiring.resonator.get_reference("opx_input_Q"),
                digital_outputs={
                    "octave_switch": DigitalOutputChannel(
                        opx_output=qubit_wiring.resonator.get_reference("digital_port"),
                        delay=87,  # 57ns for QOP222 and above
                        buffer=15,  # 18ns for QOP222 and above
                    )
                },
                opx_input_offset_I=0.0,
                opx_input_offset_Q=0.0,
                frequency_converter_up=qubit_wiring.resonator.get_reference(
                    "frequency_converter_up"
                ),
                frequency_converter_down=qubit_wiring.resonator.get_reference(
                    "frequency_converter_down"
                ),
                intermediate_frequency=-250 * u.MHz,
                depletion_time=1 * u.us,
            ),
        )
        # Add the transmon to the machine
        machine.qubits[transmon.name] = transmon
        # Add the transmon to the active qubits
        machine.active_qubit_names.append(transmon.name)
        # Add the set of default pulses
        # TODO: improve this function
        add_default_transmon_pulses(transmon)
        # Set the Octave
        # TODO: why not doing it directly in octave.initialize_frequency_converters(output_mode="always_on", LO_frequency=4.7e9, gain=0)
        RF_output = transmon.xy.frequency_converter_up
        RF_output.channel = transmon.xy.get_reference()
        RF_output.output_mode = "always_on"
        # RF_output.output_mode = "triggered"
        RF_output.LO_frequency = 4.7 * u.GHz
        # TODO: why? the LO should be in the config and the Octave will automatically be set to it when opening a QM
        print(f"Please set the LO frequency of {RF_output.get_reference()}")
        # TODO: same as above
        print(
            f"Please set the output mode of {RF_output.get_reference()} to always_on or triggered"
        )

    # Only set resonator RF outputs once
    # TODO: same as above
    RF_output_resonator = transmon.resonator.frequency_converter_up
    RF_output_resonator.channel = transmon.resonator.get_reference()
    RF_output_resonator.output_mode = "always_on"
    # RF_output_resonator.output_mode = "triggered"
    RF_output_resonator.LO_frequency = 6.2 * u.GHz
    print(f"Please set the LO frequency of {RF_output_resonator.get_reference()}")
    print(
        f"Please set the output mode of {RF_output_resonator.get_reference()} to always_on or triggered"
    )

    RF_input_resonator = transmon.resonator.frequency_converter_down
    RF_input_resonator.channel = transmon.resonator.get_reference()
    RF_input_resonator.LO_frequency = 6.2 * u.GHz
    print(f"Please set the LO frequency of {RF_input_resonator.get_reference()}")

    # Add qubit pairs along with couplers
    for qubit_pair_wairing in machine.wiring.qubit_pairs:
        qubit_pair = TransmonPair(
            qubit_control=qubit_pair_wairing.get_reference("qubit_control"),
            qubit_target=qubit_pair_wairing.get_reference("qubit_target"),
            coupler=TunableCoupler(
                opx_output=qubit_pair_wairing.coupler.get_reference("opx_output")
            ),
        )
        machine.qubit_pairs.append(qubit_pair)
        machine.active_qubit_pair_names.append(qubit_pair.name)
        add_default_transmon_pair_pulses(qubit_pair)

    # Add additional input ports for calibrating the mixers
    # TODO: what is this for? Shouldn't it belong to the wiring?
    machine.ports.get_analog_input("con1", 2, 1, create=True)
    machine.ports.get_analog_input("con1", 2, 2, create=True)

    return machine


if __name__ == "__main__":
    folder = Path(__file__).parent
    quam_folder = folder / "quam_state"
    # TODO: what is this for? WHy not specifying OPX+/Octave/LF/MW in the wiring?
    quam_class = FEMQuAM
    using_opx_1000 = quam_class is FEMQuAM
    # module refers to the FEM number (OPX1000) or OPX+ connection index (OPX+)
    # TODO: need to add chassis & MW fem
    custom_port_wiring = {
        "qubits": {
            "q1": {
                "res": (1, 1, 1, 1),  # (module, i_ch, octave, octave_ch)
                "xy": (1, 3, 1, 2),  # (module, i_ch, octave, octave_ch)
                "flux": (2, 5),  # (module, i_ch)
            },
            "q2": {
                "res": (1, 1, 1, 1),
                "xy": (1, 5, 1, 3),
                "flux": (3, 1),
            },
            "q3": {
                "res": (1, 1, 1, 1),
                "xy": (1, 7, 1, 4),
                "flux": (3, 2),
            },
            "q4": {
                "res": (1, 1, 1, 1),
                "xy": (2, 1, 2, 1),
                "flux": (3, 3),
            },
            "q5": {
                "res": (1, 1, 1, 1),
                "xy": (2, 3, 2, 2),
                "flux": (3, 4),
            },
        },
        "qubit_pairs": {
            # (module, ch)
            "q12": {"coupler": (3, 5)},
            "q23": {"coupler": (3, 6)},
            "q34": {"coupler": (3, 7)},
            "q45": {"coupler": (3, 8)},
        },
    }

    # TODO: let's first do everything with manual wiring for simplicity
    # port_allocation = default_port_allocation(
    #     num_qubits=2,
    #     using_opx_1000=using_opx_1000,
    #     starting_fem=3
    # )
    # TODO: why not having this function part of create_wiring, since the user doesn't care about it?
    port_allocation = custom_port_allocation(custom_port_wiring)
    # TODO: can we re-wire after the QuAM has been created?
    #  Use-case: I have 5 qubits on my chip but only 1 OPX+Octave so I want to be able to update the wiring based on
    #  the active qubits.
    wiring = create_wiring(port_allocation, using_opx_1000=using_opx_1000)
    machine = create_quam_superconducting(quam_class, wiring)

    machine.save(quam_folder, content_mapping={"wiring.json": {"wiring", "network"}})

    qua_file = folder / "qua_config.json"
    qua_config = machine.generate_config()
    json.dump(qua_config, qua_file.open("w"), indent=4)

    # quam_loaded = QuAM.load(quam_folder)
    # qua_config_loaded = quam_loaded.generate_config()
    # json.dump(qua_config_loaded, qua_file.open("w"), indent=4)
