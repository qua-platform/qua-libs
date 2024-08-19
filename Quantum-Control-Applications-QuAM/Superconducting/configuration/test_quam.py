import os.path
from typing import Dict, List, Type, TypeVar
from pathlib import Path
import json

from qualang_tools.units import unit
from quam.components import Octave, IQChannel, DigitalOutputChannel
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
from test_wiring import create_wiring, CLUSTER_CONFIG

# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)

# TODO: what is this for?
QuamTypes = TypeVar("QuamTypes", OPXPlusQuAM, FEMQuAM)


def create_quam_superconducting(
    quam_class: Type[QuamTypes],
    wiring: dict = None,
    octaves: Dict[str, Octave] = None,
) -> QuamTypes:
    # TODO: docstring
    """Create a QuAM with a number of qubits.

    Args:
        wiring (dict): Wiring of the qubits.

    Returns:
        QuamRoot: A QuAM with the specified number of qubits.
    """
    # Initiate the QuAM class
    machine = quam_class()

    # Define the connectivity
    if wiring is not None:
        machine.wiring = wiring
    else:
        raise ValueError("Wiring must be provided.")
    # TODO: this doesn't belong here, put into wiring or architecture
    host_ip = "192.168.1.52"
    # TODO: same here, the number of octaves and their IP/port should be part of the wiring
    octave_ips = [host_ip] * 3
    # or "192.168.88.X" if configured internally
    # octave_ips = ["192.168.88.251", ...]
    octave_ports = [11243, 11240, 11241]  # 11XXX where XXX are the last digits of the Octave IP
    # or 80 if configured internally
    # octave_ips = [80] * 3

    # Build this in the witing or network.json
    machine.network = {
        "host": host_ip,
        "cluster_name": "Cluster_Name",
        "octave_ips": octave_ips,
        "octave_ports": octave_ports,
        "data_folder": r"C:\data\path\to\folder",
    }
    print("Please update the default network settings in: quam.network")

    # This function should assume there are octaves, this should be inferred from the wiring because there is also mw fem
    if octaves is not None:
        machine.octaves = octaves
        print("If you haven't configured the octaves, please run: octave.initialize_frequency_converters()")

    else:
        # Add the Octave to the quam
        # TODO: make a utility function to get the octaves from the wiring
        for i in range(len(octave_ips)):
            octave = Octave(
                name=f"octave{i+1}",
                ip=machine.network["octave_ips"][i],
                port=machine.network["octave_ports"][i],
                calibration_db_path=os.path.dirname(__file__),
            )
            machine.octaves[f"octave{i+1}"] = octave
            octave.initialize_frequency_converters()

    # Add the transmon components (xy, z and resonator) to the quam
    for qubit_name, qubit_wiring in machine.wiring.qubits.items():
        # Create all necessary ports
        # TODO: hide it from the customer + what does it do?
        # TODO: why not adding the port directly  when building the base components?
        machine.ports.reference_to_port(qubit_wiring.xy.get_unreferenced_value("digital_port"), create=True)
        machine.ports.reference_to_port(qubit_wiring.xy.get_unreferenced_value("opx_output_I"), create=True)
        machine.ports.reference_to_port(qubit_wiring.xy.get_unreferenced_value("opx_output_Q"), create=True)
        machine.ports.reference_to_port(qubit_wiring.z.get_unreferenced_value("opx_output"), create=True)
        machine.ports.reference_to_port(qubit_wiring.resonator.get_unreferenced_value("opx_output_I"), create=True)
        machine.ports.reference_to_port(qubit_wiring.resonator.get_unreferenced_value("opx_output_Q"), create=True)
        machine.ports.reference_to_port(qubit_wiring.resonator.get_unreferenced_value("opx_input_I"), create=True)
        machine.ports.reference_to_port(qubit_wiring.resonator.get_unreferenced_value("opx_input_Q"), create=True)
        machine.ports.reference_to_port(qubit_wiring.resonator.get_unreferenced_value("digital_port"), create=True)

        # Create qubit components
        transmon = Transmon(
            id=qubit_name,
            xy=IQChannel(
                opx_output_I=qubit_wiring.xy.get_unreferenced_value("opx_output_I"),
                opx_output_Q=qubit_wiring.xy.get_unreferenced_value("opx_output_Q"),
                frequency_converter_up=qubit_wiring.xy.frequency_converter_up.get_reference(),
                intermediate_frequency=-200 * u.MHz,
                digital_outputs={
                    "octave_switch": DigitalOutputChannel(
                        opx_output=qubit_wiring.xy.get_unreferenced_value("digital_port"),
                        delay=87,  # 57ns for QOP222 and above
                        buffer=15,  # 18ns for QOP222 and above
                    )
                },
            ),
            z=FluxLine(opx_output=qubit_wiring.z.get_unreferenced_value("opx_output")),
            resonator=ReadoutResonator(
                opx_output_I=qubit_wiring.resonator.get_unreferenced_value("opx_output_I"),
                opx_output_Q=qubit_wiring.resonator.get_unreferenced_value("opx_output_Q"),
                opx_input_I=qubit_wiring.resonator.get_unreferenced_value("opx_input_I"),
                opx_input_Q=qubit_wiring.resonator.get_unreferenced_value("opx_input_Q"),
                digital_outputs={
                    "octave_switch": DigitalOutputChannel(
                        opx_output=qubit_wiring.resonator.get_unreferenced_value("digital_port"),
                        delay=87,  # 57ns for QOP222 and above
                        buffer=15,  # 18ns for QOP222 and above
                    )
                },
                opx_input_offset_I=0.0,
                opx_input_offset_Q=0.0,
                frequency_converter_up=qubit_wiring.resonator.frequency_converter_up.get_reference(),
                frequency_converter_down=qubit_wiring.resonator.frequency_converter_down.get_reference(),
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

    # Only set resonator RF outputs once
    RF_output_resonator = transmon.resonator.frequency_converter_up
    RF_output_resonator.channel = transmon.resonator.get_reference()
    RF_output_resonator.output_mode = "always_on"
    # RF_output_resonator.output_mode = "triggered"
    RF_output_resonator.LO_frequency = 6.2 * u.GHz

    RF_input_resonator = transmon.resonator.frequency_converter_down
    RF_input_resonator.channel = transmon.resonator.get_reference()
    RF_input_resonator.LO_frequency = 6.2 * u.GHz

    # Add qubit pairs along with couplers
    for i, qubit_pair_wiring in enumerate(machine.wiring.qubit_pairs):
        qubit_control_name = qubit_pair_wiring.qubit_control.name
        qubit_target_name = qubit_pair_wiring.qubit_target.name
        qubit_pair_name = f"{qubit_control_name}_{qubit_target_name}"
        coupler_name = f"coupler_{qubit_pair_name}"

        machine.ports.reference_to_port(qubit_pair_wiring.coupler.get_unreferenced_value("opx_output"), create=True)

        coupler = TunableCoupler(
            id=coupler_name, opx_output=qubit_pair_wiring.coupler.get_unreferenced_value("opx_output")
        )
        # Note: The Q channel is set to the I channel plus one. # TODO: what does it mean? I/Q should be set in the single transmon no?
        qubit_pair = TransmonPair(
            id=qubit_pair_name,
            qubit_control=qubit_pair_wiring.get_unreferenced_value("qubit_control"),
            qubit_target=qubit_pair_wiring.get_unreferenced_value("qubit_target"),
            coupler=coupler,
        )
        machine.qubit_pairs.append(qubit_pair)
        machine.active_qubit_pair_names.append(qubit_pair_name)
        add_default_transmon_pair_pulses(qubit_pair)

    # Add additional input ports for calibrating the mixers
    # TODO: what is this for?
    # machine.ports.get_analog_input("con1", 1, create=True)
    # machine.ports.get_analog_input("con1", 2, create=True)

    return machine


if __name__ == "__main__":
    folder = Path(__file__).parent
    quam_folder = folder / "quam_state"
    # TODO: what is this for? WHy not specifying OPX+/Octave/LF/MW in the wiring?
    quam_class = FEMQuAM
    # module refers to the FEM number (OPX1000) or OPX+ connection index (OPX+)
    # TODO: need to add chassis & MW fem
    custom_port_wiring = {
        "cluster": {"opx+": 0, "octaves": 2, "opx1000": 1, "lf-fems": 3, "mw-fems": 0,
                    "octave_connectivity": "default"},
        "qubits": {
            "q1": {
                "res": (1, 1, 1, 1),  # (module, i_ch, octave, octave_ch)
                "xy": (1, 1, 1, 2),  # (module, i_ch, octave, octave_ch)
                "flux": (1, 2, 5),  # (module, i_ch)
            },
            "q2": {
                "res": (1, 1, 1, 1),
                "xy": (1, 1, 1, 3),
                "flux": (1, 3, 1),
            },
            "q3": {
                "res": (1, 1, 1, 1),
                "xy": (1, 1, 1, 4),
                "flux": (1, 3, 2),
            },
            "q4": {
                "res": (1, 1, 1, 1),
                "xy": (1, 2, 2, 1),
                "flux": (1, 3, 3),
            },
            "q5": {
                "res": (1, 1, 1, 1),
                "xy": (1, 2, 2, 2),
                "flux": (1, 3, 4),
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
    # custom_port_wiring = {
    #     "cluster": {"opx+": 1, "octaves": 1, "opx1000": 0, "lf-fems": 0, "mw-fems": 0,
    #                 "octave_connectivity": "default"},
    #     "qubits": {
    #         "q1": {
    #             "res": (1, 1, 1),
    #             "xy": (1, 1, 2),
    #             "flux": (1, 5),
    #         },
    #         "q2": {
    #             "res": (1, 1, 1),
    #             "xy": (1, 1, 4),
    #             "flux": (1, 6),
    #         },
    #     },
    #     "qubit_pairs": {
    #         # (module, ch)
    #         "q12": {"coupler": (1, 5)},
    #     },
    # }
    # TODO: let's first do everything with manual wiring for simplicity
    # port_allocation = default_port_allocation(
    #     num_qubits=2,
    #     using_opx_1000=using_opx_1000,
    #     starting_fem=3
    # )
    # TODO: can we re-wire after the QuAM has been created?
    #  Use-case: I have 5 qubits on my chip but only 1 OPX+Octave so I want to be able to update the wiring based on
    #  the active qubits.
    wiring = create_wiring(custom_port_wiring, CLUSTER_CONFIG)
    machine = create_quam_superconducting(quam_class, wiring)

    machine.save(quam_folder, content_mapping={"wiring.json": {"wiring", "network"}})

    qua_file = folder / "qua_config.json"
    qua_config = machine.generate_config()
    json.dump(qua_config, qua_file.open("w"), indent=4)

    # quam_loaded = QuAM.load(quam_folder)
    # qua_config_loaded = quam_loaded.generate_config()
    # json.dump(qua_config_loaded, qua_file.open("w"), indent=4)
