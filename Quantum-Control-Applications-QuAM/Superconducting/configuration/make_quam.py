import math
import os.path
from typing import Dict

from quam.components import Octave, IQChannel, DigitalOutputChannel
from quam_components import Transmon, ReadoutResonator, QuAM, FluxLine
from qualang_tools.units import unit
from pathlib import Path
import json

from make_pulses import add_default_transmon_pulses
from make_wiring import create_default_wiring

# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)


def create_quam_superconducting(num_qubits: int = None, wiring: dict = None,
                                octaves: Dict[str, Octave] = None, using_opx_1000: bool = True) -> QuAM:
    """Create a QuAM with a number of qubits.

    Args:
        num_qubits (int): Number of qubits to create.
            Will autogenerate the wiring, and should thus only be provided if wiring is not provided.
            Current autogeneration limited to 2 qubits
        wiring (dict): Wiring of the qubits.
            Should be provided if num_qubits is not provided.
        using_opx_1000 (bool): Whether to use the OPX1000 or OPX+.
            Will generate the analog output/input ports according to the chosen OPX type's format.

    Returns:
        QuamRoot: A QuAM with the specified number of qubits.
    """
    # Initiate the QuAM class
    machine = QuAM()

    # Define the connectivity
    if wiring is not None:
        machine.wiring = wiring
    elif num_qubits is not None:
        machine.wiring = create_default_wiring(num_qubits, using_opx_1000)
    else:
        raise ValueError("Either num_qubits or wiring must be provided.")

    host_ip = "172.16.33.101"
    octave_ip = host_ip  # or "192.168.88.X" if configured internally
    octave_port = 11050  # 11XXX where XXX are the last digits of the Octave IP or 80 if configured internally

    machine.network = {
        "host": host_ip,
        "cluster_name": "Cluster_81",
        "octave_ip": octave_ip,
        "octave_port": octave_port,
        "data_folder": "C:/git/qua-libs/Quantum-Control-Applications-QuAM/Superconducting/data",
    }
    print("Please update the default network settings in: quam.network")

    if octaves is not None:
        machine.octaves = octaves
        print("If you haven't configured the octaves, please run: octave.initialize_frequency_converters()")
    else:
        # Add the Octave to the quam
        for i in range(math.ceil(num_qubits / 4)):
            # Assumes 1 Octave for every 4 qubits
            octave = Octave(
                name=f"octave{i+1}",
                ip=machine.network["octave_ip"],
                port=machine.network["octave_port"],
                calibration_db_path=os.path.dirname(__file__)
            )
            machine.octaves[f"octave{i+1}"] = octave
            octave.initialize_frequency_converters()
            print("Please update the octave settings in: quam.octave")

    # Add the transmon components (xy, z and resonator) to the quam
    for qubit_name, qubit_wiring in machine.wiring.qubits.items():
        # Create qubit components
        transmon = Transmon(
            id=qubit_name,
            xy=IQChannel(
                opx_output_I=qubit_wiring.xy.get_reference("opx_output_I"),
                opx_output_Q=qubit_wiring.xy.get_reference("opx_output_Q"),
                frequency_converter_up=qubit_wiring.xy.frequency_converter_up.get_reference(),
                intermediate_frequency=100 * u.MHz,
                digital_outputs={
                    "octave_switch": DigitalOutputChannel(
                        opx_output=qubit_wiring.get_reference("opx_output_digital"),
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
                        opx_output=qubit_wiring.get_reference("opx_output_digital"),
                        delay=87,  # 57ns for QOP222 and above
                        buffer=15,  # 18ns for QOP222 and above
                    )
                },
                opx_input_offset_I=0.0,
                opx_input_offset_Q=0.0,
                frequency_converter_up=qubit_wiring.resonator.frequency_converter_up.get_reference(),
                frequency_converter_down=qubit_wiring.resonator.frequency_converter_down.get_reference(),
                intermediate_frequency=50 * u.MHz,
                depletion_time=1 * u.us,
            ),
        )
        machine.qubits[transmon.name] = transmon
        machine.active_qubit_names.append(transmon.name)

        add_default_transmon_pulses(transmon)

        RF_output = transmon.xy.frequency_converter_up
        RF_output.channel = transmon.xy.get_reference()
        RF_output.output_mode = "always_on"
        RF_output.LO_frequency = 7 * u.GHz
        print(f"Please set the LO frequency of {RF_output.get_reference()}")

    # Only set resonator RF outputs once
    RF_output_resonator = transmon.resonator.frequency_converter_up
    RF_output_resonator.channel = transmon.resonator.get_reference()
    RF_output_resonator.output_mode = "always_on"
    RF_output_resonator.LO_frequency = 4 * u.GHz
    print(f"Please set the LO frequency of {RF_output_resonator.get_reference()}")

    RF_input_resonator = transmon.resonator.frequency_converter_down
    RF_input_resonator.channel = transmon.resonator.get_reference()
    RF_input_resonator.LO_frequency = 4 * u.GHz
    print(f"Please set the LO frequency of {RF_input_resonator.get_reference()}")

    return machine


if __name__ == "__main__":
    folder = Path(__file__).parent
    quam_folder = folder / "quam_state"

    machine = create_quam_superconducting(num_qubits=5)
    machine.save(quam_folder, content_mapping={"wiring.json": {"wiring", "network"}})

    qua_file = folder / "qua_config.json"
    qua_config = machine.generate_config()
    json.dump(qua_config, qua_file.open("w"), indent=4)

    quam_loaded = QuAM.load(quam_folder)
    qua_config_loaded = quam_loaded.generate_config()
    json.dump(qua_config_loaded, qua_file.open("w"), indent=4)
