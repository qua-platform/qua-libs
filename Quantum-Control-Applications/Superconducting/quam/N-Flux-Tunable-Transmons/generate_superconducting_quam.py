from typing import Dict
from quam.components import pulses, Octave, IQChannel, DigitalOutputChannel

from components import Transmon, ReadoutResonator, QuAM, FluxLine
from qualang_tools.units import unit
from pathlib import Path
import json


# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)


def create_default_wiring(num_qubits: int) -> dict:
    wiring = {"qubits": {}}

    #
    for q_idx in range(1, num_qubits + 1):
        octave_output_idx = 1 + q_idx
        wiring["qubits"][f"q{q_idx}"] = {
            "xy": {
                "opx_output_I": ("con1", 3 * q_idx),  # 3, 6
                "opx_output_Q": ("con1", 3 * q_idx + 1),  # 4, 7
                "frequency_converter_up": f"#/octaves/octave1/RF_outputs/{octave_output_idx}",
            },
            "z": {"opx_output": ("con1", 3 * q_idx + 2)},  # 5, 8
            "opx_output_digital": ("con1", 1),
            "resonator": {
                "opx_output_I": ("con1", 1),
                "opx_output_Q": ("con1", 2),
                "opx_input_I": ("con1", 1),
                "opx_input_Q": ("con1", 2),
                "digital_port": ("con1", 1),
                "frequency_converter_up": "#/octaves/octave1/RF_outputs/1",
                "frequency_converter_down": "#/octaves/octave1/RF_inputs/1",
            },
        }
    return wiring


def add_default_transmon_pulses(transmon):
    # TODO: make sigma=length/5
    # TODO: Make gates amplitude a reference to x180 amplitude
    transmon.xy.operations["x180_DragGaussian"] = pulses.DragPulse(
        amplitude=0.1,
        sigma=7,
        alpha=1.0,
        anharmonicity=f"#/qubits/{transmon.name}/anharmonicity",
        length=40,
        axis_angle=0,
        digital_marker="ON",
    )
    transmon.xy.operations["x90_DragGaussian"] = pulses.DragPulse(
        amplitude=0.1 / 2,
        sigma="#../x180_DragGaussian/sigma",
        alpha="#../x180_DragGaussian/alpha",
        anharmonicity="#../x180_DragGaussian/anharmonicity",
        length="#../x180_DragGaussian/length",
        axis_angle=0,
        digital_marker="ON",
    )
    transmon.xy.operations["-x90_DragGaussian"] = pulses.DragPulse(
        amplitude=-0.1 / 2,
        sigma="#../x180_DragGaussian/sigma",
        alpha="#../x180_DragGaussian/alpha",
        anharmonicity="#../x180_DragGaussian/anharmonicity",
        length="#../x180_DragGaussian/length",
        axis_angle=0,
        digital_marker="ON",
    )
    transmon.xy.operations["y180_DragGaussian"] = pulses.DragPulse(
        amplitude="#../x180_DragGaussian/amplitude",
        sigma="#../x180_DragGaussian/sigma",
        alpha="#../x180_DragGaussian/alpha",
        anharmonicity="#../x180_DragGaussian/anharmonicity",
        length="#../x180_DragGaussian/length",
        axis_angle=90,
        digital_marker="ON",
    )
    transmon.xy.operations["y90_DragGaussian"] = pulses.DragPulse(
        amplitude=0.1 / 2,
        sigma="#../x180_DragGaussian/sigma",
        alpha="#../x180_DragGaussian/alpha",
        anharmonicity="#../x180_DragGaussian/anharmonicity",
        length="#../x180_DragGaussian/length",
        axis_angle=90,
        digital_marker="ON",
    )
    transmon.xy.operations["-y90_DragGaussian"] = pulses.DragPulse(
        amplitude=-0.1 / 2,
        sigma="#../x180_DragGaussian/sigma",
        alpha="#../x180_DragGaussian/alpha",
        anharmonicity="#../x180_DragGaussian/anharmonicity",
        length="#../x180_DragGaussian/length",
        axis_angle=90,
        digital_marker="ON",
    )
    transmon.xy.operations["x180_Square"] = pulses.SquarePulse(
        amplitude=0.25, length=100, axis_angle=0, digital_marker="ON"
    )
    transmon.xy.operations["x90_Square"] = pulses.SquarePulse(
        amplitude=0.25 / 2, length="#../x180_Square/length", axis_angle=0, digital_marker="ON"
    )
    transmon.xy.operations["-x90_Square"] = pulses.SquarePulse(
        amplitude=-0.25 / 2, length="#../x180_Square/length", axis_angle=0, digital_marker="ON"
    )
    transmon.xy.operations["y180_Square"] = pulses.SquarePulse(
        amplitude=0.25, length="#../x180_Square/length", axis_angle=90, digital_marker="ON"
    )
    transmon.xy.operations["y90_Square"] = pulses.SquarePulse(
        amplitude=0.25 / 2, length="#../x180_Square/length", axis_angle=90, digital_marker="ON"
    )
    transmon.xy.operations["-y90_Square"] = pulses.SquarePulse(
        amplitude=-0.25 / 2, length="#../x180_Square/length", axis_angle=90, digital_marker="ON"
    )
    transmon.set_gate_shape("DragGaussian")

    transmon.xy.operations["saturation"] = pulses.SquarePulse(
        amplitude=0.25, length=10 * u.us, axis_angle=0, digital_marker="ON"
    )
    transmon.z.operations["const"] = pulses.SquarePulse(amplitude=0.1, length=100)
    transmon.resonator.operations["readout"] = pulses.SquareReadoutPulse(
        length=1 * u.us, amplitude=0.00123, threshold=0.0, digital_marker="ON"
    )


def create_quam_superconducting(num_qubits: int = None, wiring: dict = None, octaves: Dict[str, Octave] = None) -> QuAM:
    """Create a QuAM with a number of qubits.

    Args:
        num_qubits (int): Number of qubits to create.
            Will autogenerate the wiring, and should thus only be provided if wiring is not provided.
            Current autogeneration limited to 2 qubits
        wiring (dict): Wiring of the qubits.
            Should be provided if num_qubits is not provided.

    Returns:
        QuamRoot: A QuAM with the specified number of qubits.
    """
    # Initiate the QuAM class
    machine = QuAM()

    # Define the connectivity
    if wiring is not None:
        machine.wiring = wiring
    elif num_qubits is not None:
        machine.wiring = create_default_wiring(num_qubits)
    else:
        raise ValueError("Either num_qubits or wiring must be provided.")

    host_ip = "172.16.33.101"
    octave_ip = host_ip  # or "192.168.88.X" if configured internally
    octave_port = 11050  # 11XXX where XXX are the last digits of the Octave IP or 80 if configured internally

    machine.network = {
        "host": host_ip,
        "cluster_name": "Cluster_81",
        "octave_ip": host_ip,
        "octave_port": octave_port,
        "data_folder": "C:/git/qua-libs/Quantum-Control-Applications/Superconducting/quam/Two-Flux-Tunable-Transmons/DATA",
    }
    print("Please update the default network settings in: quam.network")

    if octaves is not None:
        machine.octaves = octaves
        print("If you haven't configured the octaves, please run: octave.initialize_frequency_converters()")
    else:
        # Add the Octave to the quam
        octave = Octave(
            name="octave1",
            ip=octave_ip,
            port=octave_port,
        )
        machine.octaves["octave1"] = octave
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
    folder = Path("")
    folder.mkdir(exist_ok=True)

    machine = create_quam_superconducting(num_qubits=2)
    machine.save(folder / "quam_machine", content_mapping={"wiring.json": {"wiring", "network"}})
    machine.save(folder / "state.json")

    qua_file = folder / "qua_config.json"
    qua_config = machine.generate_config()
    json.dump(qua_config, qua_file.open("w"), indent=4)

    quam_loaded = QuAM.load("state.json")
    qua_file_loaded = folder / "qua_config2.json"
    qua_config_loaded = quam_loaded.generate_config()
    json.dump(qua_config_loaded, qua_file.open("w"), indent=4)
