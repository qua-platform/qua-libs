from quam.core import QuamRoot
from quam.components import *
from quam.components.channels import IQChannel, DigitalOutputChannel
from quam.components.pulses import SquareReadoutPulse, Pulse
from quam_components import *
from qm.octave import QmOctaveConfig
from qualang_tools.units import unit
from pathlib import Path
import json


def create_quam_superconducting_referenced(num_qubits: int) -> (QuamRoot, QmOctaveConfig):
    """Create a QuAM with a number of qubits.

    Args:
        num_qubits (int): Number of qubits to create.

    Returns:
        QuamRoot: A QuAM with the specified number of qubits.
    """
    # TODO: how do I add a qubit to an existing quam without loosing the saved values
    # Class containing tools to help handling units and conversions.
    u = unit(coerce_to_integer=True)
    # Initiate the QuAM class
    quam = QuAM()
    # Define the connectivity
    quam.wiring = {
        "qubits": [
            {
                "port_I": ("con1", 4 * k + 3),
                "port_Q": ("con1", 4 * k + 4),
                "port_Z": ("con1", 4 * k + 5),
                "digital_port": ("con1", 4 * k + 3),
            }
            for k in range(num_qubits)
        ],
        "resonator": {
            "opx_output_I": ("con1", 1),
            "opx_output_Q": ("con1", 2),
            "opx_input_I": ("con1", 1),
            "opx_input_Q": ("con1", 2),
            "digital_port": ("con1", 1),
        },
    }
    quam.network = {
        "host": "172.16.33.101",
        "cluster_name": "Cluster_81",
        "octave_ip": "172.16.33.101",
        "octave_port": 11050,
        "data_folder": "C:/git/qua-libs/Quantum-Control-Applications/Superconducting/quam/Two-Flux-Tunable-Transmons/DATA",
    }
    # Add the Octave to the quam
    octave = Octave(
        name="octave1",
        ip="172.16.33.101",
        port=11050,
    )
    quam.octave = octave
    octave.initialize_frequency_converters()
    octave_config = octave.get_octave_config()
    # Add the transmon components (xy, z and resonator) to the quam
    for idx in range(num_qubits):
        # Create qubit components
        transmon = Transmon(
            id=idx,
            xy=IQChannel(
                opx_output_I=f"#/wiring/qubits/{idx}/port_I",
                opx_output_Q=f"#/wiring/qubits/{idx}/port_Q",
                frequency_converter_up=octave.RF_outputs[2 * (idx + 1)].get_reference(),
                intermediate_frequency=100 * u.MHz,
                digital_outputs={
                    "octave_switch": DigitalOutputChannel(
                        opx_output=f"#/wiring/qubits/{idx}/digital_port", delay=87, buffer=15
                    )
                },  # 57ns, 18ns for QOP222 and above
            ),
            z=FluxLine(opx_output=f"#/wiring/qubits/{idx}/port_Z"),
            resonator=ReadoutResonator(
                opx_output_I="#/wiring/resonator/opx_output_I",
                opx_output_Q="#/wiring/resonator/opx_output_Q",
                opx_input_I="#/wiring/resonator/opx_input_I",
                opx_input_Q="#/wiring/resonator/opx_input_Q",
                digital_outputs={
                    "octave_switch": DigitalOutputChannel(
                        opx_output=f"#/wiring/qubits/{idx}/digital_port", delay=87, buffer=15
                    )
                },  # 57ns, 18ns for QOP222 and above
                opx_input_offset_I=0.0,
                opx_input_offset_Q=0.0,
                frequency_converter_up=octave.RF_outputs[1].get_reference(),
                frequency_converter_down=octave.RF_inputs[1].get_reference(),
                intermediate_frequency=50 * u.MHz,
                depletion_time=1 * u.us,
            ),
        )

        # Add the transmon pulses to the quam
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
        transmon.resonator.operations["readout"] = SquareReadoutPulse(
            length=1 * u.us, amplitude=0.00123, threshold=0.0, digital_marker="ON"
        )
        quam.qubits[transmon.name] = transmon
        quam.active_qubit_names.append(transmon.name)
        # Set the Octave frequency and channels TODO: be careful to set the right upconverters!!
        octave.RF_outputs[2 * (idx + 1)].channel = transmon.xy.get_reference()
        octave.RF_outputs[2 * (idx + 1)].LO_frequency = 7 * u.GHz  # Remember to set the LO frequency
        octave.RF_outputs[2 * (idx + 1)].output_mode = "always_on"

        octave.RF_outputs[1].channel = transmon.resonator.get_reference()
        octave.RF_inputs[1].channel = transmon.resonator.get_reference()
        octave.RF_outputs[1].LO_frequency = 4 * u.GHz
        octave.RF_inputs[1].LO_frequency = 4 * u.GHz
        octave.RF_outputs[1].output_mode = "always_on"

    return quam, octave_config


if __name__ == "__main__":
    folder = Path("")
    folder.mkdir(exist_ok=True)

    machine, _ = create_quam_superconducting_referenced(num_qubits=2)
    machine.save(folder / "quam_machine", content_mapping={"wiring.json": {"wiring", "network"}})
    machine.save(folder / "state.json")

    qua_file = folder / "qua_config.json"
    qua_config = machine.generate_config()
    json.dump(qua_config, qua_file.open("w"), indent=4)

    quam_loaded = QuAM.load("state.json")
    qua_file_loaded = folder / "qua_config2.json"
    qua_config_loaded = quam_loaded.generate_config()
    json.dump(qua_config_loaded, qua_file.open("w"), indent=4)
