from pathlib import Path
import json

from quam.components import *
from quam.components.channels import IQChannel, InOutIQChannel, SingleChannel
from quam.components.pulses import ReadoutPulse
from quam.examples.superconducting_qubits.components import Transmon, QuAM
from qm.octave import QmOctaveConfig
from quam.core import QuamRoot


def create_quam_superconducting_referenced(num_qubits: int) -> (QuamRoot, QmOctaveConfig):
    """Create a QuAM with a number of qubits.

    Args:
        num_qubits (int): Number of qubits to create.

    Returns:
        QuamRoot: A QuAM with the specified number of qubits.
    """
    quam = QuAM()

    # Add the Octave to the quam
    octave = Octave(
        name="octave1",
        ip="172.16.33.101",
        port=11050,
    )
    quam.octave = octave
    octave.initialize_frequency_converters()
    octave.print_summary()
    octave_config = octave.get_octave_config()

    # Define the connectivity
    quam.wiring = {
        "qubits": [
            {
                "port_I": ("con1", 3 * k + 3),
                "port_Q": ("con1", 3 * k + 4),
                "port_Z": ("con1", 3 * k + 5),
            }
            for k in range(num_qubits)  # TODO: what if k>2?
        ],
        "resonator": {
            "opx_output_I": ("con1", 1),
            "opx_output_Q": ("con1", 2),
            "opx_input_I": ("con1", 1),
            "opx_input_Q": ("con1", 2),
        },
    }
    # Add the transmon components (xy, z and resonator) to the quam
    for idx in range(num_qubits):
        # Create qubit components
        transmon = Transmon(
            id=idx,
            xy=IQChannel(
                opx_output_I=f"#/wiring/qubits/{idx}/port_I",
                opx_output_Q=f"#/wiring/qubits/{idx}/port_Q",
                frequency_converter_up=octave.RF_outputs[2].get_reference(),
                intermediate_frequency=100e6,
            ),
            z=SingleChannel(opx_output=f"#/wiring/qubits/{idx}/port_Z"),
            resonator=InOutIQChannel(
                opx_output_I="#/wiring/resonator/opx_output_I",
                opx_output_Q="#/wiring/resonator/opx_output_Q",
                opx_input_I="#/wiring/resonator/opx_input_I",
                opx_input_Q="#/wiring/resonator/opx_input_Q",
                frequency_converter_up=octave.RF_outputs[1].get_reference(),
                frequency_converter_down=octave.RF_inputs[1].get_reference(),
            ),
        )
        # Add the transmon pulses to the quam
        transmon.xy.operations["x180"] = pulses.DragPulse(
            amplitude=0.1, sigma=7, alpha=0, anharmonicity=-200e6, length=36, axis_angle=0
        )
        transmon.xy.operations["x90"] = pulses.DragPulse(
            amplitude=0.1/2, sigma=7, alpha=0, anharmonicity=-200e6, length=36, axis_angle=0
        )
        transmon.xy.operations["-x90"] = pulses.DragPulse(
            amplitude=-0.1 / 2, sigma=7, alpha=0, anharmonicity=-200e6, length=36, axis_angle=0
        )
        transmon.xy.operations["y180"] = pulses.DragPulse(
            amplitude=0.1, sigma=7, alpha=0, anharmonicity=-200e6, length=36, axis_angle=90
        )
        transmon.xy.operations["y90"] = pulses.DragPulse(
            amplitude=0.1 / 2, sigma=7, alpha=0, anharmonicity=-200e6, length=36, axis_angle=90
        )
        transmon.xy.operations["-y90"] = pulses.DragPulse(
            amplitude=-0.1 / 2, sigma=7, alpha=0, anharmonicity=-200e6, length=36, axis_angle=90
        )
        transmon.xy.operations["saturation"] = pulses.SquarePulse(amplitude=0.25, length=10_000, axis_angle=0)
        transmon.z.operations["const"] = pulses.SquarePulse(amplitude=0.1, length=100)
        # transmon.resonator.operations["readout"] = pulses.ReadoutPulse(length=1000)
        quam.qubits.append(transmon)
        # Set the Octave frequency and channels
        octave.RF_outputs[2].channel = transmon.xy.get_reference()
        octave.RF_outputs[2].LO_frequency = 7e9  # Remember to set the LO frequency

        octave.RF_outputs[1].channel = transmon.resonator.get_reference()
        octave.RF_inputs[1].channel = transmon.resonator.get_reference()
        octave.RF_outputs[1].LO_frequency = 4e9
        octave.RF_inputs[1].LO_frequency = 4e9
    return quam, octave_config


if __name__ == "__main__":
    folder = Path("")
    folder.mkdir(exist_ok=True)

    quam, octave_config = create_quam_superconducting_referenced(num_qubits=2)
    quam.save(folder / "quam", content_mapping={"wiring.json": "wiring"})

    qua_file = folder / "qua_config.json"
    qua_config = quam.generate_config()
    json.dump(qua_config, qua_file.open("w"), indent=4)

    quam_loaded = QuAM.load(folder / "quam")

    qua_file_loaded = folder / "qua_config2.json"
    qua_config_loaded = quam_loaded.generate_config()
    # json.dump(qua_config_loaded, qua_file.open("w"), indent=4)
