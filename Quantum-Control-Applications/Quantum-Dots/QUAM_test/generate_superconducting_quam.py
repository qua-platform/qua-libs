from pathlib import Path
import json

from quam.components import *
from quam.components.channels import InOutSingleChannel, DigitalOutputChannel
from quam.components.pulses import ConstantReadoutPulse
from components import QuAM, VirtualGateSet, SingleChannel, Channel, StickyChannelAddon
from qm.octave import QmOctaveConfig
from quam.core import QuamRoot

from qualang_tools.units import unit


def create_quam_quantumdots_referenced(gates_list: list[str]) -> (QuamRoot, QmOctaveConfig):
    """Create a QuAM with a number of qubits.

    Args:
        gates_list: List of the gates to implement

    Returns:
        QuamRoot: A QuAM with the specified number of qubits.
    """
    # Class containing tools to help handling units and conversions.
    u = unit(coerce_to_integer=True)
    quam = QuAM()

    # Add the Octave to the quam
    # octave = Octave(
    #     name="octave1",
    #     ip="172.16.33.101",
    #     port=11050,
    # )
    # quam.octave = octave
    # octave.initialize_frequency_converters()
    # octave.print_summary()
    # octave_config = octave.get_octave_config()

    # Define the connectivity
    quam.wiring = {
        "gates": [{"port": ("con1", k + 1)} for k in range(len(gates_list))],
        "resonator": {
            "opx_output": ("con1", 4),
            "opx_input": ("con1", 1),
        },
        "tia": {
            "opx_output": ("con1", 4),
            "opx_input": ("con1", 2),
        },
        "qdac_triggers": [{"digital_marker": ("con1", 1)}, {"digital_marker": ("con1", 2)}],
    }
    quam.network = {"host": "172.16.33.101", "cluster_name": "Cluster_81"}
    # Add gates
    quam.gates = {
        name: SingleChannel(
            id=name,
            opx_output=f"#/wiring/gates/{k}/port",
            sticky=StickyChannelAddon(duration=200, digital=False),
        )
        for k, name in enumerate(gates_list)
    }

    quam.resonator = InOutSingleChannel(
        id="resonator",
        opx_output=f"#/wiring/resonator/opx_output",
        opx_input=f"#/wiring/resonator/opx_input",
        intermediate_frequency=50e6,
        time_of_flight=24,
    )
    quam.resonator.operations["readout"] = ConstantReadoutPulse(
        length=1 * u.us, amplitude=0.1, threshold=0.0, axis_angle=None
    )

    quam.tia = InOutSingleChannel(
        id="tia",
        opx_output=f"#/wiring/tia/opx_output",
        opx_input=f"#/wiring/tia/opx_input",
    )
    # axis_angle to None to work with single channel, otherwise IQ
    quam.tia.operations["readout"] = ConstantReadoutPulse(
        length=1 * u.us, amplitude=0.0, threshold=0.0, axis_angle=None
    )

    quam.qdac_triggers["trig1"] = SingleChannel(
        id="trig1",
        digital_outputs={
            "marker1": DigitalOutputChannel(opx_output="#/wiring/qdac_triggers/0/digital_marker", delay=0, buffer=0)
        },
        opx_output=("con1", 4)
    )
    quam.qdac_triggers["trig2"] = SingleChannel(
        id="trig2",
        digital_outputs={
            "marker2": DigitalOutputChannel(opx_output="#/wiring/qdac_triggers/1/digital_marker", delay=0, buffer=0)
        },
        opx_output=("con1", 4)
    )
    quam.qdac_triggers["trig1"].operations["trigger"] = pulses.SquarePulse(amplitude=0.0, length=16, axis_angle=None, digital_marker="ON")
    quam.qdac_triggers["trig2"].operations["trigger"] = pulses.SquarePulse(amplitude=0.10, length=16, axis_angle=None, digital_marker="ON")


    quam.virtual_gate_set = VirtualGateSet(
        gates=["#/gates/P1", "#/gates/P2"],  # Be careful with ordering
        virtual_gates={"eps": [-11.6, 6.51], "U": [11.6, 6.51]},
        pulse_defaults=[
            pulses.SquarePulse(amplitude=0.25, length=16),
            pulses.SquarePulse(amplitude=0.25, length=16),
        ],
    )

    # # Add the transmon components (xy, z and resonator) to the quam
    # for idx in range(num_qubits):
    #     # Create qubit components
    #     transmon = Transmon(
    #         id=idx,
    #         xy=IQChannel(
    #             opx_output_I=f"#/wiring/qubits/{idx}/port_I",
    #             opx_output_Q=f"#/wiring/qubits/{idx}/port_Q",
    #             frequency_converter_up=octave.RF_outputs[2 * (idx + 1)].get_reference(),
    #             intermediate_frequency=100 * u.MHz,
    #         ),
    #         z=FluxLine(opx_output=f"#/wiring/qubits/{idx}/port_Z"),
    #         resonator=ReadoutResonator(
    #             opx_output_I="#/wiring/resonator/opx_output_I",
    #             opx_output_Q="#/wiring/resonator/opx_output_Q",
    #             opx_input_I="#/wiring/resonator/opx_input_I",
    #             opx_input_Q="#/wiring/resonator/opx_input_Q",
    #             opx_input_offset_I=0.0,
    #             opx_input_offset_Q=0.0,
    #             frequency_converter_up=octave.RF_outputs[1].get_reference(),
    #             frequency_converter_down=octave.RF_inputs[1].get_reference(),
    #             intermediate_frequency=50 * u.MHz,
    #             depletion_time=1 * u.us,
    #         ),
    #     )
    #     # Add the transmon pulses to the quam
    #     transmon.xy.operations["saturation"] = pulses.SquarePulse(amplitude=0.25, length=10 * u.us, axis_angle=0)
    #     transmon.z.operations["const"] = pulses.SquarePulse(amplitude=0.1, length=100)
    #     transmon.resonator.operations["readout"] = ConstantReadoutPulse(
    #         length=1 * u.us, amplitude=0.00123, threshold=0.0
    #     )
    #     quam.qubits[transmon.name] = transmon
    #     quam.active_qubit_names.append(transmon.name)
    #     # Set the Octave frequency and channels TODO: be careful to set the right upconverters!!
    #     octave.RF_outputs[2 * (idx + 1)].channel = transmon.xy.get_reference()
    #     octave.RF_outputs[2 * (idx + 1)].LO_frequency = 7 * u.GHz  # Remember to set the LO frequency
    #
    #     octave.RF_outputs[1].channel = transmon.resonator.get_reference()
    #     octave.RF_inputs[1].channel = transmon.resonator.get_reference()
    #     octave.RF_outputs[1].LO_frequency = 4 * u.GHz
    #     octave.RF_inputs[1].LO_frequency = 4 * u.GHz
    return quam


if __name__ == "__main__":
    folder = Path("")
    folder.mkdir(exist_ok=True)

    quam = create_quam_quantumdots_referenced(gates_list=["P1", "P2", "sensor"])
    quam.save(folder / "quam", content_mapping={"wiring.json": {"wiring", "network"}})

    qua_file = folder / "qua_config.json"
    qua_config = quam.generate_config()
    json.dump(qua_config, qua_file.open("w"), indent=4)

    quam_loaded = QuAM.load(folder / "quam")
    qua_file_loaded = folder / "qua_config2.json"
    qua_config_loaded = quam_loaded.generate_config()
    # json.dump(qua_config_loaded, qua_file.open("w"), indent=4)
