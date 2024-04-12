from pathlib import Path
import json

from quam.components import *
from quam.components.channels import IQChannel, InOutSingleChannel, SingleChannel, DigitalOutputChannel
from quam.components.pulses import ConstantReadoutPulse
from simple_components import QuAM, FluxLine
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
    quam.test=SingleChannel(
            id="test",
            opx_output=("con1", 1),
        )
    quam.test.operations["trigger"] = pulses.SquarePulse(amplitude=0.10, length=16)
    trig = SingleChannel(  # TODO: have it without analog output??
        id="trig1",
        digital_outputs={
            "marker1": DigitalOutputChannel(opx_output="#/wiring/qdac_triggers/1/digital_marker", delay=0, buffer=0)
        },
        opx_output = ("con1", 1)
    )
    trig.operations["trigger"] = pulses.SquarePulse(amplitude=0.0, length=16, digital_marker="ON")
    # trig.operations["trigger"] = pulses.SquarePulse(amplitude=0.0, length=16, digital_marker=[(1,0)])  # TODO: doesn't work
    quam.qdac_trigger["trig1"] = trig

    quam.qdac_trigger["trig2"] = SingleChannel(  # TODO: type is components.SingleChannel instead of SingleChannel !!
        id="trig2",
        digital_outputs={
            "marker1": DigitalOutputChannel(opx_output="#/wiring/qdac_triggers/1/digital_marker", delay=0, buffer=0)
        },
        opx_output = ("con1", 1)
    )
    quam.qdac_trigger["trig2"].operations["trigger"] = pulses.SquarePulse(amplitude=0.0, length=16, digital_marker="ON")

    return quam, octave_config


if __name__ == "__main__":
    folder = Path("")
    folder.mkdir(exist_ok=True)

    quam, octave_config = create_quam_quantumdots_referenced(gates_list=["P1", "P2", "sensor"])
    quam.save(folder / "quam", content_mapping={"wiring.json": {"wiring", "network"}})

    qua_file = folder / "qua_config.json"
    qua_config = quam.generate_config()
    json.dump(qua_config, qua_file.open("w"), indent=4)

    quam_loaded = QuAM.load(folder / "quam")
    qua_file_loaded = folder / "qua_config2.json"
    qua_config_loaded = quam_loaded.generate_config()
    # json.dump(qua_config_loaded, qua_file.open("w"), indent=4)
