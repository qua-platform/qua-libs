# %% {Imports}
from typing import Optional, List

from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit

from qualibrate import QualibrationNode, NodeParameters
from quam_config import QuAM
from quam_experiments.parameters.qubits_experiment import get_qubits

description = """
    A simple program to calibrate Octave mixers for all qubits and resonators
"""


class Parameters(NodeParameters):
    qubits: Optional[List[str]] = None
    calibrate_resonator: bool = True
    calibrate_drive: bool = True
    timeout: int = 100


node = QualibrationNode[Parameters, QuAM](
    name="01a_mixer_calibration", description=description, parameters=Parameters()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, QuAM]):
    # You can get type hinting in your IDE by typing node.parameters.
    pass


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
node.machine = QuAM.load()

# Get the active qubits from the node and organize them by batches
node.namespace["qubits"] = qubits = get_qubits(node)

# Generate the OPX and Octave configurations
config = node.machine.generate_config()
# Open Communication with the QOP
qmm = node.machine.connect()


# %% {Execute calibration}
with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
    for qubit in qubits:
        qubit.calibrate_octave(
            qm,
            calibrate_drive=node.parameters.calibrate_drive,
            calibrate_resonator=node.parameters.calibrate_resonator,
        )
