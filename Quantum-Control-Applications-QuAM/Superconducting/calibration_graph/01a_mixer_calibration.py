# %%

"""
A simple program to calibrate Octave mixers for all qubits and resonators
"""

from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from typing import Optional, List


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = None
    calibrate_resonator: bool = True
    calibrate_drive: bool = True
    timeout: int = 100


node = QualibrationNode(name="01a_Mixer_Calibration", parameters=Parameters())

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
qmm = machine.connect()


# %% {Execute calibration}
with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
    for qubit in qubits:
        qubit.calibrate_octave(
            qm, calibrate_drive=node.parameters.calibrate_drive, calibrate_resonator=node.parameters.calibrate_resonator
        )

