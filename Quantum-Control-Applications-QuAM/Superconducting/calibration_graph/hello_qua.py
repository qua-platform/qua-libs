# %%
from qm.qua import *
from qualang_tools.units import unit
from quam_libs.components import QuAM
import matplotlib.pyplot as plt
import numpy as np
from qualang_tools.units import unit
from quam_libs.components import QuAM

###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
# path = "../configuration/quam_state"
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
# octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()
calibrate_octave = True

qubits = machine.active_qubits

with program() as prog:
    qubits[0].xy.update_frequency(-100e6)
    qubits[1].xy.update_frequency(-100e6)

    qubits[0].xy.play("x180")
    qubits[1].xy.play('x180')
    align()
    qubits[0].resonator.play("readout")


qm = qmm.open_qm(config, close_other_machines=True)
if calibrate_octave:
    machine.calibrate_octave_ports(qm)
job = qm.execute(prog)
plt.show()


# %%
