# check if config is correct by trying to open quantum machine
# add optionally your own simulator output for pulses

from quam import QuAM
from rich import print
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.simulate.credentials import create_credentials
from qm.simulate import SimulationConfig
from qm.qua import *
from pprint import pprint
import matplotlib.pyplot as plt


machine = QuAM("quam_bootstrap_state.json")

print(machine.qubits[0].f_01)
machine.qubits[0].f_01 = 4.510e9
print(machine.qubits[0].f_01)
machine.get_sequence_state(0, "Jump").amplitude = 0.25
# Mixer calibration
machine.readout_resonators[0].f_res = 6.9e9
machine.readout_lines[0].I_up.offset = 0.0147
machine.readout_lines[0].Q_up.offset = 0.0025
machine.readout_resonators[0].wiring.correction_matrix.gain = 0.1
machine.readout_resonators[0].wiring.correction_matrix.phase = -0.154

machine.drive_lines[0].I.offset = -0.0541
machine.drive_lines[0].Q.offset = -0.0541
machine.qubits[0].wiring.correction_matrix.gain = 0.0
machine.qubits[0].wiring.correction_matrix.phase = 0.0


print(len(machine.qubits))

# machine.save("quam_bootstrap_state.json")

z = [i for i in range(1, 11)]
qbts = [0, 1]
gate_shape = "drag_cosine"
config = machine.build_config(z, qbts, gate_shape=gate_shape)

# qmm = QuantumMachinesManager()
#
# with program() as hello_qua:
#     update_frequency("q0", 0)
#     update_frequency("q1", 0)
#     play("const", "rr0")
#     wait(25, "q0_flux")
#     play("const", "q0_flux", duration=200)
#     align()
#     play("x180", "q0")
#     play("x90", "q0")
#     play("x-90", "q0")
#     play("x-180", "q0")
#     play("y-180", "q0")
#     play("y-90", "q0")
#     play("y90", "q0")
#     play("y180", "q0")
#
#
#
#
#
#
# job = qmm.simulate(config, hello_qua, SimulationConfig(500))
# job.get_simulated_samples().con1.plot()
# plt.show()
