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
config = machine.build_config(digital_out=[], qubits=[0, 1], gate_shape="drag_cosine")


# qmm = QuantumMachinesManager()
qmm = QuantumMachinesManager(
    host="theo-4c195fa0.dev.quantum-machines.co",
    port=443,
    credentials=create_credentials())

with program() as hello_qua:
    play("x180", "qubit_0")



job = qmm.simulate(config, hello_qua, SimulationConfig(500))
job.get_simulated_samples().con1.plot()
plt.show()
