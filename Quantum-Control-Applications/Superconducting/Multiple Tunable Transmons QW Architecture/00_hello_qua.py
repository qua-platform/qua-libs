# check if config is correct by trying to open quantum machine
# add optionally your own simulator output for pulses

from quam import QuAM
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.simulate import SimulationConfig
from qm.qua import *
from scipy.signal.windows import dpss
import matplotlib.pyplot as plt


machine = QuAM("quam_bootstrap_state.json")

# machine.qubits[0].sequence_states.arbitrary.append({"name": "slepian", "waveform": (dpss(48,5)*0.5)[:24].tolist()})
# machine.readout_lines[0].length = 1e-6
# machine.save("quam_bootstrap_state.json")
qubit_list = [0, 1, 2, 3, 4, 5]
config = machine.build_config(
    digital_out=[1, 9], qubits=qubit_list, injector_list=[0, 1], shape="drag_cosine"
)

qmm = QuantumMachinesManager(machine.network.qop_ip)


def play_pi():
    update_frequency("qubit_0", 0)
    play("x180", "qubit_0", duration=100)


with program() as hello_qua:

    a = declare(fixed)

    for q in qubit_list:
        with for_(a, 0.1, a < 1.0, a+0.1):
            play('x180'*amp(a), machine.qubits[q].name)
            align()
            measure('readout', machine.readout_resonators[q].name, None)

        align()

job = qmm.simulate(config, hello_qua, SimulationConfig(1500))
job.get_simulated_samples().con1.plot()
plt.show()
machine.save("latest_quam.json")

