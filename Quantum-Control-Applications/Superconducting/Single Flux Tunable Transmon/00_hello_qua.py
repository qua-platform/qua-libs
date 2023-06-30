from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from configuration import *


###################
# The QUA program #
###################
with program() as hello_qua:
    play("cw", "qubit")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip, qop_port, octave=octave_config)

simulation = True
if simulation:
    simulation_config = SimulationConfig(duration=8000)
    job = qmm.simulate(config, hello_qua, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(hello_qua)
