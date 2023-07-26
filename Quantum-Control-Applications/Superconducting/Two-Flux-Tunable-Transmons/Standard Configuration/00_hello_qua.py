from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *


###################
# The QUA program #
###################
with program() as hello_qua:
    set_dc_offset("q1_z", "single", 0.153)
    play("cw", "q1_xy")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, octave=octave_config)

simulation = True
if simulation:
    simulation_config = SimulationConfig(duration=8000)
    job = qmm.simulate(config, hello_qua, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(hello_qua)
