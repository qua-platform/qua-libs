from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from quam import QuAM
from configuration import build_config

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("quam_bootstrap_state.json", flat_data=False)
config = build_config(machine)

###################
# The QUA program #
###################
with program() as hello_qua:
    set_dc_offset("q0_z", "single", 0.153)
    play("cw", "q0_xy")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.qop_port)

simulation = True
if simulation:
    simulation_config = SimulationConfig(duration=8000)
    job = qmm.simulate(config, hello_qua, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(hello_qua)
