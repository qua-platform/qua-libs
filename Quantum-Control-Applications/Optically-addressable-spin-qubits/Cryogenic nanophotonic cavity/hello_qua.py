"""
hello_qua.py: template for basic qua program demonstration
"""
import time
from qm import SimulationConfig, LoopbackInterface
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
from qm.simulate.credentials import create_credentials

qmm = QuantumMachinesManager(
    host="faraon-eea6492d.quantum-machines.co",
    port=443,
    credentials=create_credentials(),
)

###################
# The QUA program #
###################
with program() as hello_QUA:

    # amplitude modulation
    a = declare(fixed)
    with infinite_loop_():
        with for_(a, 0, a < 1.1, a + 0.05):
            play("pi" * amp(a), "Yb")
        wait(25, "Yb")

    # frequency modulation
    # with infinite_loop_():
    #     update_frequency('Yb', int(10e6))
    #     play('pi', 'Yb')
    #     wait(4, 'Yb')
    #     update_frequency('Yb', int(35e6))
    #     play('pi', 'Yb')
    #     wait(4, 'Yb')
    #     update_frequency('Yb', int(75e6))
    #     play('pi', 'Yb')
    #     wait(60, 'Yb')

    # phase modulation
    # with infinite_loop_():
    #     play('pi', 'Yb')
    #     frame_rotation_2pi(0.15, 'Yb')
    #     play('pi', 'Yb')
    #     frame_rotation_2pi(0.35, 'Yb')
    #     play('pi', 'Yb')
    #     frame_rotation_2pi(0.75, 'Yb')
    #     wait(25, 'Yb')

    # duration modulation
    # with infinite_loop_():
    #     play('pi', 'Yb', duration=16)
    #     wait(16, 'Yb')
    #     play('pi', 'Yb', duration=50)
    #     wait(16, 'Yb')
    #     play('pi', 'Yb', duration=150)
    #     wait(16, 'Yb')


#####################################
#  Open Communication with the QOP  #
#####################################


simulation_config = SimulationConfig(
    duration=700, simulation_interface=LoopbackInterface([("con1", 3, "con1", 1)])  # in clock cycle
)
job_sim = qmm.simulate(config, hello_QUA, simulation_config)
# Simulate blocks python until the simulation is done
job_sim.get_simulated_samples().con1.plot()
