"""
hello_octave.py: template for basic usage of octave
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from set_octave import *
from configuration import *
from qm import SimulationConfig
import time

############################
# Set octave configuration #
############################
octave_config = octave_configuration(octave_ip=octave_ip, octave_port=octave_port)

###################################
# Open Communication with the QOP #
###################################
qmm = QuantumMachinesManager(host=opx_ip, port=opx_port, octave=octave_config)

###################
# The QUA program #
###################
with program() as hello_octave:
    with infinite_loop_():
        play('cw', 'qe1')

###################
# Octave settings #
###################
octave_settings(qmm=qmm, qm=qmm.open_qm(config), prog=hello_octave, octave_config=octave_config)

simulate = False
if simulate:
    simulation_config = SimulationConfig(duration=400)  # in clock cycles
    job_sim = qmm.simulate(config, hello_octave, simulation_config)
    # Simulate blocks python until the simulation is done
    job_sim.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(hello_octave)
    # Execute does not block python! As this is an infinite loop, the job would run forever. In this case, we've put a 10
    # seconds sleep and then halted the job.
    # time.sleep(10)
    # job.halt()

