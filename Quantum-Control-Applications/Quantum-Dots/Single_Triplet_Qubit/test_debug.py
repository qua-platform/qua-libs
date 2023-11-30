"""
A simple sandbox to showcase different QUA functionalities during the installation.
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt

###################
# The QUA program #
###################
level_empty = (-0.2, +0.2)
level_init = (0.1, 0.05)
level_manip = (-0.05, 0.1)
level_readout = (0.2, -0.1)

duration_empty = 1000 // 4
duration_init_ramp = 1000 // 4
duration_init = 2000 // 4
duration_manip = 8000 // 4
duration_readout = 500 // 4

with program() as hello_qua:
    j = declare(int)
    k = declare(int)

    align()
    with strict_timing_():
        # Manipulation EDSR
        with for_(j, 0, j < 3, j + 1):
            wait(1000, "qubit_left")
            play("cw", "qubit_left", duration=200)  # replaced by "gauss" on the 2nd figure
            wait(500, "qubit_left")
            play("cw", "qubit_left", duration=200)  # replaced by "gauss" on the 2nd figure
            wait(500, "qubit_left")

        # Manipulation Singlet-Triplet
        with for_(k, 0, k < 3, k + 1):
            wait(1000, "P1")
            play("bias", "P1", duration=200)
            wait(500, "P1")
            play("bias", "P1", duration=200)
            wait(500, "P1")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################

simulate = True

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, hello_qua, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()
    plt.legend("")
else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
    job = qm.execute(hello_qua)