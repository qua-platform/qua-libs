"""
A simple sandbox to showcase different QUA functionalities during the installation.
"""

from qm.qua import *
from qm import QuantumMachinesManager, SimulationConfig
from configuration_lf_fem_and_octave import *

###################
#   QUA Program   #
###################

with program() as PROGRAM:

    play("cw", "rr1")
    play("cw", "q2_xy")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################

simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=1_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done

    job = qmm.simulate(config, PROGRAM, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()

else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)

    # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
    job = qm.execute(PROGRAM)
