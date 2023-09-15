"""
A simple sandbox to showcase different QUA functionalities during the installation.
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *

#######################################################
# Get the config from the machine in configuration.py #
#######################################################
machine = QuAM("quam_bootstrap_state.json")
config = build_config(machine)

# Get the QuAM components used in this experiment
qb1 = machine.qubits[active_qubits[0]]
qb2 = machine.qubits[active_qubits[1]]
rr1 = machine.resonators[active_qubits[0]]
rr2 = machine.resonators[active_qubits[1]]

###################
# The QUA program #
###################
with program() as hello_qua:
    set_dc_offset(qb1.name + "_z", "single", 0.153)
    play("cw", qb1.name + "_xy")
    play("cw", qb2.name + "_xy")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name, octave=octave_config)

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
else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
    job = qm.execute(hello_qua)
