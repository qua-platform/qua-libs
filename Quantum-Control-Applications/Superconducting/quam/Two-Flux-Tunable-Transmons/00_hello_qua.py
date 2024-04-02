"""
A simple sandbox to showcase different QUA functionalities during the installation.
"""


from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from quam.examples.superconducting_qubits.components import QuAM


###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
machine = QuAM.load("quam")
config = machine.generate_config()
octave_config = machine.octave.get_octave_config()
qmm = QuantumMachinesManager(host="172.16.33.101", cluster_name="Cluster_81", octave=octave_config)

###################
# The QUA program #
###################
with program() as hello_qua:
    a = declare(fixed)
    with infinite_loop_():
        with for_(a, 0, a < 1.1, a + 0.05):
            play("x180" * amp(a), machine.qubits["q0"].xy.name)
        wait(25, machine.qubits["q0"].xy.name)


###########################
# Run or Simulate Program #
###########################

simulate = True

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=1_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, hello_qua, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()
else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
    job = qm.execute(hello_qua)
