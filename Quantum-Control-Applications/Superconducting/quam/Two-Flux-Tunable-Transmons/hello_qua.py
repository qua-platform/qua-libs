"""
A simple sandbox to showcase different QUA functionalities during the installation.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from components import QuAM


###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
quam = QuAM.load("quam")
config = quam.generate_config()
octave_config = quam.octave.get_octave_config()
# Open Communication with the QOP
qmm = QuantumMachinesManager(host="172.16.33.101", cluster_name="Cluster_81", octave=octave_config)

# Get the relevant QuAM components


###################
# The QUA program #
###################
with program() as hello_qua:
    a = declare(fixed)
    adc_st = declare_stream(adc_trace=True)
    with for_(a, 0, a < 1.1, a + 0.05):
        play("x180" * amp(a), quam.qubits[0].xy.name)
        wait(25, quam.qubits[0].xy.name)
        quam.qubits[0].resonator.measure("readout", stream=adc_st)

    with stream_processing():
        adc_st.input1().save("adc")


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