"""
A simple program to play a continuous wave for spectrum analysis
"""
from qm import SimulationConfig
from qm.qua import *
from pathlib import Path
from quam_components import QuAM
import matplotlib.pyplot as plt

###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Instantiate the QuAM class from the state file

# Define a path relative to this script, i.e., ../configuration/quam_state
config_path = Path(__file__).parent.parent / "configuration" / "quam_state"
# Load the machine state
machine = QuAM.load(config_path)

# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()
qm = qmm.open_qm(config)

element = machine.active_qubits[0].resonator
# element = machine.active_qubits[0].xy

###################
# The QUA program #
###################
with program() as continuous_wave:
    i = declare(int)
    # play for 10s
    with for_(i, 0, i < 300_000, i + 1):
        # play("x180_Square", element.name, duration=25_000)
        play("const", element.name, duration=25_000)


###########################
# Run or Simulate Program #
###########################

simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=5_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, continuous_wave, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()
    plt.show()
else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
    job = qm.execute(continuous_wave)
