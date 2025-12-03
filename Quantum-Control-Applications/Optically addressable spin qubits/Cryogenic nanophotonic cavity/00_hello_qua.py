"""
00_hello_qua.py: template for basic qua program demonstration
"""

import time
from qm import SimulationConfig, LoopbackInterface
from qm.qua import *
from qm import QuantumMachinesManager
from configuration import *

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
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name)

###########################
# Run or Simulate Program #
###########################

simulate = True

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=1_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, hello_QUA, simulation_config)
    # Get the simulated samples
    samples = job.get_simulated_samples()
    # Plot the simulated samples
    samples.con1.plot()
    # Get the waveform report object
    waveform_report = job.get_simulated_waveform_report()
    # Cast the waveform report to a python dictionary
    waveform_dict = waveform_report.to_dict()
    # Visualize and save the waveform report
    waveform_report.create_plot(samples, plot=True, save_path=str(Path(__file__).resolve()))
else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)

    # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
    job = qm.execute(hello_QUA)
