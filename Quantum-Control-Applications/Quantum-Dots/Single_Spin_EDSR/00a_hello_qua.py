"""
A simple sandbox to showcase different QUA functionalities during the installation.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration_with_lf_fem_and_mw_fem import *
import matplotlib.pyplot as plt
from qm import generate_qua_script


###################
# The QUA program #
###################
with program() as hello_qua:
    t = declare(int, value=16)
    a = declare(fixed, value=0.2)
    i = declare(int)

    play("init", "P1")
    play("init", "P2")
    play("manip", "P1", duration=t)
    play("manip", "P2", duration=t)
    play("readout", "P1")
    play("readout", "P2")


#####################################
#  Open Communication with the QOP  #
#####################################
# qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################

simulate = True

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=1_250)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    from test_tools.cloud_simulator import client
    from qm_saas import QoPVersion

    with client.simulator(QoPVersion.v3_2_4) as instance:
        # Use the instance object to simulate QUA programs
        qmm = QuantumMachinesManager(
            host=instance.host,
            port=instance.port,
            connection_headers=instance.default_connection_headers,
        )
        job = qmm.simulate(config, hello_qua, simulation_config)
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
    job = qm.execute(hello_qua)
