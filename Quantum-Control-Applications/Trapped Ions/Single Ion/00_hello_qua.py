"""
A simple sandbox to showcase different QUA functionalities during the installation.
"""

from pathlib import Path

from qm import QuantumMachinesManager, SimulationConfig
from qm.qua import *

from configuration import config, qop_ip, cluster_name
from macros import doppler_cool

###################
# The QUA program #
###################
with program() as hello_qua:
    a = declare(fixed)
    with infinite_loop_():
        with for_(a, 0.1, a < 1.0, a + 0.1):
            play("constant" * amp(a), "cooling", duration=100)
            play("constant", "repump", duration=100)
            align()
        doppler_cool()
        play("x180", "qubit")
        wait(1000)

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)

###########################
# Run or Simulate Program #
###########################

simulate = True

if simulate:
    simulation_config = SimulationConfig(duration=10_000)
    job = qmm.simulate(config, hello_qua, simulation_config)
    samples = job.get_simulated_samples()
    samples.con1.plot()
    waveform_report = job.get_simulated_waveform_report()
    waveform_report.create_plot(samples, plot=True, save_path=str(Path(__file__).resolve()))
else:
    qm = qmm.open_qm(config, close_other_machines=True)
    job = qm.execute(hello_qua)
