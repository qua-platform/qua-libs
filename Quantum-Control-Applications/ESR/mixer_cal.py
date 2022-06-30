"""
A script used to calibrate the corrections for mixer imbalances
"""

from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
from qm.simulate.credentials import create_credentials

###################
# The QUA program #
###################

with program() as mixer_cal:

    with infinite_loop_():
        play("const", "ensemble")


################################
# Open quantum machine manager #
################################

qmm = QuantumMachinesManager(host=qop_ip, port=qop_port)

#######################
# Simulate or execute #
#######################

simulate = False

if simulate:
    # simulation properties
    simulate_config = SimulationConfig(
        duration=2000,
        simulation_interface=LoopbackInterface(([("con1", 3, "con1", 1), ("con1", 4, "con1", 2)]), latency=180),
    )
    job = qmm.simulate(config, mixer_cal, simulate_config)  # do simulation with qmm
    job.get_simulated_samples().con1.plot()  # visualize played pulses

else:
    qm = qmm.open_qm(config)

    job = qm.execute(mixer_cal)  # execute QUA program
