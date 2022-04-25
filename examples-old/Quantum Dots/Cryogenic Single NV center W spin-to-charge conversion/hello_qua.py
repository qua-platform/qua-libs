from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from configuration import *
from qm import SimulationConfig
from qm import LoopbackInterface

################################
# Open quantum machine manager #
################################

qmm = QuantumMachinesManager()

########################
# Open quantum machine #
########################

qm = qmm.open_qm(config)

###################
# The QUA program #
###################

simulate = True

with program() as hello_qua:

    play("charge_init", "laser_705nm")
    play("spin_init", "laser_E12")
    align("qubit", "laser_E12")
    wait(50, "qubit")
    play("pi", "qubit")
    align("qubit", "laser_EX1151")
    play("SCC", "laser_EX1151")
    wait(10, "laser_EX12")
    play("charge_readout", "laser_EX12")


#######################
# Simulate or execute #
#######################

if simulate:

    simulate_config = SimulationConfig(
        duration=int(1000),
        simulation_interface=LoopbackInterface(([("con1", 1, "con1", 1)])),
    )  # simulation properties
    job = qmm.simulate(config, hello_qua, simulate_config)  # do simulation
    job.get_simulated_samples().con1.plot()  # visualize played pulses

else:
    job = qm.execute(hello_qua)
