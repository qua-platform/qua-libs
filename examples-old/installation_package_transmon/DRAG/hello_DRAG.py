from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration_DRAG import *
import matplotlib.pyplot as plt

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

with program() as hello_qua:

    play("X_d", "qubit")
    wait(10)
    play("X/2_d", "qubit")
    wait(10)
    play("-X_d", "qubit")
    wait(10)
    play("-X/2_d", "qubit")
    wait(10)
    play("Y_d", "qubit")
    wait(10)
    play("Y/2_d", "qubit")
    wait(10)
    play("-Y_d", "qubit")
    wait(10)
    play("-Y/2_d", "qubit")
    wait(40)
    play("X_c", "qubit")
    wait(10)
    play("X/2_c", "qubit")
    wait(10)
    play("-X_c", "qubit")
    wait(10)
    play("-X/2_c", "qubit")
    wait(10)
    play("Y_c", "qubit")
    wait(10)
    play("Y/2_c", "qubit")
    wait(10)
    play("-Y_c", "qubit")
    wait(10)
    play("-Y/2_c", "qubit")
    wait(10)

#######################
# Simulate or execute #
#######################

simulate = True

if simulate:
    # simulation properties
    simulate_config = SimulationConfig(duration=700, simulation_interface=LoopbackInterface(([("con1", 1, "con1", 1)])))
    job = qmm.simulate(config, hello_qua, simulate_config)  # do simulation with qmm
    job.get_simulated_samples().con1.plot()  # visualize played pulses

else:
    job = qm.execute(hello_qua)  # execute QUA program
