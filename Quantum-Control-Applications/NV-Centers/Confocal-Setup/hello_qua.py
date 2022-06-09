import time

from qm import SimulationConfig
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np


with program() as hello_QUA:
    a = declare(fixed)
    with infinite_loop_():
        with for_(a, 0, a < 1.1, a + 0.05):
            play("pi" * amp(a), "NV")
        wait(25, "NV")


qmm = QuantumMachinesManager(qop_ip)
qm = qmm.open_qm(config)
simulation_duration = 800  # clock cycle units - 4ns
job_sim = qm.simulate(hello_QUA, SimulationConfig(simulation_duration))
# Simulate blocks python until the simulation is done
job_sim.get_simulated_samples().con1.plot()

job = qm.execute(hello_QUA)
# Execute does not block python! As this is an infinite loop, the job would run forever. In this case, we've put a 10
# seconds sleep and then halted the job.
time.sleep(10)
job.halt()
