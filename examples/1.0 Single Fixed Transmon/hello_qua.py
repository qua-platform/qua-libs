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
        with for_(a, 0, a < 1.1, a+0.05):
            play('pi'*amp(a), 'qubit')
        wait(25, 'qubit')


qmm = QuantumMachinesManager(host=qop_ip, port=qop_port)
qm = qmm.open_qm(config)
simulation_duration = 400  # clock cycle units - 4ns
job_sim = qm.simulate(hello_QUA, SimulationConfig(simulation_duration))
job_sim.get_simulated_samples().con1.plot()

job = qm.execute(hello_QUA)
time.sleep(10)
job.halt()

