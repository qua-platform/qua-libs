"""
active_reset.py: Single qubit active reset
Author: Niv Drucker - Quantum Machines
Created: 7/11/2020
Created on QUA version: 0.5.138
"""

from configuration import config
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
import matplotlib.pyplot as plt


N = 1000
th = -0.1

with program() as active_reset:

    n = declare(int)

    I = declare(fixed)
    I_stream = declare_stream()

    Q = declare(fixed)
    Q_stream = declare_stream()

    measure("readout", "rr", None, demod.full("integW1", I))

    with for_(n, 0, n < 1000, n + 1):

        play("pi", "qubit", condition=I > th)
        align("qubit", "rr")
        measure("readout", "rr", None, demod.full("integW1", I), demod.full("integW2", Q))

        save(I, I_stream)
        save(Q, Q_stream)

    with stream_processing():

        I_stream.save_all("I")
        Q_stream.save_all("Q")


qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)
job = qm.simulate(active_reset, SimulationConfig(2000))
samps = job.get_simulated_samples()
I_q1 = samps.con1.analog.get("1")
Q_q1 = samps.con1.analog.get("2")
I_rr1 = samps.con1.analog.get("3")
Q_rr1 = samps.con1.analog.get("4")
plt.plot(I_q1)
plt.plot(Q_q1)
plt.plot(I_rr1)
plt.plot(Q_rr1)
