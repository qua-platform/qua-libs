"""
orthogonal-qubit-control.py:
Author: Gal Winer  - Quantum Machines
Created: 22/02/2021
Created on QUA version: 0.8.439
"""

# Importing the necessary from qm
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import numpy as np
import matplotlib.pyplot as plt
import time
from configuration import *


N_max = 3
t_max = int(660 / 4)
dt = 1

theta_max_2pi = 1
dtheta = 0.1
N_t = t_max // dt
f_max = 200e6
f_min = 100e6
df = 1e6
N_f = int((f_max - f_min) / df)
t_min = 16
t_poison_min = 16
t_poison_max = 500
dt_poison = 20

qmManager = QuantumMachinesManager()
QM = qmManager.open_qm(config)  # Generate a Quantum Machine based on the configuration described above

with program() as bias_current_sweeping:  #
    I = declare(fixed)  # QUA variables declaration
    Q = declare(fixed)
    state = declare(bool)
    th = declare(fixed, value=2.0)  # Threshold assumed to have been calibrated by state discrimination exp
    t = declare(int)
    t_poison = declare(int)
    f = declare(int)
    Nrep = declare(int)  # Variable looping for repetitions of experiments
    theta = declare(fixed, value=1)
    f_stream = declare_stream()
    state_stream = declare_stream()
    t_stream = declare_stream()
    I_stream = declare_stream()
    Q_stream = declare_stream()
    update_frequency("SFQ_trigger", int(200e6))  # assumed this is far off-resonant

    with for_(Nrep, 0, Nrep < N_max, Nrep + 1):
        with for_(t_poison, t_poison_min, t_poison < t_poison_max, t_poison + dt_poison):
            with for_(t, t_min, t < t_max, t + dt):
                play("pi_pulse", "SFQ_bias", duration=t_poison)
                play("pi_pulse", "SFQ_trigger", duration=t_poison)  # poison stage
                align("SFQ_trigger", "qubit_control")
                play("pi_pulse", "qubit_control")
                align("RR", "qubit_control")
                wait(t, "RR")
                measure("meas_pulse", "RR", "samples", ("integW1", I), ("integW2", Q))
                assign(state, I > th)
                save(I, I_stream)
                save(Q, Q_stream)

                save(state, state_stream)

                with while_(state == True):  # active reset
                    play("pi_pulse", "qubit_control")
                    measure("meas_pulse", "RR", "samples", demod.full("integW1", I))
                    assign(state, I > th)

                save(t, t_stream)

                save(f, f_stream)
    with stream_processing():
        I_stream.save_all("I")
        Q_stream.save_all("Q")
        f_stream.buffer(N_f).save("f")
        t_stream.buffer(N_t).save("t")
        state_stream.boolean_to_int().buffer(N_f, N_t).average().save("state")

job = qmManager.simulate(config, bias_current_sweeping, SimulationConfig(int(1000)))

samples = job.get_simulated_samples()
samples.con1.plot()
