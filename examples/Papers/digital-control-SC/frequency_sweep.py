"""
frequency_sweep.py: Rabi-chevron experiment
Author: Arthur Strauss - Quantum Machines
Created: 25/01/2021
Created on QUA version: 0.8.439
"""

# Importing the necessary from qm
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm.qua import math
from qm import SimulationConfig
import numpy as np
import matplotlib.pyplot as plt
import time
from configuration import *


N_max = 3
t_max = int(500 / 4)
dt = 1
N_t = t_max // dt
f_max = 1.66e9
f_min = 1.64e9
df = 1.e8
N_f = int((f_max-f_min)/df)

qmManager = QuantumMachinesManager()
QM = qmManager.open_qm(config)  # Generate a Quantum Machine based on the configuration described above

with program() as bias_current_sweeping:  #
    I = declare(fixed)  # QUA variables declaration
    Q = declare(fixed)
    state = declare(bool)
    th = declare(fixed, value=2.0)  # Threshold assumed to have been calibrated by state discrimination exp
    t = declare(int)
    f = declare(fixed)
    Nrep = declare(int)  # Variable looping for repetitions of experiments
    tau = declare(fixed, value=1.0)

    f_stream = declare_stream()
    state_stream = declare_stream()
    t_stream = declare_stream()
    I_stream = declare_stream()
    Q_stream = declare_stream()

    with for_(Nrep, 0, Nrep < N_max, Nrep + 1):
        with for_(f, f_min, f < f_max, f + df):  # Sweep from 0 to V_c the bias voltage
            with for_(t, 0.00, t < t_max, t + dt):
                update_frequency("SFQ_trigger", f)
                play("playOp", "SFQ_bias")
                play("const_pulse", "SFQ_trigger", duration=t)
                align("qubit_control", "RR", "SFQ_trigger")
                wait(tau, "qubit_control")
                measure("meas_pulse", "RR", "samples", ("integW1", I), ("integW2", Q))
                assign(state, I > th)
                save(I, I_stream)
                save(Q, Q_stream)
                with while_(I > th):  # Active reset
                    play("pi_pulse", 'SFQ_trigger', condition=I > th)
                    measure("meas_pulse", "RR", "samples", ("integW1", I), ("integW2", Q))

                save(state, state_stream)
                save(t, t_stream)

            save(f, f_stream)
    with stream_processing():
        I_stream.save_all("I")
        Q_stream.save_all("Q")
        f_stream.buffer(N_f).save("f")
        t_stream.buffer(N_t).save('t')
        state_stream.boolean_to_int().buffer(N_f, N_t).average().save("state")

job = qmManager.simulate(config, bias_current_sweeping,
                         SimulationConfig(int(50000)))

time.sleep(1.0)

# Retrieving results of the experiments
results = job.result_handles
f = results.f.fetch_all()
t = results.t.fetch_all()
state = results.state.fetch_all()
I = results.I.fetch_all()["value"]
Q = results.Q.fetch_all()["value"]
fig = plt.figure()

# Plot the surface.
plt.pcolormesh(f, t, state, shading="nearest")
plt.xlabel("ω_d/2π [GHz]")
plt.ylabel("Pulse duration [ns]")
plt.colorbar()
plt.title("Rabi chevron experiment")
