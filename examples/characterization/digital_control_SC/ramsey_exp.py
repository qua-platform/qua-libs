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
from qm import LoopbackInterface
from qm import SimulationConfig
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit
from configuration import *


N_max = 3
t_max = int(500 / 4)
dt = 1
N_t = t_max // dt
ω_max = 1.658e9
ω_min = 1.648e9
dω = 1.e8
N_ω = int((ω_max-ω_min)/dω)

qmManager = QuantumMachinesManager("3.122.60.129")
QM = qmManager.open_qm(config)  # Generate a Quantum Machine based on the configuration described above

with program() as bias_current_sweeping:  #
    I = declare(fixed)  # QUA variables declaration
    Q = declare(fixed)
    state = declare(bool)
    th = declare(fixed, value=2.0)  # Threshold assumed to have been calibrated by state discrimination exp
    t = declare(int)
    ω = declare(fixed)
    Nrep = declare(int)  # Variable looping for repetitions of experiments
    ω_stream = declare_stream()
    state_stream = declare_stream()
    t_stream = declare_stream()

    with for_(Nrep, 0, Nrep < N_max, Nrep + 1):
        with for_(ω, ω_min, ω < ω_max, ω + dω):  # Sweep from 0 to V_c the bias voltage
            with for_(t, 0.00, t < t_max, t + dt):
                update_frequency("qubit", ω)
                play("playOp", "SFQ_driver")
                play("gauss_pulse", 'qubit', duration=t)
                align("qubit", "RR")
                measure("meas_pulse", "RR", "samples", ("integW1", I), ("integW2", Q))
                play("pi_pulse", 'qubit', condition=I > th)  # Active reset
                assign(state, I > th)
                save(state, state_stream)
                save(t, t_stream)

            save(ω, ω_stream)
    with stream_processing():
        ω_stream.buffer(N_ω).save("ω")
        t_stream.buffer(N_t).save('t')
        state_stream.boolean_to_int().buffer(N_ω, N_t).average().save("state")

job = qmManager.simulate(config, bias_current_sweeping,
                         SimulationConfig(int(50000),
                                          simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)])
                                          )
                         )  # Use LoopbackInterface to simulate the response of the qubit
time.sleep(1.0)

# Retrieving results of the experiments
results = job.result_handles
ω = results.ω.fetch_all()
t = results.t.fetch_all()
state = results.state.fetch_all()

fig = plt.figure()

# Plot the surface.
plt.pcolormesh(ω, t, state, shading="nearest")
plt.xlabel("Bias current I_b [µA]")
plt.ylabel("Pulse duration [ns]")
plt.colorbar()
plt.title("SFQ-based Rabi oscillations as a function of bias current to SFQ driver circuit")
