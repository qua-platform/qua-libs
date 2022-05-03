"""
time_rabi.py: Rabi experiment for metamaterial resonator
Author: Arthur Strauss - Quantum Machines
Created: 25/01/2021
Created on QUA version: 0.8.439
"""

# Importing the necessary from qm
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
import time
from configuration import *
from mock_LO_source import mock_LO_source

N_max = 3
t_max = int(1000 / 4)
dt = 1
N_t = t_max // dt
f_max = 60e6
f_min = 20e6
df = 1e6
N_f = int((f_max - f_min) / df)

qmManager = QuantumMachinesManager()
QM = qmManager.open_qm(config)  # Generate a Quantum Machine based on the configuration described above
LO_source = mock_LO_source(9.2e9)
with program() as bias_current_sweeping:  #
    I = declare(fixed)  # QUA variables declaration
    Q = declare(fixed)
    t = declare(int)
    f = declare(fixed)
    Nrep = declare(int)  # Variable looping for repetitions of experiments
    tau = declare(int, value=16)
    f_stream = declare_stream()
    t_stream = declare_stream()
    I_stream = declare_stream()
    Q_stream = declare_stream()

    with for_(Nrep, 0, Nrep < N_max, Nrep + 1):
        with for_(f, f_min, f < f_max, f + df):  # Sweep from 0 to V_c the bias voltage
            with for_(t, 0.00, t < t_max, t + dt):
                update_frequency("RR", f)
                play("measure", "RR", duration=t)
                align("qubit", "RR")
                measure("measure", "RR", "samples", ("integW1", I), ("integW2", Q))
                save(I, I_stream)
                save(Q, Q_stream)
                save(t, t_stream)
                wait(tau, "qubit")

            save(f, f_stream)
    with stream_processing():
        I_stream.buffer(N_f, N_t).average().save("I")
        Q_stream.buffer(N_f, N_t).average().save("Q")
        f_stream.buffer(N_f).save("f")
        t_stream.buffer(N_t).save("t")


job = qmManager.simulate(config, bias_current_sweeping, SimulationConfig(int(50000)))

time.sleep(1.0)

# Retrieving results of the experiments
results = job.result_handles
f = results.f.fetch_all()
t = results.t.fetch_all()
I = results.I.fetch_all()["value"]
Q = results.Q.fetch_all()["value"]
fig = plt.figure()

# Plot the surface.
plt.pcolormesh(f, t, I, shading="nearest")
plt.xlabel("f_spec [GHz]")
plt.ylabel("Pulse duration [ns]")
plt.colorbar()
plt.title("Rabi chevron experiment")
