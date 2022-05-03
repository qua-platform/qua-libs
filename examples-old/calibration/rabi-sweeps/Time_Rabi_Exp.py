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

t_min = 10
t_max = 100  # Maximum pulse duration (in clock cycles, 1 clock cycle =4 ns)
dt = 1  # timestep
N_t = len(np.arange(t_min, t_max + dt, dt))  # Number of timesteps
N_max = 3

qmManager = QuantumMachinesManager()  # Reach OPX's IP address
my_qm = qmManager.open_qm(config)  # Generate a Quantum Machine based on the configuration described above

with program() as timeRabiProg:  # Time Rabi QUA program
    I = declare(fixed)  # QUA variables declaration
    Q = declare(fixed)
    t = declare(int)  # Sweeping parameter over the set of durations
    Nrep = declare(int)  # Number of repetitions of the experiment
    I_stream = declare_stream()  # Declare streams to store I and Q components
    Q_stream = declare_stream()
    t_stream = declare_stream()
    with for_(Nrep, 0, Nrep < N_max, Nrep + 1):  # Do a 100 times the experiment to obtain statistics
        with for_(t, t_min, t <= t_max, t + dt):  # Sweep from 0 to 100 *4 ns the pulse duration
            play("gauss_pulse", "qubit", duration=t)
            align("qubit", "RR")
            measure("meas_pulse", "RR", None, ("integW1", I), ("integW2", Q))
            save(I, I_stream)
            save(Q, Q_stream)
            save(t, t_stream)
    with stream_processing():
        I_stream.buffer(N_t).average().save("I")
        Q_stream.buffer(N_t).average().save("Q")
        t_stream.buffer(N_t).save("t")

my_job = my_qm.simulate(
    timeRabiProg,
    SimulationConfig(int(200000), simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)])),
)  ##Use Loopback interface for simulation of the output of the resonator readout
time.sleep(1.0)
my_timeRabi_results = my_job.result_handles
I1 = my_timeRabi_results.I.fetch_all()
Q1 = my_timeRabi_results.Q.fetch_all()
t1 = my_timeRabi_results.t.fetch_all()

samples = my_job.get_simulated_samples()

# Processing the data
def fit_function(x_values, y_values, function, init_params):
    fitparams, conv = curve_fit(function, x_values, y_values, init_params)
    y_fit = function(x_values, *fitparams)

    return fitparams, y_fit


# Build a fitting tool for finding the right amplitude
# #(initial parameters to be adapted according to qubit and RR frequencies)
I_params, I_fit = fit_function(
    t1,
    I1,
    lambda x, A, drive_period, phi: (A * np.cos(2 * np.pi * x / drive_period - phi)),
    [0.004, 200, 1],
)
Q_params, Q_fit = fit_function(
    t1,
    Q1,
    lambda x, A, drive_period, phi: (A * np.cos(2 * np.pi * x / drive_period - phi)),
    [0.004, 200, 0.0],
)

plt.figure()
plt.plot(t1, I1, marker="x", color="blue", label="I")
plt.plot(t1, Q1, marker="o", color="green", label="Q")
plt.plot(t1, I_fit, color="red", label="Sinusoidal fit")
plt.plot(t1, Q_fit, color="black", label="Sinusoidal fit")
plt.xlabel("Pulse duration [clock cycles]")
plt.ylabel("Measured signal [a.u]")
plt.axvline(I_params[1] / 2, color="red", linestyle="--")
plt.axvline(0, color="red", linestyle="--")
plt.annotate(
    "",
    xy=(0, 0),
    xytext=(I_params[1] / 2, 0),
    arrowprops=dict(arrowstyle="<->", color="red"),
)
plt.annotate("$\pi$", xy=(I_params[1] / 4, 0.0001), color="red")
plt.show()

print("The duration required to perform a X gate is", I_params[1] / 2, "cycles")
