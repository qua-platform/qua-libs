"""
Power_Rabi_Exp.py: Power Rabi experiment simulation
Author: Arthur Strauss - Quantum Machines
Created: 13/11/2020
Created on QUA version: 0.5.138
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

a_max = 0.700  # Maximum amplitude
da = 0.001  # amplitude sweeping step
N_a = len(np.arange(0.0, a_max, da))  # Number of steps
N_max = 3


qmManager = QuantumMachinesManager()  # Reach OPX's IP address
my_qm = qmManager.open_qm(config)  # Generate a Quantum Machine based on the configuration described above

with program() as powerRabiProg:  # Power Rabi QUA program
    I = declare(fixed)  # QUA variables declaration
    Q = declare(fixed)
    a = declare(fixed)  # Sweeping parameter over the set of amplitudes
    Nrep = declare(int)  # Variable looping for repetitions of experiments
    I_stream = declare_stream()  # Declare streams to store I and Q components
    Q_stream = declare_stream()
    a_stream = declare_stream()

    with for_(Nrep, 0, Nrep < N_max, Nrep + 1):  # Do 10 times the experiment
        with for_(a, 0.00, a < a_max - da / 2, a + da):  # Sweep from 0 to 0.7 V the amplitude
            play("gauss_pulse" * amp(a), "qubit")  # Modulate the Gaussian pulse with the varying amplitude a
            align("qubit", "RR")
            measure("meas_pulse", "RR", None, ("integW1", I), ("integW2", Q))
            save(I, I_stream)  # Save the results
            save(Q, Q_stream)
            save(a, a_stream)
    with stream_processing():
        a_stream.buffer(N_a).save("a")
        I_stream.buffer(N_a).average().save("I")
        Q_stream.buffer(N_a).average().save("Q")

my_job = my_qm.simulate(
    powerRabiProg,
    SimulationConfig(int(600000), simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)])),
)  # Use LoopbackInterface to simulate the response of the qubit
time.sleep(1.0)

## Retrieving results of the experiments
my_powerRabi_results = my_job.result_handles
I1 = my_powerRabi_results.I.fetch_all()
Q1 = my_powerRabi_results.Q.fetch_all()
a1 = my_powerRabi_results.a.fetch_all()


# Processing the data
def fit_function(x_values, y_values, function, init_params=[0.003, 0.01, 0.001]):
    fitparams, conv = curve_fit(function, x_values, y_values, init_params)
    y_fit = function(x_values, *fitparams)

    return fitparams, y_fit


# Build a fitting tool for finding the right amplitude
# #(initial parameters to be adapted according to qubit and RR frequencies)
I_params, I_fit = fit_function(
    a1,
    I1,
    lambda x, A, drive_period, phi: (A * np.cos(2 * np.pi * x / drive_period - phi)),
    [0.004, 0.5, 0.0],
)
Q_params, Q_fit = fit_function(
    a1,
    Q1,
    lambda x, A, drive_period, phi: (A * np.cos(2 * np.pi * x / drive_period - phi)),
    [0.003, 0.5, 0.1],
)

plt.figure()
plt.plot(a1, I1, marker="x", color="blue", label="I")
plt.plot(a1, Q1, marker="o", color="green", label="Q")
plt.plot(a1, I_fit, color="red", label="Sinusoidal fit")
plt.plot(a1, Q_fit, color="black", label="Sinusoidal fit")
plt.xlabel("Amplitude [a.u]")
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

print("The amplitude required to perform a X gate is", I_params[1] / 2)
