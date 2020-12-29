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
N_a = int(a_max / da)  # Number of steps
N_max = 3


qmManager = QuantumMachinesManager("3.122.60.129")  # Reach OPX's IP address
my_qm = qmManager.open_qm(config)  # Generate a Quantum Machine based on the configuration described above

with program() as powerRabiProg:  # Power Rabi QUA program
    I = declare(fixed)  # QUA variables declaration
    Q = declare(fixed)
    a = declare(fixed)  # Sweeping parameter over the set of amplitudes
    Nrep = declare(int)  # Variable looping for repetitions of experiments
    I_stream = declare_stream()  # Declare streams to store I and Q components
    Q_stream = declare_stream()

    with for_(a, 0.00, a <= a_max, a + da):  # Sweep from 0 to 1 V the amplitude
        with for_(Nrep, 0, Nrep < N_max, Nrep + 1):  # Do 10 times the experiment
            play('gauss_pulse' * amp(a), 'qubit')  # Modulate the Gaussian pulse with the varying amplitude a
            align("qubit", "RR")
            measure('meas_pulse', 'RR', 'samples', ('integW1', I), ('integW2', Q))
            save(I, I_stream)  # Save the results
            save(Q, Q_stream)

        save(a, 'a')
    with stream_processing():
        I_stream.buffer(N_max).save_all('I')
        Q_stream.buffer(N_max).save_all('Q')

my_job = my_qm.simulate(powerRabiProg,
                        SimulationConfig(int(100000), simulation_interface=LoopbackInterface(
                            [("con1", 1, "con1", 1)])))  # Use LoopbackInterface to simulate the response of the qubit
time.sleep(1.0)

## Retrieving results of the experiments
my_powerRabi_results = my_job.result_handles
I1 = my_powerRabi_results.I.fetch_all()['value']
Q1 = my_powerRabi_results.Q.fetch_all()['value']
a1 = my_powerRabi_results.a.fetch_all()['value']


# Processing the data
def fit_function(x_values, y_values, function, init_params=[0.003, 0.01, 0.001]):
    fitparams, conv = curve_fit(function, x_values, y_values, init_params)
    y_fit = function(x_values, *fitparams)

    return fitparams, y_fit


I_avg = []
Q_avg = []
for i in range(len(I1)):
    I_avg.append((np.mean(I1[i])))
    Q_avg.append((np.mean(Q1[i])))

# Build a fitting tool for finding the right amplitude
# #(initial parameters to be adapted according to qubit and RR frequencies)
I_params, I_fit = fit_function(a1,
                               I_avg,
                               lambda x, A, drive_period, phi: (A * np.cos(2 * np.pi * x / drive_period - phi)),
                               [0.003, 0.2, 0.1]
                               )
Q_params, Q_fit = fit_function(a1,
                               Q_avg,
                               lambda x, A, drive_period, phi: (A * np.cos(2 * np.pi * x / drive_period - phi)),
                               [0.003, 0.2, 0.1]
                               )

plt.figure()
plt.plot(a1, I_avg, marker='x', color='blue', label='I')
plt.plot(a1, Q_avg, marker='o', color='green', label='Q')
plt.plot(a1, I_fit, color='red', label='Sinusoidal fit')
plt.plot(a1, Q_fit, color='black', label='Sinusoidal fit')
plt.xlabel('Amplitude [a.u]')
plt.ylabel('Measured signal [a.u]')
plt.axvline(I_params[1] / 2, color='red', linestyle='--')
plt.axvline(I_params[1], color='red', linestyle='--')
plt.annotate("", xy=(I_params[1], 0), xytext=(I_params[1] / 2, 0), arrowprops=dict(arrowstyle="<->", color='red'))
plt.annotate("$\pi$", xy=(I_params[1] / 2 - 0.03, 0.1), color='red')
plt.show()

print("The amplitude required to perform a X gate is", I_params[1] / 2)
