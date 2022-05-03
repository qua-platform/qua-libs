"""
chirp.py: Introduction to chirp
Author: Gal Winer - Quantum Machines
Created: 17/01/2021
Revised by Tomer Feld - Quantum Machines
Revision date: 04/04/2022
Created on QUA version: 0.8.477
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import numpy as np
import matplotlib.pyplot as plt
from configuration import *

qmm = QuantumMachinesManager()

### First example - Linear chirp

rate = int(3e8 / pulse_duration)
units = "Hz/nsec"
with program() as prog:
    play("const", "qe1", chirp=(rate, units))

job = qmm.simulate(config, prog, SimulationConfig(int(pulse_duration // 4)))  # in clock cycles, 4 ns

samples = job.get_simulated_samples()
x = samples.con1.analog["1"]
NFFT = 2**10
Fs = 1e9
ax1 = plt.subplot(111)
Pxx, freqs, bins, im = plt.specgram(x, NFFT=NFFT, Fs=Fs, noverlap=100, cmap=plt.cm.gist_heat)
plt.show()
ax1.set_xticklabels((ax1.get_xticks() * 1e6).astype(int))
ax1.set_yticklabels((ax1.get_yticks() / 1e6).astype(int))
plt.xlabel("t [us]")
plt.ylabel("f [MHz]")


### Second example: Piecewise linear chirp to form a quadratic chirp

f_start = 1e6
f_end = 4e8
n_segs = 40
dt = pulse_duration / n_segs

a = (f_end - f_start) / (pulse_duration) ** 2

time_vec = dt * np.array(range(n_segs + 1))
freq_vec = a * time_vec**2 + f_start
rates = (np.diff(freq_vec) / (pulse_duration / n_segs)).astype(int).tolist()
units = "Hz/nsec"
with program() as prog:
    update_frequency("qe1", f_start)
    play("const", "qe1", chirp=(rates, units))

job = qmm.simulate(config, prog, SimulationConfig(int(pulse_duration // 4)))  # in clock cycles, 4 ns

samples = job.get_simulated_samples()

x = samples.con1.analog["1"]
NFFT = 2**10
Fs = 1e9
plt.figure()
ax1 = plt.subplot(111)
Pxx, freqs, bins, im = plt.specgram(x, NFFT=NFFT, Fs=Fs, noverlap=1000, cmap=plt.cm.gist_heat)
plt.show()
ax1.set_xticklabels((ax1.get_xticks() * 1e6).astype(int))
ax1.set_yticklabels((ax1.get_yticks() / 1e6).astype(int))
plt.title("Quadratic Chirp")
plt.xlabel("t [us]")
plt.ylabel("f [MHz]")
