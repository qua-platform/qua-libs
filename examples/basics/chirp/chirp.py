"""
chirp.py: Introduction to chirp
Author: Gal Winer - Quantum Machines
Created: 17/01/2021
Created on QUA version: 0.8.477
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import numpy as np
import matplotlib.pyplot as plt

pulse_duration = 4e5

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},
            },
        }
    },
    "elements": {
        "qe1": {
            "singleInput": {"port": ("con1", 1)},
            "intermediate_frequency": 5e6,
            "operations": {
                "playOp": "constPulse",
            },
        },
    },
    "pulses": {
        "constPulse": {
            "operation": "control",
            "length": pulse_duration,  # in ns
            "waveforms": {"single": "const_wf"},
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
    },
}
QMm = QuantumMachinesManager()
rate = int(3e8 / pulse_duration)
units = "Hz/nsec"
with program() as prog:
    play("playOp", "qe1", chirp=(rate, units))

QM1 = QMm.open_qm(config)
job = QM1.simulate(
    prog, SimulationConfig(int(pulse_duration // 4))
)  # in clock cycles, 4 ns

samples = job.get_simulated_samples()
# samples.con1.plot()
x = samples.con1.analog["1"]
NFFT = 2 ** 10
Fs = 1e9
ax1 = plt.subplot(111)
Pxx, freqs, bins, im = plt.specgram(
    x, NFFT=NFFT, Fs=Fs, noverlap=100, cmap=plt.cm.gist_heat
)
plt.show()
ax1.set_xticklabels((ax1.get_xticks() * 1e6).astype(int))
ax1.set_yticklabels((ax1.get_yticks() / 1e6).astype(int))
plt.xlabel("t[us]")
plt.ylabel("f[MHz]")

# second example: piecewise linear chirp to form a quadratic chirp
config["pulses"]["constPulse"]["length"] = 5e5
f_start = 1e6
f_end = 4e8
n_segs = 40
dt = pulse_duration / n_segs

a = (f_end - f_start) / (pulse_duration) ** 2

time_vec = dt * np.array(range(n_segs + 1))
freq_vec = a * time_vec ** 2 + f_start
rates = (np.diff(freq_vec) / (pulse_duration / n_segs)).astype(int).tolist()
# rate=int(3e8/pulse_duration)
units = "Hz/nsec"
with program() as prog:
    update_frequency("qe1", f_start)
    play("playOp", "qe1", chirp=(rates, units))

QM1 = QMm.open_qm(config)
job = QM1.simulate(
    prog, SimulationConfig(int(pulse_duration // 4))
)  # in clock cycles, 4 ns

samples = job.get_simulated_samples()

x = samples.con1.analog["1"]
NFFT = 2 ** 10
Fs = 1e9
plt.figure()
ax1 = plt.subplot(111)
Pxx, freqs, bins, im = plt.specgram(
    x, NFFT=NFFT, Fs=Fs, noverlap=1000, cmap=plt.cm.gist_heat
)
plt.show()
ax1.set_xticklabels((ax1.get_xticks() * 1e6).astype(int))
ax1.set_yticklabels((ax1.get_yticks() / 1e6).astype(int))
plt.xlabel("t[us]")
plt.ylabel("f[MHz]")
