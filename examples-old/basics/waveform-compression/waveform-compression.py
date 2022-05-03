"""
waveform-compression.py: an intro to user selected sampling rate in arbitrary waveforms
Author: Gal Winer - Quantum Machines
Created: 26/12/2020
Created on QUA version: 0.6.393
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import config
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


QMm = QuantumMachinesManager()
length = 1000  # 1000 samples
sample_rate = 1e7  # in units of GSPS
pulse = ((np.random.random(length) / 2) - 0.5).tolist()  # arbitrary pulse samples must be a python list
b, a = signal.butter(3, 0.5)

pulse = signal.filtfilt(b, a, pulse)
# filter the random with LPF, get a random pulse with limited bandwidth

config["waveforms"]["arb_wf"] = {
    "type": "arbitrary",
    "samples": pulse,
    "sampling_rate": sample_rate,
}
config["pulses"]["arbPulse"] = {
    "operation": "control",
    "length": length * (1e9 / sample_rate),
    "waveforms": {"single": "arb_wf"},
}
config["elements"]["qe1"]["operations"]["arbOp"] = "arbPulse"

with program() as prog:
    play("arbOp", "qe1")

QM1 = QMm.open_qm(config)
job = QM1.simulate(prog, SimulationConfig(int(4000)))


samples = job.get_simulated_samples()
f, axes = plt.subplots(2, 2, sharex="col")

axes[0][0].plot(samples.con1.analog["1"])  # samples.con1.plot()
axes[1][0].plot(pulse)
axes[1][1].plot(pulse)
axes[0][1].axis("off")
axes[0][0].set_title("at 1 $\mu$s per sample")
axes[1][0].set_title("at 1 ns per sample")
axes[1][1].set_title("Pulse samples")
f.suptitle("Manually setting pulse sample rate")

# test: increase max_allowed_error for fitting a pulse into memory (>65k samples at 1e-4 error)
length = 400000
pulse = ((np.random.random(length) / 2) - 0.25).tolist()  # arbitrary pulse samples must be a python list
b, a = signal.butter(3, 0.01)
pulse = signal.filtfilt(b, a, pulse)

config["waveforms"]["arb_wf"].pop("sampling_rate", None)

config["waveforms"]["arb_wf"]["max_allowed_error"] = 1e-4  # default value
config["waveforms"]["arb_wf"]["samples"] = pulse
config["pulses"]["arbPulse"]["length"] = length
QM1 = QMm.open_qm(config)
job = QM1.simulate(prog, SimulationConfig(int(4000)))
#
samples = job.get_simulated_samples()
f, axes = plt.subplots(2, 2)
shift = 240  # shift depends on filter and on compiler details. may change for other versions
analog_out = samples.con1.analog["1"][shift:15000]
axes[0][0].plot(analog_out)
axes[1][0].plot(pulse[:15000])
axes[1][1].plot(np.array(analog_out) - np.array(pulse[: len(analog_out)]))
axes[1][1].set_ylim([-1e-6, 1e-6])
axes[0][1].axis("off")
axes[0][0].set_title("Analog output")
axes[1][0].set_title("Before compression")
axes[1][1].set_title("Difference")
f.suptitle("Automatic compression")
