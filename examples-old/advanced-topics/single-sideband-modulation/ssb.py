from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from configuration import config
from qm import LoopbackInterface
from qm import SimulationConfig
from random import random
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy import signal

nSamples = 100
samplingRate = 10e6
pulseDuration = nSamples / samplingRate
pulseDuration_ns = pulseDuration / 1e-9
t = np.linspace(0, pulseDuration, nSamples)

freqs = np.linspace(1, 4, 15).tolist()
phases = np.zeros_like(freqs).tolist()
amps = np.ones_like(phases).tolist()
m = np.sum(
    list(
        map(
            lambda a: a[2] * np.sin(2 * pi * a[0] * 1e6 * t + a[1]),
            zip(freqs, phases, amps),
        )
    ),
    0,
)
m = m / max(m) / 2
m = m.tolist()
mc = signal.hilbert(m)
wf1 = np.real(mc)
wf2 = np.imag(mc)

config["pulses"]["ssbPulse"]["length"] = len(wf1) * (1e9 / samplingRate)
config["waveforms"]["wf1"]["samples"] = wf1
config["waveforms"]["wf1"]["sampling_rate"] = samplingRate
config["waveforms"]["wf2"]["samples"] = wf2
config["waveforms"]["wf2"]["sampling_rate"] = samplingRate
# Open communication with the server.
QMm = QuantumMachinesManager()

# Create a quantum machine based on the configuration.

QM = QMm.open_qm(config)

with program() as prog:
    play("ssb", "ssbElement")

job = QM.simulate(prog, SimulationConfig(int(1e5)))
res = job.result_handles

samples = job.get_simulated_samples()
out_vector = samples.con1.analog["1"]
f, Pxx_den = signal.periodogram(out_vector, 1e9)

plt.figure()
[plt.axvline(x=f + 50, color="k", linestyle="--") for f in freqs]
plt.semilogy(f / 1e6, Pxx_den)
plt.xlabel("Freq [MHz]")
plt.ylim([1e-15, 1e-8])
plt.xlim([40, 60])

plt.title("Single-sideband modulated signal")
plt.grid(True, which="both")
plt.show()
