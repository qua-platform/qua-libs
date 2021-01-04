"""
mixer_calibration_simulation.py: A simulation for the calibration for mixer imperfections
Author: Yoav Romach - Quantum Machines
Created: 22/12/2020
Created on QUA version: 0.7.411
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# Parameters for the imperfections
I_dc_offset = 0.05  # volt
Q_dc_offset = -0.03  # volt
gain_offset = 0.3  # dB
phase_offset_index = 100  # in units of dt below
empty_length = 216  # ns
length = 10000 + empty_length  # ns
dt = 0.01  # ns
real_length = length - empty_length

QMm = QuantumMachinesManager()

with program() as prog:
    with infinite_loop_():
        play('test_pulse', 'qubit')


QM1 = QMm.open_qm(config)
# QM1.set_output_dc_offset_by_element('qubit', 'I', -0.05)
# QM1.set_output_dc_offset_by_element('qubit', 'Q', 0.03)
# QM1.set_mixer_correction('mixer_qubit', int(qubit_IF), int(qubit_LO), IQ_imbalance(0.15, 0.3))

# No corrections are given, pulse is played as is into the I & Q ports.
job = QM1.simulate(prog, SimulationConfig(int(np.ceil(length / 4))))
samples = job.get_simulated_samples()

t = np.arange(0, real_length - 1, dt)
IQ = samples.con1.analog

# This code simulates the imperfections caused by the mixer.
i = (1 + gain_offset / 2) * np.roll(np.interp(t, [x for x in range(real_length)], IQ.get('1')[empty_length:]),
                                    -phase_offset_index) + I_dc_offset
q = (1 - gain_offset / 2) * np.roll(np.interp(t, [x for x in range(real_length)], IQ.get('2')[empty_length:]),
                                    +phase_offset_index) + Q_dc_offset

# This is the
LO = np.cos(t * qubit_LO * 1e-9)
RF = LO * (i + 1j * q)

# plt.figure
# f, Pxx_den=signal.periodogram(RF, 1e11)
# plt.semilogy(f/1e6, Pxx_den)
# plt.show()

plt.figure
RF_fft = np.fft.fft(RF)
RF_fft = RF_fft[:int(np.ceil(len(RF_fft)/2))]
freqs = np.fft.fftfreq(RF.size, dt)
freqs = freqs[:int(np.ceil(len(freqs)/2))]
plt.plot(freqs, 20 * np.log10(np.abs(RF_fft)))
plt.plot(freqs, np.abs(RF_fft))
plt.axis(xmin=0.75)
plt.axis(xmax=1.25)
