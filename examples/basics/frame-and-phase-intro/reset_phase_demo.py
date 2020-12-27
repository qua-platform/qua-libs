"""
reset_phase_demo.py:
Author: Gal Winer - Quantum Machines
Created: 8/11/2020
Created on QUA version: 0.5.138
"""

import matplotlib.pyplot as plt
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import numpy as np
from scipy.optimize import leastsq

config = {

    'version': 1,

    'controllers': {
        "con1": {
            'type': 'opx1',
            'analog_outputs': {
                1: {'offset': +0.0},
            },
        }
    },

    'elements': {
        "qubit": {
            "singleInput": {
                "port": ("con1", 1)
            },
            'intermediate_frequency': 5e6,
            'operations': {
                'const': "constPulse",
            },
        },
    },

    "pulses": {
        "constPulse": {
            'operation': 'control',
            'length': 1000,  # in ns
            'waveforms': {
                'single': 'const_wf'
            }
        },
    },

    "waveforms": {
        'const_wf': {
            'type': 'constant',
            'sample': 0.2
        },
    },
}
QMm = QuantumMachinesManager()

# Create a quantum machine based on the configuration.

QM1 = QMm.open_qm(config)

with program() as no_reset_ph:
    play('const', 'qubit')  # Plays a 5 MHz sin wave for 1 us (defined in config).
    update_frequency('qubit', int(10e6))  # Change to 10 MHz and play it
    play('const', 'qubit')

with program() as no_reset_ph_and_rotate:
    play('const', 'qubit')  # Plays a 5 MHz sin wave for 1 us (defined in config).
    update_frequency('qubit', int(10e6))  # Change to 10 MHz, rotate by pi, and play it
    frame_rotation(np.pi, 'qubit')
    play('const', 'qubit')


with program() as reset_ph:
    play('const', 'qubit')  # Plays a 5 MHz sin wave for 1 us (defined in config).
    update_frequency('qubit', int(10e6))  # Change to 10 MHz, reset phase, and play it
    reset_phase('qubit')
    play('const', 'qubit')


with program() as reset_ph_and_rotate:
    play('const', 'qubit')  # Plays a 5 MHz sin wave for 1 us (defined in config).
    update_frequency('qubit', int(10e6))  # Change to 10 MHz, reset phase, rotate by pi, and play it
    reset_phase('qubit')
    frame_rotation(np.pi, 'qubit')
    play('const', 'qubit')


job = QM1.simulate(no_reset_ph,
                   SimulationConfig(int(800)))
samples = job.get_simulated_samples()
ax1 = plt.subplot(411)
samples.con1.plot()
plt.title('No phase reset')

t = np.arange(1500, 2000)
data = samples.con1.analog['1'][1500:2000]
fit_func = lambda x: x[0]*np.sin(x[1]*t+x[2]) - data
est_amp, est_freq, est_phase = leastsq(fit_func, [0.2, 2*np.pi*10e-3, 0])[0]
t2 = np.arange(750, 2000)
plt.plot(t2, est_amp*np.sin(est_freq*t2 + est_phase))
plt.legend(('Output', '10MHz Fit'))

job = QM1.simulate(no_reset_ph_and_rotate,
                   SimulationConfig(int(800)))
samples = job.get_simulated_samples()
plt.subplot(412, sharex=ax1)
samples.con1.plot()
plt.title('Pi rotation for 10 MHz')

t = np.arange(1500, 2000)
data = samples.con1.analog['1'][1500:2000]
fit_func = lambda x: x[0]*np.sin(x[1]*t+x[2]) - data
est_amp, est_freq, est_phase = leastsq(fit_func, [0.2, 2*np.pi*10e-3, 0])[0]
t2 = np.arange(750, 2000)
plt.plot(t2, est_amp*np.sin(est_freq*t2 + est_phase))
plt.legend(('Output', '10MHz Fit'))

job = QM1.simulate(reset_ph,
                   SimulationConfig(int(800)))
samples = job.get_simulated_samples()
plt.subplot(413, sharex=ax1)
samples.con1.plot()
plt.title('With phase reset')

t = np.arange(1500, 2000)
data = samples.con1.analog['1'][1500:2000]
fit_func = lambda x: x[0]*np.sin(x[1]*t+x[2]) - data
est_amp, est_freq, est_phase = leastsq(fit_func, [0.2, 2*np.pi*10e-3, 0])[0]
t2 = np.arange(750, 2000)
plt.plot(t2, est_amp*np.sin(est_freq*t2 + est_phase))
plt.legend(('Output', '10MHz Fit'))


job = QM1.simulate(reset_ph_and_rotate,
                   SimulationConfig(int(800)))
samples = job.get_simulated_samples()
plt.subplot(414, sharex=ax1)
samples.con1.plot()
plt.title('With phase reset and pi rotation for 10 MHz')

t = np.arange(1500, 2000)
data = samples.con1.analog['1'][1500:2000]
fit_func = lambda x: x[0]*np.sin(x[1]*t+x[2]) - data
est_amp, est_freq, est_phase = leastsq(fit_func, [0.2, 2*np.pi*10e-3, 0])[0]
t2 = np.arange(750, 2000)
plt.plot(t2, est_amp*np.sin(est_freq*t2 + est_phase))
plt.legend(('Output', '10MHz Fit'))

plt.xlabel('time [ns]')
plt.tight_layout()
plt.show()
