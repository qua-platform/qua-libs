"""
reset_phase_demo.py:
Author: Gal Winer - Quantum Machines
Created: 8/11/2020
Created on QUA version: 0.5.138
"""

import matplotlib.pyplot as plt
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig, LoopbackInterface
import numpy as np
from scipy.optimize import curve_fit
from configuration import config

QMm = QuantumMachinesManager()

# Create a quantum machine based on the configuration.

QM1 = QMm.open_qm(config)

with program() as reset_ph:
    # reset_phase('qubit')

    play('const','qubit')
    play('const', 'qubit')
    update_frequency('qubit', int(10e6))
    reset_phase('qubit')
    play('const', 'qubit')

    update_frequency('qubit', int(20e6))
    reset_phase('qubit')
    play('const', 'qubit')

    update_frequency('qubit', int(10e6))
    play('const', 'qubit')

    update_frequency('qubit', int(20e6))
    play('const', 'qubit')

    update_frequency('qubit', int(20e6))
    reset_phase('qubit')
    frame_rotation(np.pi/2,'qubit')
    play('const', 'qubit')

    update_frequency('qubit', int(10e6))
    reset_frame('qubit')
    play('const', 'qubit')

    # play('X', 'qubit')

job = QM1.simulate(reset_ph,
                   SimulationConfig(int(1000), simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)])))

samples = job.get_simulated_samples()
samples.con1.plot()

# plt.figure()
# plt.plot(samples.con1.analog['1']/np.max(samples.con1.analog['1']))
# plt.plot(np.arange(230,2550+230),np.sin(2*np.pi*10e-3*np.arange(0,2550)))
