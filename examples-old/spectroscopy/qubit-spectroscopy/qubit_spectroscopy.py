"""
qubit_spectroscopy.py: Qubit spectroscopy to find resonance frequency
Author: Niv Drucker - Quantum Machines
Created: 8/11/2020
Created on QUA version: 0.6.156
"""

from configuration import config
from qm.qua import *
from qm import SimulationConfig
import numpy as np
import matplotlib.pyplot as plt
from qm.QuantumMachinesManager import QuantumMachinesManager

###############
# rr_spec_prog:
###############

f_min = 37e6
f_max = 43e6
df = 0.2e6


with program() as qubit_spectroscopy:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)  # Averaging
    f = declare(int)  # Frequencies
    I = declare(fixed)
    Q = declare(fixed)

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < 10, n + 1):

        with for_(f, f_min, f < f_max, f + df):

            update_frequency("qubit", f)
            play("saturation", "qubit")

            align("qubit", "rr")
            measure(
                "readout",
                "rr",
                None,
                demod.full("integW1", I),
                demod.full("integW2", Q),
            )

            save(I, "I")
            save(Q, "Q")


qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)
job = qm.simulate(qubit_spectroscopy, SimulationConfig(10000))
samples = job.get_simulated_samples()
samples.con1.plot()
