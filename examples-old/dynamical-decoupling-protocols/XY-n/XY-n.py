"""
CPMG.py: XY-n protocol
Author: Gal Winer - Quantum Machines
Created: 8/11/2020
Created on QUA version: 0.5.138
"""

import matplotlib.pyplot as plt
from configuration import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig, LoopbackInterface
import numpy as np
from configuration import config

QMm = QuantumMachinesManager()

# Create a quantum machine based on the configuration.

QM1 = QMm.open_qm(config)

NAVG = 10
NTAU = 10
tau_vec = np.linspace(8, 4000, NTAU).astype(int).tolist()


def XY_n_asym(tau, n):
    ind = declare(int)
    with for_(ind, 0, ind < n, ind + 1):
        wait(tau, "qe1")
        play("X", "qe1")
        wait(tau, "qe1")
        play("Y", "qe1")


def XY_n_sym(tau, n):
    ind = declare(int)
    wait(tau / 2, "qe1")
    with for_(ind, 0, ind < n, ind + 1):
        play("X", "qe1")
        wait(tau, "qe1")
        play("Y", "qe1")
        wait(tau, "qe1")
    wait(tau / 2, "qe1")


with program() as XY8:
    I = declare(fixed)
    Q = declare(fixed)
    n = declare(int)
    tau = declare(int)
    out_str = declare_stream()
    th = declare(fixed, value=0)
    s1 = declare(int, value=1)
    s0 = declare(int, value=0)
    with for_(n, 0, n < NAVG, n + 1):
        with for_each_(tau, tau_vec):
            XY_n_sym(tau, 8)
            align("qe1", "rr")
            measure(
                "readout",
                "rr",
                None,
                demod.full("integW1", I),
                demod.full("integW2", Q),
            )
            save(I, out_str)
            with if_(I > th):
                save(s1, out_str)
            with else_():
                save(s0, out_str)

    with stream_processing():
        out_str.save_all("tau_vec")

job = QM1.simulate(
    XY8,
    SimulationConfig(int(100000), simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)])),
)

samples = job.get_simulated_samples()
samples.con1.plot()
