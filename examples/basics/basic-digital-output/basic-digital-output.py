"""
basic-digital-output.py: introduction to digital output (trigger out)
Author: Gal Winer - Quantum Machines
Created: 6/1/2021
Created on QUA version: 0.6.393
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *

QMm = QuantumMachinesManager()

with program() as prog:
    play("playOp", "qe1")
    wait(500, "qe2")
    play("playOp", "qe2")
    wait(500, "qe2")
    play("constPulse_trig", "qe2")
    wait(500, "qe2")
    play("constPulse_stutter", "qe2")


QM1 = QMm.open_qm(config)
job = QM1.simulate(prog, SimulationConfig(int(3000)))  # in clock cycles, 4 ns

samples = job.get_simulated_samples()
samples.con1.plot()
