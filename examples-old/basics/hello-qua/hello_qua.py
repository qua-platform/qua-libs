"""
hello_qua.py: Your first QUA script
Author: Gal Winer - Quantum Machines
Created: 7/11/2020
Created on QUA version: 0.5.138
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig

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
            "length": 1000,  # in ns
            "waveforms": {"single": "const_wf"},
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
    },
}
QMm = QuantumMachinesManager()

with program() as prog:
    play("playOp", "qe1")

QM1 = QMm.open_qm(config)
job = QM1.simulate(prog, SimulationConfig(int(1000)))  # in clock cycles, 4 ns

samples = job.get_simulated_samples()
samples.con1.plot()
# In Pycharm, the plot will only show if the execution is "Run with Python Console"
