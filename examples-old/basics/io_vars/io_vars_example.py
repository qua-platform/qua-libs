"""
io_vars_example.py: Demonstrating IO variable usage
Author: Arthur Strauss & Gal Winer - Quantum Machines
Created: 24/11/2020
Created on QUA version: 0.5.138
"""
import time

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
            "length": 1000,
            "waveforms": {"single": "const_wf"},
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
    },
}
QMm = QuantumMachinesManager()


def parse_io():
    io1_local = declare(int)
    assign(io1_local, IO1)
    assign(a, io1_local & int("1111", 2))
    assign(b, (io1_local >> 4) & int("1111", 2))


with program() as prog:

    a = declare(int)
    b = declare(int)
    play("playOp", "qe1")
    pause()

    # assign(a, io1_local & int('1111', 2))
    # assign(b, (io1_local >> 4) & int('1111', 2))
    parse_io()
    save(a, "this")
    save(b, "that")

QM1 = QMm.open_qm(config)


# job = QM1.simulate(prog,
#                    SimulationConfig(int(1000)))
parse_io()
job = QM1.execute(prog)
while job.is_paused():
    QM1.set_io1_value(int("10010011", 2))
    QM1.set_io2_value(6)
    job.resume()

res = job.result_handles
res.wait_for_all_values()
print(res.this.fetch_all())
print(res.that.fetch_all())
