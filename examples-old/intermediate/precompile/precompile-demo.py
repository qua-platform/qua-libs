"""
precompile-demo.py: Showcasing usage of the precompile functionality
Author: Gal Winer - Quantum Machines
Created: 7/11/2020
Created on QUA version: 0.5.138
"""

import time
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
import numpy as np

arb_len = 1000

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
            "intermediate_frequency": 0,
            "operations": {
                "playOp": "constPulse",
                "arbOp": "arbPulse",
            },
        },
    },
    "pulses": {
        "constPulse": {
            "operation": "control",
            "length": 1000,  # in ns
            "waveforms": {"single": "const_wf"},
        },
        "arbPulse": {
            "operation": "control",
            "length": arb_len,  # in ns
            "waveforms": {"single": "arb_wf"},
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
        "arb_wf": {
            "type": "arbitrary",
            "samples": [0.2] * arb_len,
            "is_overridable": True,
        },
    },
}

QMm = QuantumMachinesManager()


with program() as prog:
    for ind in range(2000):
        play("arbPulse", "qe1")

QM1 = QMm.open_qm(config)
program_id = QM1.compile(prog)


def make_wf():
    return (np.sin(np.linspace(0, 10 * np.pi + np.random.uniform(0, 2 * np.pi), arb_len)) / 2).tolist()


def run_and_time_cjob(compiled_program):
    t1 = time.time()
    job = QM1.queue.add_compiled(
        compiled_program,
        overrides={
            "waveforms": {
                "arb_wf": make_wf(),
            }
        },
    ).wait_for_execution()
    t2 = time.time()
    print(f"{t2 - t1}")
    return t2 - t1


def run_and_time_job(prog, config):
    t1 = time.time()
    config["waveforms"]["arb_wf"]["samples"] = make_wf()
    QM1 = QMm.open_qm(config)
    job = QM1.queue.add(prog).wait_for_execution()
    t2 = time.time()
    print(f"{t2 - t1}")
    return t2 - t1


n = 10
t_c = [run_and_time_cjob(program_id) for x in range(n)]
t = [run_and_time_job(prog, config) for x in range(n)]

print("#" * 50)
print(f"Without precompile:{np.mean(t):.2}s +/- {np.std(t):.2}s")
print(f"With precompile: {np.mean(t_c):.2}s +/- {np.std(t_c):.2}s")
print("#" * 50)
