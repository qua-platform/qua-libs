"""
raw_adc_intro.py: Demonstrate recording raw analog input
Author: Gal Winer - Quantum Machines
Created: 7/11/2020
Created on QUA version: 0.5.138
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm.qua import math
from qm import LoopbackInterface
from qm import SimulationConfig
import numpy as np
import matplotlib.pyplot as plt
pulse_len=1000
config = {

    'version': 1,

    'controllers': {

        "con1": {
            'type': 'opx1',
            'analog_outputs': {
                1: {'offset': +0.0},
            },

            'analog_inputs': {
                1: {'offset': +0.0},
            }
        }
    },

    'elements': {

        "qe1": {
            "singleInput": {
                "port": ("con1", 1)
            },
            'outputs': {
                'output1': ('con1', 1)
            },
            'intermediate_frequency': 1e6,
            'operations': {
                'readoutOp': 'readoutPulse',

            },
            'time_of_flight': 180,
            'smearing': 0
        },
    },

    "pulses": {
        'readoutPulse': {
            'operation': 'measure',
            'length': 1000,
            'waveforms': {
                'single': 'const_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'x': 'xWeights',
                'y': 'yWeights'}
        },
    },

    "waveforms": {
        'const_wf': {
            'type': 'constant',
            'sample': 0.2
        },
    },
    'digital_waveforms': {
        'ON': {
            'samples': [(1, 0)]
        },

    },
    'integration_weights': {
        'xWeights': {
            'cosine': [1.0] * (pulse_len // 4),
            'sine': [0.0] * (pulse_len // 4)
        },
        'yWeights': {
            'cosine': [0.0] * (pulse_len // 4),
            'sine': [1.0] * (pulse_len // 4)
        }
    }
}

# Open communication with the server.
QMm = QuantumMachinesManager()

# Create a quantum machine based on the configuration.

QM1 = QMm.open_qm(config)

#1. Assigning values to variables and saving variables to tags
with program() as saving_a_var:
    a = declare(int, value=5)
    b = declare(fixed, value=0.23)
    save(a, 'a_var')
    assign(a, 7)
    save(a, 'a_var')
    assign(a, a + 1)
    save(a, 'a_var')
    assign(b, math.sin2pi(b))
    save(b, 'b_var')

job = QM1.simulate(saving_a_var,
                   SimulationConfig(1000))

res = job.result_handles
a = res.a_var.fetch_all()['value']
b = res.b_var.fetch_all()['value']

print("##################")
print("1: Assigning values to variables and saving variables to tags")
print(f"a={a}")
print(f"b={b}")
print("##################")


#2. Saving variables to streams and using stream processing
with program() as streamProg:
    out_str = declare_stream()
    a = declare(int)
    with for_(a, 0, a <= 100, a + 1):
        save(a, out_str)

    with stream_processing():
        out_str.average().save('out')

job = QM1.simulate(streamProg,
                   SimulationConfig(500, simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)])))

res = job.result_handles
out = res.out.fetch_all()

print("##################")
print("2: Saving variables to streams and using stream processing")
print(f"out={out}")
print("##################")


#3. Using the buffer operator in stream processing

with program() as streamProg_buffer:
    out_str = declare_stream()

    a = declare(int)
    with for_(a, 0, a <= 100, a + 1):
        save(a, out_str)

    with stream_processing():
        out_str.buffer(3).average().save_all('out')

job = QM1.simulate(streamProg_buffer,
                   SimulationConfig(500, simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)])))

res = job.result_handles
out = res.out.fetch_all()['value']

print("##################")
print("3: Using the buffer operator in stream processing")
print(f"out={out}")
print("##################")

# 4. Saving a stream to multiple tags
with program() as multiple_tags:
    out_str1 = declare_stream()


    a = declare(int)
    with for_(a, 0, a <= 100, a + 1):
        save(a, out_str1)


    with stream_processing():
        out_str1.buffer(1).average().save('out_avg')
        out_str1.save_all('out_raw')


job = QM1.simulate(multiple_tags,
                   SimulationConfig(500, simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)])))

res = job.result_handles
out_avg = res.out_avg.fetch_all()
out_raw = res.out_raw.fetch_all()['value']
print("##################")
print("4:Saving a stream to multiple tags")
print(f"out_avg={out_avg}")
print(f"out_raw={out_raw}")
print("##################")
