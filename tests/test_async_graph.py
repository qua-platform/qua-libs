from qualibs.graph import PyNode, QuaNode, ProgramGraph
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import LoopbackInterface
from qm import SimulationConfig
from random import random
import matplotlib.pyplot as plt
import numpy as np

import asyncio
import time
import random
from colorama import Fore

config = {

    'version': 1,

    'controllers': {

        "con1": {
            'type': 'opx1',
            'analog_outputs': {
                1: {'offset': +0.0},
                2: {'offset': +0.},
                3: {'offset': +0.},

            },
            'digital_outputs': {
                1: {},
            },
            'analog_inputs': {
                1: {'offset': +0.0},
            }
        }
    },

    'elements': {

        "qe1": {
            "singleInput": {
                "port": ("con1", 3)
            },
            'intermediate_frequency': 100e6,
            'operations': {
                'playOp': "constPulse",
            },
        },
        "qe2": {
            "singleInput": {
                "port": ("con1", 2)
            },
            'outputs': {
                'output1': ('con1', 1)
            },
            'intermediate_frequency': 100e6,
            'operations': {
                'readoutOp': 'readoutPulse',

            },
            'time_of_flight': 28,
            'smearing': 0
        },
    },

    "pulses": {

        "constPulse": {
            'operation': 'control',
            'length': 1000,
            'waveforms': {
                'single': 'const_wf'
            }
        },
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
            'sample': 0.4
        },
    },
    'digital_waveforms': {
        'ON': {
            'samples': [(1, 0)]
        },
    },
    'integration_weights': {
        'xWeights': {
            'cosine': [1.0] * 500,
            'sine': [0.0] * 500
        },
        'yWeights': {
            'cosine': [0.0] * 500,
            'sine': [1.0] * 500
        }
    }
}

# Open communication with the server.
QMm = QuantumMachinesManager()

# Create a quantum machine based on the configuration.

QM = QMm.open_qm(config)

sim_args = {'simulate': SimulationConfig(int(1e5), simulation_interface=LoopbackInterface([("con1", 3, "con1", 1)]))}


async def a():
    rest = random.random()
    print(f"Node a resting for {rest}")
    await asyncio.sleep(rest)
    return {'x': rest}


async def b(x):
    rest = random.random()
    print(f"Node b resting for {x * rest}")
    await asyncio.sleep(x * rest)
    return {'zx': x * rest}


async def c(x):
    rest = random.random()
    print(f"Node c resting for {x * rest}")
    await asyncio.sleep(x * rest)
    return {'yx': x * rest}


async def d(zx, yx):
    print(f"Node c resting for {zx * yx}")
    await asyncio.sleep(zx * yx)
    print(f"Result: {yx * zx}")
    return {'zxyx': zx * yx}


a1 = PyNode('a', a, None, {'x'})
b1 = PyNode('b', b, {'x': a1.output('x')}, {'zx'})
c1 = PyNode('c', c, {'x': a1.output('x')}, {'yx'})
d1 = PyNode('d', d, {'zx': b1.output('zx'), 'yx': c1.output('yx')}, {'zxyx'})

g = ProgramGraph()
g.add_nodes([a1, b1, c1, d1])

for i in g.get_next():
    print(g.nodes[i].label)
