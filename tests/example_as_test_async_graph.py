from qualibs.graph import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import LoopbackInterface
from qm import SimulationConfig
from random import random

import time
import asyncio
import random

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
    rest = 5
    print(f"Node a resting for {rest}")
    await asyncio.sleep(rest)
    with program() as qua_prog:
        x = declare(fixed, value=rest)
        save(x, 'x')
    return qua_prog


async def b():
    rest = 3
    # m = m.result_handles
    print(f"Node b resting for {rest}")
    await asyncio.sleep(rest)
    # time.sleep(rest)
    return {'y': rest}


async def c():
    rest = 1
    print(f"Node c resting for {rest}")
    await asyncio.sleep(rest)
    # time.sleep(rest)
    return {'z': rest}


async def d(x, y):
    print(f"Node d resting for {x / y}")
    await asyncio.sleep(x / y)
    print(f"d Result: {x + y}")
    return {'x_y': x + y}


async def e(y, z, q):
    print(f"Node e resting for {y / z}")
    print(q.quantum_machine)
    await asyncio.sleep(y / z)
    print(f"e Result: {y + z}")
    return {'y_z': y + z}


def f(x):
    print(x)
    return {'y_z': x}


a1 = QuaNode('a', a, None, {'x'})
a1.quantum_machine = QM
a1.simulation_kwargs = sim_args
v = a1.deepcopy()
b1 = PyNode('b', b, None, {'y'})
c1 = PyNode('c', c, None, {'z'})
d1 = PyNode('d', d, {'x': v.output('x'), 'y': b1.output('y')}, {'x_y'})
e1 = d1.deepcopy()
e1.program = e
del e1.input_vars['x']
e1.input_vars['z'] = c1.output('z')
e1.input_vars['q'] = v.job()
e1.label = 'e'
e1.output_vars = {'y_z'}
db = GraphDB('here1.db')
g = ProgramGraph('hello', graph_db=db)
g.add_nodes(*[v, c1, b1, d1, e1])
new_g = ProgramGraph()
f1 = PyNode('f', f, {'x': 1}, {'y_z'})
new_g.add_nodes(f1)
# f1.input_vars.x = e1.output('y_z')
# g.run()
g.join(new_g)
n = g.deepcopy()
n.run()

print(n.export_dot_graph())
