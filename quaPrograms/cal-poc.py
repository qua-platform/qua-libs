from ProgramNode import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *

from qm import LoopbackInterface
from qm import SimulationConfig
from random import random
import matplotlib.pyplot as plt
import numpy as np

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


def spectroscopy(res_freq, range_):
    # to simulate response
    x_vals = np.linspace(-range_ / 2, range_ / 2, 100)
    a_vals = (1 / (1 + x_vals ** 2)).tolist()

    freqs = np.linspace((1 - range_ / 20) * res_freq, (1 + range_ / 20) * res_freq, 100).astype(int).tolist()

    with program() as qua_prog:
        vec = declare(fixed, value=a_vals)
        freq_vec = declare(int, value=freqs)
        ind = declare(int)
        I = declare(fixed)
        Q = declare(fixed)
        I_stream = declare_stream()
        Q_stream = declare_stream()

        with for_(ind, 0, ind < 100, ind + 1):
            update_frequency('qe1', freq_vec[ind])
            update_frequency('qe2', freq_vec[ind])
            play('playOp' * amp(vec[ind]), 'qe1')
            measure('readoutOp', 'qe2', 'raw', demod.full('x', I), demod.full('y', Q))
            save(I, I_stream)
            save(Q, Q_stream)
            save(freq_vec[ind], 'freqs')

        with stream_processing():
            I_stream.with_timestamps().save_all('I')
            Q_stream.with_timestamps().save_all('Q')
    return qua_prog


def rand_freq(freq, range_):
    return {'rand_freq': int(freq + (random() - 0.5) * range_)}


def extract_res_freq(freqs, I, Q, name):
    res_freq = freqs[np.argmax(np.sqrt(I ** 2 + Q ** 2))]

    plt.plot(freqs * 1e-6, np.sqrt(I ** 2 + Q ** 2), label=name)
    plt.axvline(x=res_freq * 1e-6, color='r')
    plt.xlabel('Frequence [MHz]')
    plt.ylabel('Response')
    plt.legend()

    return {'res_freq': res_freq}


def blobs(I, Q):
    plt.figure()
    plt.hist2d(I, Q, bins=100)
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.title('thermal distribution')

    # return dict()


r = PyNode('rand_resonator_freq', rand_freq, {'freq': 100e6, 'range_': 10e6}, {'rand_freq'})

a = QuaNode('resonator_spect', spectroscopy)
a.input = {'res_freq': r.output('rand_freq'), 'range_': 10}
a.quantum_machine = QM
a.simulation_kwargs = sim_args
a.output_vars = {'I', 'Q', 'freqs'}

b = PyNode('extract_res_freq', extract_res_freq)
b.input = {'freqs': a.output('freqs'), 'I': a.output('I'), 'Q': a.output('Q'), 'name': 'Readout resonator'}
b.output_vars = {'res_freq'}

c = PyNode('rand_qubit_freq', rand_freq, {'freq': b.output('res_freq'), 'range_': 1e6}, {'rand_freq'})

d = QuaNode('qubit_spect', spectroscopy)
d.input = {'res_freq': c.output('rand_freq'), 'range_': 1}
d.output_vars = {'freqs', 'I', 'Q'}
d.quantum_machine = QM
d.simulation_kwargs = sim_args

e = PyNode('extract_qubit_freq', extract_res_freq)
e.input = {'freqs': d.output('freqs'), 'I': d.output('I'), 'Q': d.output('Q'), 'name': 'Qubit'}
e.output_vars = {'res_freq'}

g = PyNode('IQ_blobs', blobs, {'I': d.output('I'), 'Q': d.output('Q')})

cal_graph = ProgramGraph()
cal_graph.add_nodes([a, b, r, c, d, e, g])
cal_graph.add_edges([(e, g)])

cal_graph.run()

print("TO visualize graph put the following string in webgraphviz.com:\n")
print(cal_graph.export_dot_graph())
