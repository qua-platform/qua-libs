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


def resonator_spectroscopy(res_freq):
    x_vals = np.linspace(-5, 5, 100)
    a_vals = (1 / (1 + x_vals ** 2)).tolist()
    freqs = np.linspace(0.95 * res_freq, 1.05 * res_freq, 100).astype(int).tolist()

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


cal_graph = ProgramGraph()
#
a = QuaNode('resonator_spect', resonator_spectroscopy)

a.input = {'res_freq': int(100e6 + (random() - 0.5) * 10e6)}
a.quantum_machine = QM
a.simulation_kwargs = sim_args
a.output_vars = {'I', 'Q', 'freqs'}
cal_graph.add_nodes([a])
cal_graph.run()
print(a.result)
plt.plot(a.result['freqs'], np.sqrt(a.result['Q'] ** 2 + a.result['I'] ** 2))
