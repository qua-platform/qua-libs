from ProgramNode import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig, LoopbackInterface
from random import random

config = {

    'version': 1,

    'controllers': {

        "con1": {
            'type': 'opx1',
            'analog_outputs': {
                1: {'offset': 0},
                2: {'offset': 0}
            },
            'analog_inputs': {
                1: {'offset': 0}  # SET READ
            }
        },
    },

    'elements': {

        "qe1": {
            "singleInput": {
                "port": ("con1", 1)
            },
            'intermediate_frequency': 0,  # randomized frequency for testing
            'operations': {
                'playOp': "constPulse",
            },

        },

        "qe2": {
            "singleInput": {
                # the port we play on to influence qe2
                "port": ("con1", 2)
            },
            'intermediate_frequency': 0,
            'outputs': {
                # the port we measure to get the output of qe2
                'output1': ('con1', 1)
            },
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
            'integration_weights': {
                'integW1': 'integW1',
                'integW2': 'integW2',
            }
        },
    },
    'integration_weights': {
        'integW1': {
            'cosine': [1] * 1000,
            'sine': [0] * 1000
        },
        'integW2': {
            'cosine': [0] * 1000,
            'sine': [1] * 1000
        },
    },
    "waveforms": {
        'const_wf': {
            'type': 'constant',
            'sample': 0.2
        },
    }
}

# Open communication with the server.
QMm = QuantumMachinesManager()

# Create a quantum machine based on the configuration.

QM = QMm.open_qm(config)

sim_args = {'simulate': SimulationConfig(int(1e3), simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)]))}


def resonator_spectroscopy(res_freq):
    with program() as qua_prog:
        stream = declare_stream()
        x = declare(fixed, value=(res_freq - 90e6) / 100e6)
        save(x, 'x')
        int_freq = declare(int)
        I = declare(fixed)
        Q = declare(fixed)

        with for_(int_freq, 90e6, int_freq < 110e6, int_freq + 1e6):
            update_frequency('qe1', int_freq)
            update_frequency('qe2', int_freq)
            play('playOp' * amp(1 / (1 + x * x)), 'qe1')
            measure('readoutOp', 'qe2', stream, ('integW1', I), ('integW2', Q))
            assign(x, x + 0.01)
            save(I, 'I')
            save(Q, 'Q')
        with stream_processing():
            stream.input1().with_timestamps().save_all('data')
    return qua_prog


cal_graph = ProgramGraph()

a = QuaNode('resonator_spect', resonator_spectroscopy)
a.input = {'res_freq': int(100e6 + (random() - 0.5) * 10e6)}
a.quantum_machine = QM
a.simulation_kwargs = sim_args
a.output_vars = {'data', 'x', 'I', 'Q'}
cal_graph.add_nodes([a])
cal_graph.run()

print(a.result)
