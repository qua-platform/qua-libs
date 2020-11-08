from random import randint

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import numpy as np
from itertools import chain

################
# configuration:
################

# The configuration is our way to describe your setup.
pulse_len = 100
t = np.linspace(-3, 3, pulse_len)
gauss = (0.2 * np.exp(-t ** 2)).tolist()
config = {

    'version': 1,

    'controllers': {

        "con1": {
            'type': 'opx1',
            'analog_outputs': {
                1: {'offset': +0.0},  # qubit 1
                2: {'offset': +0.0},  # flux line
                3: {'offset': +0.0},  # qubit 2
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
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
            },
            'outputs': {
                'output1': ('con1', 1)
            },
            'intermediate_frequency': 0,
            'operations': {
                # 'playOp': "constPulse",
                'X/2': "X/2Pulse",
                'X': "XPulse",
                '-X/2': "-X/2Pulse",
                'Y/2': "Y/2Pulse",
                'Y': "YPulse",
                '-Y/2': "-Y/2Pulse",
                # 'gaussianOp': "gaussianPulse",
                'readout': 'readoutPulse'
            },
            'time_of_flight': 180,
            'smearing': 0
        },
    },

    "pulses": {
        "constPulse": {
            'operation': 'control',
            'length': pulse_len,
            'waveforms': {
                'I': 'gauss_wf',
                'Q': 'gauss_wf'
            }
        },
        "XPulse": {
            'operation': 'control',
            'length': pulse_len,
            'waveforms': {
                'I': 'gauss_wf',
                'Q': 'zero_wf'
            }
        },
        "X/2Pulse": {
            'operation': 'control',
            'length': pulse_len,
            'waveforms': {
                'I': 'gauss_wf',
                'Q': 'zero_wf'
            }
        },
        "-X/2Pulse": {
            'operation': 'control',
            'length': pulse_len,
            'waveforms': {
                'I': 'gauss_wf',
                'Q': 'zero_wf'
            }
        },
        "YPulse": {
            'operation': 'control',
            'length': pulse_len,
            'waveforms': {
                'I': 'zero_wf',
                'Q': 'gauss_wf'
            }
        },
        "Y/2Pulse": {
            'operation': 'control',
            'length': pulse_len,
            'waveforms': {
                'I': 'zero_wf',
                'Q': 'gauss_wf'
            }
        },
        "-Y/2Pulse": {
            'operation': 'control',
            'length': pulse_len,
            'waveforms': {
                'I': 'zero_wf',
                'Q': 'gauss_wf'
            }
        },
        "readoutPulse": {
            'operation': 'measure',
            'length': pulse_len,
            'waveforms': {
                'I': 'gauss_wf',
                'Q': 'gauss_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'x': 'xWeights',
                'y': 'yWeights'}
        }

    },

    "waveforms": {
        'const_wf': {
            'type': 'constant',
            'sample': 0.2
        },
        'gauss_wf': {
            'type': 'arbitrary',
            'samples': gauss
        },
        'zero_wf': {
            'type': 'constant',
            'sample': 0
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

cliffords = [['I'], ['X'], ['Y'], ['Y', 'X'],
             ['X/2', 'Y/2'], ['X/2', '-Y/2'], ['-X/2', 'Y/2'], ['-X/2', '-Y/2'], ['Y/2', 'X/2'], ['Y/2', '-X/2'],
             ['-Y/2', 'X/2'], ['-Y/2', '-X/2'],
             ['X/2'], ['-X/2'], ['Y/2'], ['-Y/2'], ['-X/2', 'Y/2', 'X/2'], ['-X/2', '-Y/2', 'X/2'],
             ['X', 'Y/2'], ['X', '-Y/2'], ['Y', 'X/2'], ['Y', '-X/2'], ['X/2', 'Y/2', 'X/2'], ['-X/2', 'Y/2', '-X/2']]


def recovery_clifford(state):
    operations = {'x': ['I'], '-x': ['Y'], 'y': ['X/2', '-Y/2'], '-y': ['-X/2', '-Y/2'], 'z': ['-Y/2'], '-z': ['Y/2']}
    return operations[state]


def transform_state(input_state, transformation):
    transformations = {'x': {'I': 'x', 'X/2': 'x', 'X': 'x', '-X/2': 'x', 'Y/2': 'z', 'Y': '-x', '-Y/2': '-z'},
                       '-x': {'I': '-x', 'X/2': '-x', 'X': '-x', '-X/2': '-x', 'Y/2': '-z', 'Y': 'x', '-Y/2': 'z'},
                       'y': {'I': 'y', 'X/2': 'z', 'X': '-y', '-X/2': '-z', 'Y/2': 'y', 'Y': 'y', '-Y/2': 'y'},
                       '-y': {'I': '-y', 'X/2': '-z', 'X': 'y', '-X/2': 'z', 'Y/2': '-y', 'Y': '-y', '-Y/2': '-y'},
                       'z': {'I': 'z', 'X/2': '-y', 'X': '-z', '-X/2': 'y', 'Y/2': '-x', 'Y': '-z', '-Y/2': 'x'},
                       '-z': {'I': '-z', 'X/2': 'y', 'X': 'z', '-X/2': '-y', 'Y/2': 'x', 'Y': 'z', '-Y/2': '-x'}}

    return transformations[input_state][transformation]


def play_clifford(clifford, state):
    for op in clifford:
        state = transform_state(state, op)
        if op != 'I':
            play(op, 'qe1')
    return state


def randomize_circuit(n_gates: int, init_state: str = 'x'):
    state = init_state
    for ind in range(n_gates):
        state = play_clifford(cliffords[np.random.randint(0, len(cliffords))], state)
    return state


QM1 = QMm.open_qm(config)

N_avg = 10
circuit_depth_vec = list(range(10, 50, 1))

with program() as RBprog:
    N = declare(int)
    I = declare(fixed)
    out_str = declare_stream()

    with for_(N, 0, N < N_avg, N + 1):
        for depth in circuit_depth_vec:
            final_state = randomize_circuit(depth)
            play_clifford(recovery_clifford(final_state), final_state)
            measure('readout', 'qe1', None, integration.full('x', I))
            save(I, out_str)

    with stream_processing():
        out_str.save_all('out_stream')

# job = QM1.simulate(RBprog,
#                    SimulationConfig(int(10000)))
job = QM1.execute(RBprog)

# def estimate_state(I, Q, d):


# return 0 with prob exp(-d/alpha)

# res = job.result_handles
# I = res.out_stream.fetch_all()

# samples = job.get_simulated_samples()
#
# samples.con1.plot()
