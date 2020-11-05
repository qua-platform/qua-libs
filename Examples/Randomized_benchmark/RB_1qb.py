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
t = np.linspace(-3, 3, 1000)
gauss = (0.2 * np.exp(-t ** 2)).tolist()
config = {

    'version': 1,

    'controllers': {

        "con1": {
            'type': 'opx1',
            'analog_outputs': {
                1: {'offset': +0.0},  # qubit 1
                2: {'offset': +0.0}what,  # flux line
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
                'Rx_pi/2': "XPulse",
                'Rx_pi': "XPulse",
                'Rx_-pi/2': "XPulse",
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
            'length': 1000,
            'waveforms': {
                'I': 'gauss_wf',
                'Q': 'gauss_wf'
            }
        },
        "XPulse": {
            'operation': 'control',
            'length': 1000,
            'waveforms': {
                'I': 'gauss_wf',
                'Q': 'zero_wf'
            }
        },
        "readoutPulse": {
            'operation': 'measure',
            'length': 1000,
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
            'cosine': [1.0] * 250,
            'sine': [0.0] * 250
        },
        'yWeights': {
            'cosine': [0.0] * 250,
            'sine': [1.0] * 250
        }
    }
}

# Open communication with the server.
QMm = QuantumMachinesManager()

def make_cliffords():
    cliffords = np.empty([0, 2], dtype=int)
    # x_ops = ["0", "Rz_pi", "Rz_pi/2", "Rz_-pi/2", ["Rx_pi/2", "Rz_pi/2"], ["Rx_-pi/2", "Rz_pi/2"]]
    # x_anti_ops = ["0", "Rz_pi", "Rz_-pi/2", "Rz_pi/2", ["Rz_-pi/2", "Rx_-pi/2"], ["Rx_pi/2", "Rz_-pi/2"]]
    x_transformation = np.array([[0, 0], [0, 2], [0, 1], [0, 3], [1, 1], [3, 1]])
    # z_ops = ["0", "Rx_pi/2", "Rx_pi", "Rx_-pi/2"]
    # z_anti_ops = ["0", "Rx_-pi/2", "Rx_pi", "Rx_pi/2"]
    z_transformation = np.array([[0, 0], [3, 0], [2, 0], [1, 0]])

    for op1 in x_transformation:
        for op2 in z_transformation:
            cliffords = np.append(cliffords, [(op1 + op2) % 4], axis=0)
    return cliffords

def invert_circuit(circ):
    return invert_clifford(sum(circ)%4)

def invert_clifford(clifford):
    inv_clifford = (4 - clifford) % 4
    return inv_clifford


def play_clifford(clifford):
    z_state = clifford[1]
    x_state = clifford[0]

    if z_state == 0:
        pass
    elif z_state == 1:
        frame_rotation(np.pi/2,'qe1')
    elif z_state == 2:
        frame_rotation(np.pi, 'qe1')
    elif z_state == 3:
        frame_rotation(-np.pi / 2, 'qe1')

    if x_state == 0:
        pass
    elif x_state == 1:
        play('Rx_-pi/2', 'qe1')
    elif x_state == 2:
        play('Rx_pi', 'qe1')
    elif x_state == 3:
        play('Rx_pi/2', 'qe1')


def randomize_circuit(n_gates: int):

    cliffords = make_cliffords()
    circuit = np.empty([0, 2], dtype=int)
    for ind in np.random.randint(0, len(cliffords), size=(1, n_gates)):
        circuit = np.append(circuit, cliffords[ind], axis=0)

    return circuit


QM1 = QMm.open_qm(config)

N_avg = 10
circuit_depth_vec = list(range(500, 1000, 100))

with program() as RBprog:
    N = declare(int)
    I = declare(fixed)
    out_str = declare_stream()

    with for_(N, 0, N < N_avg, N + 1):
        for depth in circuit_depth_vec:
            circ = randomize_circuit(depth)
            for gate in circ:
                play_clifford(gate)
                play_clifford(invert_circuit(circ))
                measure('readout', 'qe1', None, integration.full('x', I))
                save(I, out_str)

    with stream_processing():
        out_str.save_all('out_stream')



job = QM1.simulate(RBprog,
                   SimulationConfig(int(50000)))

# res = job.result_handles
# I = res.out_stream.fetch_all()

samples = job.get_simulated_samples()

samples.con1.plot()
