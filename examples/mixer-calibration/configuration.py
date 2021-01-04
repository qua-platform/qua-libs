import numpy as np

pulse_len = 100
qubit_IF = 50e6
qubit_LO = 2e9


def IQ_imbalance_correction(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g ** 2) * (2 * c ** 2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


config = {
    'version': 1,

    'controllers': {
        "con1": {
            'type': 'opx1',
            'analog_outputs': {
                1: {'offset': +0.0},  # I
                2: {'offset': +0.0},  # Q
            }
        }
    },

    'elements': {
        "qubit": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                'lo_frequency': qubit_LO,
                'mixer': 'mixer_qubit'
            },
            'intermediate_frequency': qubit_IF,
            'operations': {
                'test_pulse': "constPulse",
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
                'I': 'const_wf',
                'Q': 'const_wf'
            }
        },

    },

    "waveforms": {
        'const_wf': {
            'type': 'constant',
            'sample': 0.2
        },
    },

    'mixers': {
        'mixer_qubit': [
            {'intermediate_frequency': qubit_IF, 'lo_frequency': qubit_LO,
             'correction': IQ_imbalance_correction(0, 0)}
        ],
    }
}
