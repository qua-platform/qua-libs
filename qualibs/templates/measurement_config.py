config = {

    'version': 1,

    'controllers': {

        "con1": {
            'type': 'opx1',
            'analog_outputs': {
                1: {'offset': +0.0},
                2: {'offset': +0.},
                3: {'offset': +0.},  #

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

        "drivingElem": {
            "singleInput": {
                "port": ("con1", 1)
            },
            'intermediate_frequency': 1e6,
            'operations': {
                'rabiOp': "constPulse",
            },
        },
        "qubitSimElem": {
            "singleInput": {
                "port": ("con1", 3)
            },
            'intermediate_frequency': 0,
            'operations': {
                'shineOp': "constPulse",
            },
        },
        "measElem": {
            "singleInput": {
                "port": ("con1", 2)
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
        "onPulse": {
            'operation': 'control',
            'length': 1000,
            'waveforms': {
                'single': 'const_wf'
            },
            'digital_marker': 'ON',
        },
        "rabiPulse": {
            'operation': 'control',
            'length': 1000,
            'waveforms': {
                'single': 'const_wf'
            }
        },
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
            'cosine': [1.0] * 1000,
            'sine': [0.0] * 1000
        },
        'yWeights': {
            'cosine': [0.0] * 1000,
            'sine': [1.0] * 1000
        }
    }
}