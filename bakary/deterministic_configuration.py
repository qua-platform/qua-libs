import numpy as np
t = np.linspace(-3, 3, 16)
sigma = 1
gauss = 0.1*np.exp(-t ** 2 / (2*sigma**2))
lmda = 0.5
alpha = -1
d_gauss = lmda * (-t) * gauss / alpha
config = {

    'version': 1,

    'controllers': {

        "con1": {
            'type': 'opx1',
            'analog_outputs': {
                1: {'offset': +0.0},  # qubit 1 I
                2: {'offset': +0.0},  # qubit 1 Q
                3: {'offset': +0.0},  # flux line
            },
            'analog_inputs': {
                1: {'offset': +0.0},
            }
        },

    },

    'elements': {

        "qe1": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
            },
            'intermediate_frequency': 0,
            'operations': {
                'playOp': "mixedConst",
                'playOp2': "mixedConst2",
                'gaussOp': "mixedGauss",
                'dragOp': "DragPulse",

            },
        },
        "fluxline": {
            'singleInput': {
                "port": ("con1", 3),
            },
            'intermediate_frequency': 0,
            'hold_offset': {'duration': 100},
            'operations': {
                'iSWAP': "constPulse",

            },
        }

    },

    "pulses": {
        "mixedConst": {
            'operation': 'control',
            'length': 100,
            'waveforms': {
                'I': 'zero_wf',
                'Q': 'const_wf'
            }
        },
        "mixedConst2": {
            'operation': 'control',
            'length': 100,
            'waveforms': {
                'I': 'const_wf',
                'Q': 'zero_wf'
            }
        },
        "mixedGauss": {
            'operation': 'control',
            'length': len(gauss),
            'waveforms': {
                'I': 'gauss_wf',
                'Q': 'zero_wf'
            }
        },

        "constPulse": {
            'operation': 'control',
            'length': 100,
            'waveforms': {
                'single': 'const_wf'
            }
        },
        "DragPulse": {
            'operation': 'control',
            'length': len(gauss),
            'waveforms': {
                'I': 'gauss_wf',
                'Q': 'd_gauss_wf'
            }

        },
        "gaussianPulse": {
            'operation': 'control',
            'length': len(gauss),
            'waveforms': {
                'single': 'gauss_wf'
            }
        },

    },

    "waveforms": {
        'zero_wf': {
            'type': 'constant',
            'sample': 0.0
        },
        'const_wf': {
            'type': 'constant',
            'sample': 0.1
        },
        'gauss_wf': {
            'type': 'arbitrary',
            'samples': gauss
        },
        'd_gauss_wf': {
            'type': 'arbitrary',
            'samples': d_gauss
        },

    },
}