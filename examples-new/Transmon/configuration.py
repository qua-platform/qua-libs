import numpy as np
from scipy.signal.windows import gaussian
from qualang_tools.config.waveform_tools import drag_gaussian_pulse_waveforms


#######################
# AUXILIARY FUNCTIONS #
#######################

# IQ imbalance matrix
def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g ** 2) * (2 * c ** 2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


#############
# VARIABLES #
#############

qop_ip = '127.0.0.1'
qop_port = 80

# Qubits
qubit_IF = 50e6
qubit_LO = 7e9
mixer_qubit_g = 0.0
mixer_qubit_phi = 0.0

saturation_len = 1000
saturation_amp = 0.1
const_len = 100
const_amp = 0.1

gauss_len = 20
gauss_sigma = gauss_len / 5
gauss_amp = 0.35
gauss_wf = gauss_amp * gaussian(gauss_len, gauss_sigma)

pi_len = 40
pi_sigma = pi_len / 5
pi_amp = 0.35
pi_wf, pi_drag_wf = np.array(drag_gaussian_pulse_waveforms(pi_amp, pi_len, pi_sigma, alpha=0, delta=1))  # No DRAG when alpha=0

pi_half_len = pi_len
pi_half_sigma = pi_half_len / 5
pi_half_amp = pi_amp / 2
pi_half_wf, pi_half_drag_wf = np.array(drag_gaussian_pulse_waveforms(pi_half_amp, pi_half_len, pi_half_sigma, alpha=0, delta=1))  # No DRAG when alpha=0


# Resonator
resonator_IF = 60e6
resonator_LO = 5.5e9
mixer_resonator_g = 0.0
mixer_resonator_phi = 0.0

time_of_flight = 180

short_readout_len = 500
short_readout_amp = 0.4
readout_len = 5000
readout_amp = 0.2
long_readout_len = 50000
long_readout_amp = 0.1

# IQ Plane Angle
rotation_angle = (0.0/180) * np.pi

config = {
    'version': 1,
    'controllers': {
        'con1': {
            'type': 'opx1',
            'analog_outputs': {
                1: {'offset': 0.0},  # I qubit
                2: {'offset': 0.0},  # Q qubit
                3: {'offset': 0.0},  # I resonator
                4: {'offset': 0.0},  # Q resonator
            },
            'digital_outputs': {},
            'analog_inputs': {
                1: {'offset': 0.0, 'gain_db': 0},  # I from down-conversion
                2: {'offset': 0.0, 'gain_db': 0},  # Q from down-conversion
            },
        },
    },
    'elements': {
        'qubit': {
            'mixInputs': {
                'I': ('con1', 1),
                'Q': ('con1', 2),
                'lo_frequency': qubit_LO,
                'mixer': 'mixer_qubit',
            },
            'intermediate_frequency': qubit_IF,
            'operations': {
                'cw': 'const_pulse',
                'saturation': 'saturation_pulse',
                'gauss': 'gaussian_pulse',
                'pi': 'pi_pulse',
                'pi_half': 'pi_half_pulse',
            },
        },
        'resonator': {
            'mixInputs': {
                'I': ('con1', 3),
                'Q': ('con1', 4),
                'lo_frequency': resonator_LO,
                'mixer': 'mixer_resonator',
            },
            'intermediate_frequency': resonator_IF,
            'operations': {
                'cw': 'const_pulse',
                'short_readout': 'short_readout_pulse',
                'readout': 'readout_pulse',
                'long_readout': 'long_readout_pulse',
            },
            'outputs': {
                'out1': ('con1', 1),
                'out2': ('con1', 2),
            },
            'time_of_flight': time_of_flight,
            'smearing': 0,
        },
    },
    'pulses': {
        'const_pulse': {
            'operation': 'control',
            'length': const_len,
            'waveforms': {
                'I': 'const_wf',
                'Q': 'zero_wf',
            },
        },
        'saturation_pulse': {
            'operation': 'control',
            'length': saturation_len,
            'waveforms': {
                'I': 'saturation_drive_wf',
                'Q': 'zero_wf'
            }
        },
        'gaussian_pulse': {
            'operation': 'control',
            'length': gauss_len,
            'waveforms': {
                'I': 'gauss_wf',
                'Q': 'zero_wf',
            },
        },
        'pi_pulse': {
            'operation': 'control',
            'length': pi_len,
            'waveforms': {
                'I': 'pi_wf',
                'Q': 'pi_drag_wf',
            },
        },
        'pi_half_pulse': {
            'operation': 'control',
            'length': pi_half_len,
            'waveforms': {
                'I': 'pi_half_wf',
                'Q': 'pi_drag_half_wf',
            },
        },
        'short_readout_pulse': {
            'operation': 'measurement',
            'length': short_readout_len,
            'waveforms': {
                'I': 'short_readout_wf',
                'Q': 'zero_wf',
            },
            'integration_weights': {
                'cos': 'short_cosine_weights',
                'sin': 'short_sine_weights',
                'minus_sin': 'short_minus_sine_weights',
            },
            'digital_marker': 'ON',
        },
        'readout_pulse': {
            'operation': 'measurement',
            'length': readout_len,
            'waveforms': {
                'I': 'readout_wf',
                'Q': 'zero_wf',
            },
            'integration_weights': {
                'cos': 'cosine_weights',
                'sin': 'sine_weights',
                'minus_sin': 'minus_sine_weights',
            },
            'digital_marker': 'ON',
        },
        'long_readout_pulse': {
            'operation': 'measurement',
            'length': long_readout_len,
            'waveforms': {
                'I': 'long_readout_wf',
                'Q': 'zero_wf',
            },
            'integration_weights': {
                'cos': 'long_cosine_weights',
                'sin': 'long_sine_weights',
                'minus_sin': 'long_minus_sine_weights',
            },
            'digital_marker': 'ON',
        },
    },

    'waveforms': {
        'const_wf': {'type': 'constant', 'sample': const_amp},
        'saturation_drive_wf': {'type': 'constant', 'sample': saturation_amp},
        'zero_wf': {'type': 'constant', 'sample': 0.0},
        'gauss_wf': {'type': 'arbitrary', 'samples': gauss_wf.tolist()},
        'pi_wf': {'type': 'arbitrary', 'samples': pi_wf.tolist()},
        'pi_drag_wf': {'type': 'arbitrary', 'samples': pi_drag_wf.tolist()},
        'pi_half_wf': {'type': 'arbitrary', 'samples': pi_half_wf.tolist()},
        'pi_half_drag_wf': {'type': 'arbitrary', 'samples': pi_half_drag_wf.tolist()},
        'short_readout_wf': {'type': 'constant', 'sample': short_readout_amp},
        'readout_wf': {'type': 'constant', 'sample': readout_amp},
        'long_readout_wf': {'type': 'constant', 'sample': long_readout_amp},
    },

    'digital_waveforms': {
        'ON': {'samples': [(1, 0)]},
    },

    'integration_weights': {
        'short_cosine_weights': {
            'cosine': [(1.0, short_readout_len)],
            'sine': [(0.0, short_readout_len)],
        },
        'short_sine_weights': {
            'cosine': [(0.0, short_readout_len)],
            'sine': [(1.0, short_readout_len)],
        },
        'short_minus_sine_weights': {
            'cosine': [(0.0, short_readout_len)],
            'sine': [(-1.0, short_readout_len)],
        },
        'cosine_weights': {
            'cosine': [(1.0, readout_len)],
            'sine': [(0.0, readout_len)],
        },
        'sine_weights': {
            'cosine': [(0.0, readout_len)],
            'sine': [(1.0, readout_len)],
        },
        'minus_sine_weights': {
            'cosine': [(0.0, readout_len)],
            'sine': [(-1.0, readout_len)],
        },
        'long_cosine_weights': {
            'cosine': [(1.0, long_readout_len)],
            'sine': [(0.0, long_readout_len)],
        },
        'long_sine_weights': {
            'cosine': [(0.0, long_readout_len)],
            'sine': [(1.0, long_readout_len)],
        },
        'long_minus_sine_weights': {
            'cosine': [(0.0, long_readout_len)],
            'sine': [(-1.0, long_readout_len)],
        },
    },
    'mixers': {
        'mixer_qubit': [{'intermediate_frequency': qubit_IF, 'lo_frequency': qubit_LO,
                         'correction': IQ_imbalance(mixer_qubit_g, mixer_qubit_phi)}],
        'mixer_resonator': [{'intermediate_frequency': resonator_IF, 'lo_frequency': resonator_LO,
                             'correction': IQ_imbalance(mixer_resonator_g, mixer_resonator_phi)}],
    },
}
