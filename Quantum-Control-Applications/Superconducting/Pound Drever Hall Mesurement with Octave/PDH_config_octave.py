import numpy as np

######################
# AUXILIARY FUNCTIONS:
######################


def gauss(amplitude, mu, sigma, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    return [float(x) for x in gauss_wave]


def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1-g**2)*(2*c**2-1))
    return [float(N * x) for x in [(1-g)*c, (1+g)*s, (1-g)*s, (1+g)*c]]




################
# CONFIGURATION:
################

qop_ip = '169.254.10.0'
opx_port = 80
octave_port = 50

# generating the mixer calibration matrix from the g and phi parameters
# to this you need to also add the DC offsets which you set in the analog outputs section
# usage is calibration_matrix = IQ_imbalance(g, phi)
readout_mixer = IQ_imbalance(0.02355-0.03, np.pi/2-0.02852*0 - 0.02)

# Resonator parameters
RR_TOF = 192 # 192 this Time of Flight needs to be adjusted to accomodate the propagation delay in the lab
rr_IF = -41.759e6 # initial resonator frequency
rr_LO = int(5.8e9) # resonator LO frequency

# parameters used for other things, not in this PDH code.
# leaving it here for you to use these other pulses if you need
constant_amplitude=0.1
gauss_amplitude = 0.2
noise_amplitude=0.01
long_redout_len = 3600
readout_len = 100
single_gauss_len = 100
#simulated_pound_error_signal_length = 1000


# PDH pulse config
pound_pulse_length=60000 # length of the PDH pulse
t = np.linspace(0, pound_pulse_length-1, pound_pulse_length) #[t*1e-9 for t in range(pound_pulse_length)]
pound_modulation_freq = 20e6 # sideband frequency in Hz
pound_modulation_amplitude = 0.1 # PDH modulation depth
pound_phase = 2*np.pi*0.3  #0.26 for 25MHz sideband no filter

# Generating the I and Q signals for the PDH modulation
pound_I_wf = pound_modulation_amplitude*np.ones(pound_pulse_length)
pound_Q_wf_vec = 0.5*(pound_modulation_amplitude)*np.sin(2*np.pi*pound_modulation_freq*t*1e-9+pound_phase)
pound_Q_wf = [float(x) for x in pound_Q_wf_vec]


# These parameters are used in the old PDH demod way (with integration weights) but then you can't do pulses
# longer than about 8us.. so it's not used really
pound_pulse_length_IW = 10000
pound_demod_phase = 0.0 #-2*np.pi*0.26
N_zeros_at_first = 200 # in clock cycles
# Generating the integration weights for the demodulation, used only in the old PDH way
t_demod = np.linspace(0,pound_pulse_length_IW-4, int(pound_pulse_length_IW/4))
pound_demod_vec = np.cos(2*np.pi*pound_modulation_freq*t_demod*1e-9+pound_demod_phase)
pound_demod_cos = [float(x) for x in pound_demod_vec]
pound_demod_cos[0:N_zeros_at_first]=np.zeros(N_zeros_at_first)
#pound_demod_cos = [1.0]*int(pound_pulse_length/4)
pound_demod_sin = [0.0]*int(pound_pulse_length_IW/4)

controller='con1'

config = {

    'version': 1,

    'controllers': {
        controller: {
            'type': 'opx1',
            'analog_outputs': {
                1: {'offset': 0.0},  #
                2: {'offset': 0.0},  #

                3: {'offset': 0.0},  # RR I
                4: {'offset': 0.0},  # RR Q
                5: {'offset': 0.0}, # dummy for PDH demod - this is needed to get the raw adc data
                6: {'offset': 0.0}, # dummy for PDH demod 2
                9: {'offset': 0.0}, # Pound simulator channel 1
                10: {'offset': 0.0}, # Pound simulator channel 2


            },
            'digital_outputs': {
                1: {},
                2: {},
                3: {},
                4: {},
            },
            'analog_inputs': {
                1: {'offset': 0.0, 'gain_db': 20},# adding 20dB gain in the OPX and DC offset
                2: {'offset': 0.0, 'gain_db': 0}
            }
        }
    },

    'elements': {

        # Resonator element
        'RR': {
            'mixInputs': {
                'I': (controller, 1),
                'Q': (controller, 2),
                'lo_frequency': rr_LO,
                'mixer': 'octave_octave1_1' #'mixer_RR'
            },
            'intermediate_frequency': rr_IF,
            'operations': {
                'CW': 'CW',
                'long_readout': 'long_readout_pulse',
                'readout': 'readout_pulse',
                'gauss': 'gauss_pulse',
                'pound_pulse': 'pound_pulse'
            },
            "digitalInputs": {
                'switch': {
                    'port': (controller, 1),
                    'delay': 144,
                    'buffer': 0,
                }
            },
            "outputs": {
                'out1': (controller, 1),
                'out2': (controller, 2)
            },
            'time_of_flight': RR_TOF,
            'smearing': 0
        },

        # 'Pound_simulator': {
        #     'mixInputs': {
        #         'I': (controller, 9),
        #         'Q': (controller, 10),
        #     },
        #     # "singleInput": {
        #     #      "port": (controller, 10),
        #     #  },
        #     'intermediate_frequency': pound_modulation_freq+100000,
        #     'operations': {
        #         'simulated_pound_error_signal': 'simulated_pound_error_signal'
        #     },
        #     "digitalInputs": {
        #         'switch': {
        #             'port': (controller, 3),
        #             'delay': 144,
        #             'buffer': 0,
        #         }
        #     },
        #     # "outputs": {
        #     #     'out1': (controller, 1),
        #     # },
        #     # 'time_of_flight': RR_TOF,
        #     # 'smearing': 0
        #
        # },

        'Pound_demod': {
             "singleInput": {
                 "port": (controller, 5),
             },
            'intermediate_frequency': pound_modulation_freq,
            'operations': {
                'pound_demod_pulse': 'pound_demod_pulse'
            },
            "digitalInputs": {
                'switch': {
                    'port': (controller, 3),
                    'delay': 144,
                    'buffer': 0,
                }
            },
            "outputs": {
                'out1': (controller, 1),
                'out2': (controller, 2),
            },
            'time_of_flight': RR_TOF,
            'smearing': 0

        },
        'Pound_demod_2': {
             "singleInput": {
                 "port": (controller, 6),
             },
            'intermediate_frequency': pound_modulation_freq,
            "digitalInputs": {
                'switch': {
                    'port': (controller, 4),
                    'delay': 144,
                    'buffer': 0,
                }
            },
            'operations': {
                'pound_demod_pulse': 'pound_demod_pulse'
            },
            "outputs": {
                'out1': (controller, 1),
                'out2': (controller, 2),
            },
            'time_of_flight': RR_TOF,
            'smearing': 0

        },

    },

    "pulses": {

        'simulated_pound_error_signal': {
            'operation': 'control',
            'length': pound_pulse_length,
            'waveforms': {
                'I': 'const_wf',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON'

        },

        'pound_demod_pulse': {
            'operation': 'measurement',
            'length': pound_pulse_length,
            'waveforms': {
                "single": "zero_wf",
            },
            'integration_weights': {
                'integ_pound': 'integ_pound',
            },
            'digital_marker': 'ON'
        },

        "noise_CW": {
            'operation': 'control',
            'length': 100,
            'waveforms': {
                'I': 'const_wf',
                'Q': 'zero_wf'
            }
        },

        "CW": {
            'operation': 'control',
            'length': 100,
            'waveforms': {
                'I': 'const_wf',
                'Q': 'zero_wf'
            }
        },

        "saturation_pulse": {
            'operation': 'control',
            'length': 20000,  # several T1s
            'waveforms': {
                'I': 'saturation_wf',
                'Q': 'zero_wf'
            }
        },


        'pi_pulse': {
            'operation': 'control',
            'length': 60,
            'waveforms': {
                'I': 'pi_wf',
                'Q': 'zero_wf'
            }
        },

        'pi2_pulse': {
            'operation': 'control',
            'length': 60,
            'waveforms': {
                'I': 'pi2_wf',
                'Q': 'zero_wf'
            }
        },

        'minus_pi2_pulse': {
            'operation': 'control',
            'length': 60,
            'waveforms': {
                'I': 'minus_pi2_wf',
                'Q': 'zero_wf'
            }
        },

        'long_readout_pulse': {
            'operation': 'measurement',
            'length': long_redout_len,
            'waveforms': {
                'I': 'long_readout_wf',
                'Q': 'zero_wf'
            },
            'integration_weights': {
                'long_integW1': 'long_integW1',
                'long_integW2': 'long_integW2',
            },
            'digital_marker': 'ON'
        },

        'readout_pulse': {
            'operation': 'measurement',
            'length': readout_len,
            'waveforms': {
                'I': 'readout_wf',
                'Q': 'zero_wf'
            },
            'integration_weights': {
                'integW1': 'integW1',
                'integW2': 'integW2',
                'optW1': 'optW1',
                'optW2': 'optW2',
                'integW_cos': 'integW1_cos',
                'integW_sin': 'integW1_sin',
                'integW_minus_sin': 'integW1_minus_sin',
                #'integW_minus_cos': 'integW_minus_cos'
            },
            'digital_marker': 'ON'
        },
        'gauss_single_pulse': {
            'operation': 'measurement',
            'length': single_gauss_len,
            'waveforms': {
                "single": "single_gauss_wf",
            },
            'integration_weights': {
                'integ_gauss': 'integ_gauss'
            },
            'digital_marker': 'ON'

        },

        'gauss_pulse': {
            'operation': 'measurement',
            'length': readout_len,
            'waveforms': {
                'I': 'gauss_wf',
                'Q': 'zero_wf'
            },
            'integration_weights': {
                'integW1': 'integW1',
                'integW2': 'integW2',
                'optW1': 'optW1',
                'optW2': 'optW2',
                'integW_cos': 'integW1_cos',
                'integW_sin': 'integW1_sin',
                'integW_minus_sin': 'integW1_minus_sin',
                'integ_gauss': 'integ_gauss',
                #'integW_minus_cos': 'integW_minus_cos'
            },
            'digital_marker': 'ON'
        },
        'pound_pulse': {
            'operation': 'measurement',
            'length': pound_pulse_length,
            'waveforms': {
                'I': 'pound_I_wf',
                'Q': 'pound_Q_wf'
            },
            'integration_weights': {
                'integW1': 'integW1',
                'integW2': 'integW2',
                'optW1': 'optW1',
                'optW2': 'optW2',
                'integW_cos': 'integW1_cos',
                'integW_sin': 'integW1_sin',
                'integW_minus_sin': 'integW1_minus_sin',
                'integ_gauss': 'integ_gauss',
                'pound_demod': 'pound_demod',
            },
            'digital_marker': 'ON'

        }

    },

    'waveforms': {

        'simulated_pound_error_wf': {
            'type': 'constant',
            'sample': 0.2
        },
        'const_wf': {
            'type': 'constant',
            'sample': constant_amplitude
        },

        'zero_wf': {
            'type': 'constant',
            'sample': 0.0
        },

        'saturation_wf': {
            'type': 'constant',
            'sample': 0.211
        },

        'gauss_wf': {
            'type': 'arbitrary',
            'samples': gauss(gauss_amplitude, 0.0, int(readout_len/4), readout_len) #gauss(amplitude, mu, sigma, length):
        },
        'pound_I_wf': {
            'type': 'arbitrary',
            'samples': pound_I_wf
        },
        'pound_Q_wf': {
            'type': 'arbitrary',
            'samples': pound_Q_wf
        },
        'single_gauss_wf': {
            'type': 'arbitrary',
            'samples': gauss(gauss_amplitude, 0.0, 15.0, single_gauss_len) #gauss(amplitude, mu, sigma, length):
        },

        'pi_wf': {
            'type': 'arbitrary',
            'samples': gauss(0.3, 0.0, 6.0, 60)
        },

        'pi2_wf': {
            'type': 'arbitrary',
            'samples': gauss(0.15, 0.0, 6.0, 60)
        },

        'minus_pi2_wf': {
            'type': 'arbitrary',
            'samples': gauss(-0.15, 0.0, 6.0, 60)
        },

        'long_readout_wf': {
            'type': 'constant',
            'sample': constant_amplitude
        },

        'readout_wf': {
            'type': 'constant',
            'sample': constant_amplitude
        },
    },

    'digital_waveforms': {
        'ON': {
            'samples': [(1, 0)]
        }
    },

    'integration_weights': {
        'pound_demod': {
            'cosine': pound_demod_cos,
            'sine': pound_demod_sin
        },

        'integ_gauss': {
            'cosine': [1.0]*int(single_gauss_len/4),
            'sine': [0.0]*int(single_gauss_len/4)
        },

        'long_integW1': {
            'cosine': [1.0] * int(long_redout_len / 4),
            'sine': [0.0] * int(long_redout_len / 4)
        },

        'long_integW2': {
            'cosine': [0.0] * int(long_redout_len / 4),
            'sine': [1.0] * int(long_redout_len / 4)
        },

        'integW1': {
            'cosine': [1.0] * int(readout_len / 4),
            'sine': [0.0] * int(readout_len / 4),
        },

        'integW2': {
            'cosine': [0.0] * int(readout_len / 4),
            'sine': [1.0] * int(readout_len / 4),
        },

        'optW1': {
            'cosine': [1.0] * int(readout_len / 4),
            'sine': [0.0] * int(readout_len / 4)
        },

        'optW2': {
            'cosine': [0.0] * int(readout_len / 4),
            'sine': [1.0] * int(readout_len / 4)
        },

        "integW1_cos": {
            "cosine": [1.0] * int(readout_len / 4),
            "sine": [0.0] * int(readout_len / 4),
        },
        "integW1_sin": {
            "cosine": [0.0] * int(readout_len/4),
            "sine": [1.0] * int(readout_len/4),
        },
        "integW1_minus_sin": {
            "cosine": [0.0] * int(readout_len/4),
            "sine": [-1.0] * int(readout_len/4),
        },
        "integW2_cos": {
            "cosine": [1.0] * int(readout_len/4),
            "sine": [0.0] * int(readout_len/4),
        },
        "integW2_sin": {
            "cosine": [0.0] * int(readout_len/4),
            "sine": [1.0] * int(readout_len/4),
        },
        "integW2_minus_sin": {
            "cosine": [0.0] * int(readout_len/4),
            "sine": [-1.0] * int(readout_len/4),
        },
        'integ_pound': {
            "cosine": [1.0] * int(pound_pulse_length/4),
            "sine": [0.0] * int(pound_pulse_length/4),
        }
    },

    'mixers': {

        'octave_octave1_1': [
            {'intermediate_frequency': rr_IF, 'lo_frequency': rr_LO,
             'correction': [1,0,0,1]}
        ],

    }
}

