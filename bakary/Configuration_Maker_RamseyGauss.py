import numpy as np
from importlib import reload

def gauss(amplitude, mu, sigma, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    return [float(x) for x in gauss_wave]
    #return [amplitude for x in t]

def IQ_imbalance_corr(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g ** 2) * (2 * c ** 2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s,
                                   (1 - g) * s, (1 + g) * c]]


Ramsey_operations = ["Ramsey_{}ns".format(arg) for arg in range(16)]
Ramsey_pulses = ["Ramsey_pulses_{}ns".format(arg) for arg in range(16)]
Ramsey_waveforms = ['Ramsey_waveforms_{}ns'.format(arg) for arg in range(16)]
Ramsey_markers = ['Ramsey_markers_{}ns'.format(arg) for arg in range(16)]



def configMakerRamseyGauss(Resonator_TOF = 332, Readout_pulse_length = 500, Load_pulse_length = 172,
                      Resonator_freq = 6.6067e9, Drive_freq = 5e9, Tpihalf = 32):

    flag_digital = 1 # set to 1 if use 'adc', 0 otherwise

    Resonator_IF = 50e6
    Resonator_LO = Resonator_freq-Resonator_IF
    DriftCorrection_IF = 70e6
    Readout_drift_pulse_length = 5000

    Drive_IF = 31.25e6 * 0.5
    #Drive_IF = 0
    Drive_LO = Drive_freq - Drive_IF

    Readout_Amp = 0.1 #meas pulse amplitude
    Loadpulse_Amp = 0.3 #prepulse to fast load cavity /!\ <0.5 V
    DriftCorrection_Amp = 0.2 #amplitude of the drift correction pulse


    Ramsey_pulse_length = Tpihalf + 16 #pulses of length 48ns if we take Tpihalf=16
    Drive_gauss_pulse_length = Tpihalf
    Drive_Amp = 0.1
    gauss_drive_amp = 0.1
    gauss_drive_mu = 0
    gauss_drive_sigma = Drive_gauss_pulse_length/6

    Resonator_I0 = -0.01496987
    Resonator_Q0 = -0.0087747
    Resonator_g = 0.0321445
    Resonator_phi = -0.39511618  # parameters resonator at 6.6076 GHz

    #Drive_I0 = -0.01281131 # parameters at 5 GHz with IF = 50MHz with VattDrive=1, gauss_drive_amp = 0.1 and the Agile amp after Vatt and after IQmixer.
    #Drive_Q0 = -0.01503331
    #Drive_g = 0.16567664
    #Drive_phi = 0.4414121

    Drive_I0 = -0.01305361  # parameters at 5 GHz with IF = 31.25MHz with VattDrive=0, gauss_drive_amp = 0.1 and the Agile amp after Vatt and after IQmixer.
    Drive_Q0 = -0.01468115
    Drive_g = 0.08477777
    Drive_phi = 0.31050767

    Drive_I0 = 0  # parameters at 5 GHz with IF = 31.25MHz with VattDrive=0, gauss_drive_amp = 0.1 and the Agile amp after Vatt and after IQmixer.
    Drive_Q0 = 0
    Drive_g = 0
    Drive_phi = -np.pi/2

    Resonator_correction_matrix = IQ_imbalance_corr(Resonator_g, Resonator_phi)
    Drive_correction_matrix = IQ_imbalance_corr(Drive_g, Drive_phi)

    Input1_offset = 0.0910490302734375
    Input2_offset = 0.0786689296875



    config = {
        'version': 1,

        'controllers': {
            'con1': {
                'type': 'opx1',
                'analog_outputs': {
                    1: {'offset': Resonator_I0}, # Resonator I
                    2: {'offset': Resonator_Q0}, # Resonator Q
                    3: {'offset': Drive_I0}, # Drive I
                    4: {'offset': Drive_Q0}, # Drive Q
                    5: {'offset': 0}, # Drive LO amplitude modulation ---------------> SHOULD BE A DIGITAL OUTPUT
                },
                'digital_outputs': {
                    1: {}, # Resonator digital marker
                    2: {}, # Drive digital marker
                    3: {}, # Drive LO synthesizer trigger
                    4: {}, # Drive LO microwave source trigger --------------------------
                },
                'analog_inputs': {
                    1: {'offset': Input1_offset},  # I readout from resonator demod
                    2: {'offset': Input2_offset},  # Q readout from resonator demod
                },
            },
        },

        'elements': {
            'Resonator': {  # Resonator element
                'mixInputs': {
                    'I': ('con1', 1),
                    'Q': ('con1', 2),
                    'lo_frequency': Resonator_LO,
                    'mixer': 'Resonator_mixer',
                },
                'intermediate_frequency': Resonator_IF,
                'operations': {
                    'readout': 'readout_pulse', # play('readout', 'Resonator'),
                    'chargecav': 'load_pulse',
                },
                'digitalInputs': {
                    'switchR': {
                        'port': ('con1', 1),
                        'delay': 144-12+5,
                        'buffer': 0,
                    }
                },
                'outputs': {
                    'out1': ('con1', 1),
                    'out2': ('con1', 2),
                },
                'time_of_flight': Resonator_TOF,
                'smearing': 0,
            },

            'DriftCorrection': {
                'mixInputs': {
                    'I': ('con1', 1),
                    'Q': ('con1', 2),
                    'lo_frequency': Resonator_LO,
                    'mixer': 'DriftCorrection_mixer',
                },
                'intermediate_frequency': DriftCorrection_IF,
                'operations': {
                    'readout_drift': 'readout_drift_pulse',  # play('readout_drift', 'DriftCorrection'),
                },
                'digitalInputs': {
                    'switchR': {
                        'port': ('con1', 1),
                        'delay': 144-12+5,
                        'buffer': 0,
                    }
                },
                'outputs': {
                    'out1': ('con1', 1),
                    'out2': ('con1', 2),
                },
                'time_of_flight': Resonator_TOF,
                'smearing': 0,
            },

            'Drive': {  # Drive element
                'mixInputs': {
                    'I': ('con1', 4),
                    'Q': ('con1', 3),
                    'lo_frequency': Drive_LO,
                    'mixer': 'Drive_mixer',
                },
                'digitalInputs': {
                    'switchE': {
                        'port': ('con1', 3),
                        'delay': 144-12+5,
                        'buffer': 0,
                    }
                },
                'intermediate_frequency': Drive_IF,
                'operations': {
                    'gauss_drive': 'gauss_drive_pulse',
                },
            },

            'Drive2': {  # Resonator element
                'mixInputs': {
                    'I': ('con1', 4),
                    'Q': ('con1', 3),
                    'lo_frequency': Drive_LO,
                    'mixer': 'Drive_mixer',
                },
                'digitalInputs': {
                    'switchE': {
                        'port': ('con1', 3),
                        'delay': 144 - 12+5,
                        'buffer': 0,
                    }
                },
                'intermediate_frequency': Drive_IF,
                'operations': {
                    **{Ramsey_operations[i]: Ramsey_pulses[i] for i in range(16)},
                },
            },

        },

        "pulses": {

            'readout_pulse': {
                'operation': 'measurement',
                'length': Readout_pulse_length,
                'waveforms': {
                    'I': 'readout_wf',
                    'Q': 'zero_wf',
                },
                'integration_weights': {
                    'integW_cos': 'integW_cosine',
                    'integW_sin': 'integW_sine',
                },
                'digital_marker': 'ON', #Put ON instead of Modulate if you want raw adc time traces
            },
            'readout_drift_pulse': {
                'operation': 'measurement',
                'length': Readout_drift_pulse_length,
                'waveforms': {
                    'I': 'driftCorrection_wf',
                    'Q': 'zero_wf',
                },
                'integration_weights': {
                    'integW_cos': 'integW_cosine',
                    'integW_sin': 'integW_sine',
                },
                'digital_marker': 'ON',  # Put ON instead of Modulate if you want raw adc time traces
            },
            'load_pulse': {
                'operation': 'control',
                'length': Load_pulse_length,
                'waveforms': {
                    'I': 'loadpulse_wf',
                    'Q': 'zero_wf',
                },
                'digital_marker': 'ON',
            },

            'gauss_drive_pulse': {
                'operation': 'control',
                'length': Drive_gauss_pulse_length,
                'waveforms': {
                    'I': 'gauss_drive_wf',
                    'Q': 'zero_wf',
                },
                'digital_marker': 'ON',
            },

            **{
                Ramsey_pulses[i]: {
                    'operation': 'control',
                    'length': Ramsey_pulse_length,
                    'waveforms': {
                        'I': Ramsey_waveforms[i],
                        'Q': 'zero_wf',
                    },
                    'digital_marker': Ramsey_markers[i],
                } for i in range(16)
            },

        },

        'waveforms': {

            'readout_wf': {
                'type': 'constant',
                'sample': Readout_Amp,
            },

            'driftCorrection_wf': {
                'type': 'constant',
                'sample': DriftCorrection_Amp,
            },

            'loadpulse_wf': {
                'type': 'constant',
                'sample': Loadpulse_Amp,
            },

            'zero_wf': {
                'type': 'constant',
                'sample': 0.0,
            },

            'drive_wf': {
                'type': 'constant',
                'sample': Drive_Amp,
            },
            'gauss_drive_wf': {
                'type': 'arbitrary',
                'samples': gauss(gauss_drive_amp, gauss_drive_mu, gauss_drive_sigma, Drive_gauss_pulse_length),
            },

            **{
                Ramsey_waveforms[i]: {
                    'type': 'arbitrary',
                    'samples': [0] * (16-i) + gauss(gauss_drive_amp, 0, Tpihalf/6, Tpihalf) + [0] * i,
                } for i in range(16)
            },

        },

        'digital_waveforms': {

            'ON': {
                'samples': [(flag_digital, 0)]  # [(value, length)] 0: until the end of the pulse. Level has to be 1 if use 'adc' to acquire raw adc data acquisition
            },

            'OFF': {
                'samples': [(0, 0)]  # [(value, length)]
            },

            'Modulate': {
                'samples': [(1, Readout_pulse_length / 20), (0, Readout_pulse_length / 20)] * 10  # [(value, length)]
            },

            **{
                Ramsey_markers[i]: {
                    'samples': [(0, 16-i), (1, Tpihalf), (0, i)]
                } for i in range(16)
            },

        },

        'integration_weights': {

            'integW_cosine': {
                'cosine': [1.0] * int(Readout_pulse_length / 4),
                'sine': [0.0] * int(Readout_pulse_length / 4),
            },

            'integW_sine': {
                'cosine': [0.0] * int(Readout_pulse_length / 4),
                'sine': [1.0] * int(Readout_pulse_length / 4),
            },

        },

        'mixers': {
            'Resonator_mixer': [
                {'intermediate_frequency': Resonator_IF,
                 'lo_frequency': Resonator_LO,
                 'correction': Resonator_correction_matrix}
            ],
            'DriftCorrection_mixer': [
                {'intermediate_frequency': DriftCorrection_IF,
                 'lo_frequency': Resonator_LO,
                 'correction': Resonator_correction_matrix}
            ],
            'Drive_mixer': [
                {'intermediate_frequency': Drive_IF,
                 'lo_frequency': Drive_LO,
                 'correction': Drive_correction_matrix}
            ],
        },

    }
    return config