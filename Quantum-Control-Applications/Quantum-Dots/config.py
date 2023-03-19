import numpy as np

from qualang_tools.units import unit

#######################
# AUXILIARY FUNCTIONS #
#######################


#############
# VARIABLES #
#############

u = unit()
host = '129.67.84.123'

measurement_length = 20480  # clock cycles (4 ns)
rf_frequency = 116e6

## waveform parameters

const_amplitude = 0.2

ramp_wf_start = 0.
ramp_wf_end = 0.5
ramp_wf_samples = 100
ramp_waveform = np.linspace(ramp_wf_start, ramp_wf_end, ramp_wf_samples)

measure_amplitude = 0.05

top_hat_low = 0.
top_hat_high = 0.5
top_hat_waveform = [top_hat_low] * 1 + [top_hat_high] * 14 + [top_hat_low] * 1

transition_measure_full_length = 512

transition_measure_sliced_length = 10000

# non-sticky waveforms

waveform_length = 10000
non_sticky_top_hat_amp = 0.5

top_hat_non_sticky_0_samples = np.full(waveform_length, fill_value=non_sticky_top_hat_amp).tolist()
top_hat_non_sticky_1_samples = [0.] + np.full(waveform_length - 1, fill_value=non_sticky_top_hat_amp).tolist()
top_hat_non_sticky_2_samples = [0., 0.] + np.full(waveform_length - 2, fill_value=non_sticky_top_hat_amp).tolist()
top_hat_non_sticky_3_samples = [0., 0., 0.] + np.full(waveform_length - 3, fill_value=non_sticky_top_hat_amp).tolist()

top_hat_non_sticky_0_length = len(top_hat_non_sticky_0_samples)
top_hat_non_sticky_1_length = len(top_hat_non_sticky_1_samples)
top_hat_non_sticky_2_length = len(top_hat_non_sticky_2_samples)
top_hat_non_sticky_3_length = len(top_hat_non_sticky_3_samples)

ramp_pulse_length = 100
constant_pulse_length = 16
jump_pulse_length = 16
top_hat_pulse_length = 16
top_hat_non_sticky_length = 2000


##########
# CONFIG #
##########

config = {
    'version': 1,
    'controllers': {
        'con1': {
            'analog_outputs': {
                1: {'offset': 0.0, 'filter': {'feedforward': [-1]}},
                2: {'offset': 0.0, 'filter': {'feedforward': [-1]}},
                3: {'offset': 0.0, 'filter': {'feedforward': [-1]}},  # dot3
                4: {'offset': 0.0, 'filter': {'feedforward': [-1]}},  # charge_sensor gate
                5: {'offset': 0.0},  # rf_in
                6: {'offset': 0.0, 'filter': {'feedforward': [-1]}},
            },
            'digital_outputs': {},
            'analog_inputs': {1: {'offset': 0.0}, 2: {'offset': 0.0}},
        }
    },
    'elements': {
        'dac12': {
            'singleInput': {'port': ('con1', 3)},
            'hold_offset': {'duration': 200},
            'operations': {
                'constant': 'constant',
                'top_hat': 'top_hat',
                'ramp': 'ramp'
            },
        },
        'dac14': {
            'singleInput': {'port': ('con1', 4)},
            'hold_offset': {'duration': 200},
            'operations': {
                'constant': 'constant',
                'top_hat': 'top_hat',
                'ramp': 'ramp'
            },
        },
        'charge_sensor_ohmic': {
            'singleInput': {'port': ('con1', 1)},
            'time_of_flight': 236,
            'smearing': 0,
            'intermediate_frequency': rf_frequency,
            'outputs': {'out1': ('con1', 1)},
            'operations': {
                'measure': 'measure',
                'transition_measure_full': 'transition_measure_full',
                'transition_measure_sliced': 'transition_measure_sliced'
            },
        },

    },
    'pulses': {
        'ramp': {
            'operation': 'control',
            'length': ramp_pulse_length,
            'waveforms': {
                'single': 'ramp'
            }
        },

        'constant': {
            'operation': 'control',
            'length': constant_pulse_length,
            'waveforms': {'single': 'constant'},
        },

        'jump': {
            'operation': 'control',
            'length': jump_pulse_length,
            'waveforms': {'single': 'constant'},
        },

        'top_hat': {
            'operation': 'control',
            'length': top_hat_pulse_length,
            'waveforms': {'single': 'top_hat'},
        },

        'top_hat_non_sticky': {
            'operation': 'control',
            'length': top_hat_non_sticky_length,
            'waveforms': {'single': 'constant'},
        },

        'top_hat_non_sticky_0': {
            'operation': 'control',
            'length': top_hat_non_sticky_0_length,
            'waveforms': {'single': 'top_hat_non_sticky_0'},
        },

        'top_hat_non_sticky_1': {
            'operation': 'control',
            'length': top_hat_non_sticky_1_length,
            'waveforms': {'single': 'top_hat_non_sticky_1'},
        },

        'top_hat_non_sticky_2': {
            'operation': 'control',
            'length': top_hat_non_sticky_2_length,
            'waveforms': {'single': 'top_hat_non_sticky_2'},
        },

        'top_hat_non_sticky_3': {
            'operation': 'control',
            'length': top_hat_non_sticky_3_length,
            'waveforms': {'single': 'top_hat_non_sticky_3'},
        },

        'measure': {
            'operation': 'measurement',
            'length': measurement_length,
            'waveforms': {'single': 'measure'},
            'digital_marker': 'ON',
            'integration_weights': {
                'integW1': 'measure_I',
                'integW2': 'measure_Q',
            },
        },

        'transition_measure_full': {
            'operation': 'measurement',
            'length': transition_measure_full_length,
            'waveforms': {'single': 'measure'},
            'digital_marker': 'ON',
            'integration_weights': {
                'integW1': 'transition_full_I',
                'integW2': 'transition_full_Q',
            },
        },

        'transition_measure_sliced': {
            'operation': 'measurement',
            'length': transition_measure_sliced_length,
            'waveforms': {'single': 'measure'},
            'digital_marker': 'ON',
            'integration_weights': {
                'integW1': 'transition_sliced_I',
                'integW2': 'transition_sliced_Q',
            },
        }
    },
    'waveforms': {
        'constant': {'type': 'constant', 'sample': const_amplitude},
        'ramp': {'type': 'arbitrary', 'samples': ramp_waveform},
        'measure': {'type': 'constant', 'sample': measure_amplitude},
        'top_hat': {'type': 'arbitrary', 'samples': top_hat_waveform},

        'top_hat_non_sticky_0': {'type': 'arbitrary', 'samples': top_hat_non_sticky_0_samples},
        'top_hat_non_sticky_1': {'type': 'arbitrary', 'samples': top_hat_non_sticky_1_samples},
        'top_hat_non_sticky_2': {'type': 'arbitrary', 'samples': top_hat_non_sticky_2_samples},
        'top_hat_non_sticky_3': {'type': 'arbitrary', 'samples': top_hat_non_sticky_3_samples}
    },
    'digital_waveforms': {
        "ON": {"samples": [(1, 0)]}
    },
    'integration_weights': {
        "measure_I": {
            "cosine": [(1., measurement_length)],
            "sine": [(0., measurement_length)]
        },
        "measure_Q": {
            "cosine": [(0., measurement_length)],
            "sine": [(1., measurement_length)]
        },
        "transition_full_I": {
            "cosine": [(1., transition_measure_full_length)],
            "sine": [(0., transition_measure_full_length)]
        },
        "transition_full_Q": {
            "cosine": [(0., transition_measure_full_length)],
            "sine": [(1., transition_measure_full_length)]
        },
        "transition_sliced_I": {
            "cosine": [(1., transition_measure_sliced_length)],
            "sine": [(0., transition_measure_sliced_length)]
        },
        "transition_sliced_Q": {
            "cosine": [(0., transition_measure_sliced_length)],
            "sine": [(1., transition_measure_sliced_length)]
        },

    },
    'mixers': {}
}
