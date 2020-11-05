import numpy as np
import matplotlib.pyplot as plt

#############################
# simulation helpers #
#############################


def simulate_pulse(IF_freq, chi, k, Ts, Td, power):
    I = [0]
    Q = [0]

    for t in range(Ts):
        I.append(I[-1]+(power/2 - k*I[-1]+Q[-1]*chi))
        Q.append(Q[-1]+(power/2 - k*Q[-1]-I[-1]*chi))

    for t in range(Td - 1):
        I.append(I[-1]+(-k*I[-1]+Q[-1]*chi))
        Q.append(Q[-1]+(-k*Q[-1]-I[-1]*chi))

    I = np.array(I)
    Q = np.array(Q)
    t = np.arange(len(I))

    S = I * np.cos(2 * np.pi * IF_freq * t * 1e-9) + Q * np.sin(2 * np.pi * IF_freq * t * 1e-9)

    return t, I, Q, S


resonators_lo = 7.1e9  # High Band Pass
WG1_lo = resonators_lo
WG2_lo = resonators_lo
q1a_res_IF = 40e6
q2a_res_IF = 41e6

# readout_len = 480
# IF_freq = q1a_res_IF
# Ts = readout_len-200
# Td = 200
# power = 0.2*2
# k = 0.04/3
# chi = 0.023*10
# [tg_, Ig_, Qg_, Sg_] = simulate_pulse(IF_freq, -1 * chi, k, Ts, Td, power)
# [te_, Ie_, Qe_, Se_] = simulate_pulse(IF_freq, 1 * chi, k, Ts, Td, power)
# [tf_, If_, Qf_, Sf_] = simulate_pulse(IF_freq, 2 * chi, k, Ts, Td, power)

# readout_len = 480
# IF_freq = q1a_res_IF
# Ts = readout_len-200
# Td = 200
# power = 0.2*2
# k = 0.04/3
# chi = 0.023*20
# [tg_, Ig_, Qg_, Sg_] = simulate_pulse(IF_freq, -1 * chi, k, Ts, Td, power)
# [te_, Ie_, Qe_, Se_] = simulate_pulse(IF_freq, 1 * chi, k, Ts, Td, power)
# [tf_, If_, Qf_, Sf_] = simulate_pulse(IF_freq, 2 * chi, k, Ts, Td, power)

readout_len = 480
IF_freq = q1a_res_IF
Ts = readout_len-200
Td = 200
power = 0.2
k = 0.04
chi = 0.023
[tg_, Ig_, Qg_, Sg_] = simulate_pulse(IF_freq, -1 * chi, k, Ts, Td, power)
[te_, Ie_, Qe_, Se_] = simulate_pulse(IF_freq, 1 * chi, k, Ts, Td, power)
[tf_, If_, Qf_, Sf_] = simulate_pulse(IF_freq, 5 * chi, k, Ts, Td, power)

plt.figure()
plt.plot(Ig_, Qg_)
plt.plot(Ie_, Qe_)
plt.plot(If_, Qf_)

plt.figure()
plt.plot(Ig_)
plt.plot(Qg_)
plt.figure()
plt.plot(Ie_)
plt.plot(Qe_)
plt.plot(If_)
plt.plot(Qf_)
plt.figure()
plt.plot(np.abs(Ig_+1j*Qg_))
plt.plot(np.abs(Ie_+1j*Qe_))
plt.plot(np.abs(If_+1j*Qf_))
divide_signal_factor = 100

config = {

    'version': 1,

    'controllers': {

        'con1': {
            'type': 'opx1',
            'analog_outputs': {
                1: {'offset': 0},
                2: {'offset': 0},
            },
            'digital_outputs': {
                1: {},
            },
            'analog_inputs': {
                1: {'offset': 0},
                2: {'offset': 0},
            }
        },
    },

    'elements': {

        # readout resonators:
        'rr1a': {
            'mixInputs': {
                'I': ('con1', 1),
                'Q': ('con1', 2),
                'lo_frequency': WG1_lo,
                'mixer': 'mixer_WG1',
            },
            'intermediate_frequency': q1a_res_IF,
            'operations': {
                'readout': 'readout_pulse_q1a',
                'readout_pulse_g': 'readout_pulse_g',
                'readout_pulse_e': 'readout_pulse_e',
                'readout_pulse_f': 'readout_pulse_f'
            },
            'outputs': {
                'out1': ('con1', 1),
                'out2': ('con1', 2),
            },
            'time_of_flight': 188,
            'smearing': 0,
        },

        'rr2a': {
            'mixInputs': {
                'I': ('con1', 1),
                'Q': ('con1', 2),
                'lo_frequency': WG2_lo,
                'mixer': 'mixer_WG2',
            },
            'intermediate_frequency': q2a_res_IF,
            'operations': {
                'readout': 'readout_pulse_q1a',
                'readout_pulse_g': 'readout_pulse_g',
                'readout_pulse_e': 'readout_pulse_e',
                'readout_pulse_f': 'readout_pulse_f'
            },
            'outputs': {
                'out1': ('con1', 1),
                'out2': ('con1', 2),
            },
            'time_of_flight': 188,
            'smearing': 0,
        },



    },

    'pulses': {

        'readout_pulse_q1a': {
            'operation': 'measurement',
            'length': readout_len,
            'waveforms': {
                'I': 'ro_wf_q1a',
                'Q': 'zero_wf'
            },
            'integration_weights': {
                'integW_cos': 'integW_cos',
                'integW_sin': 'integW_sin'
            },
            'digital_marker': 'ON'
        },

        'readout_pulse_g': {
            'operation': 'measurement',
            'length': readout_len,
            'waveforms': {
                'I': 'Ig_wf',
                'Q': 'Qg_wf'
            },
            'integration_weights': {
                'integW_cos': 'integW_cos',
                'integW_sin': 'integW_sin'
            },
            'digital_marker': 'ON'
        },

        'readout_pulse_e': {
            'operation': 'measurement',
            'length': readout_len,
            'waveforms': {
                'I': 'Ie_wf',
                'Q': 'Qe_wf'
            },
            'integration_weights': {
                'integW_cos': 'integW_cos',
                'integW_sin': 'integW_sin'
            },
            'digital_marker': 'ON'
        },

        'readout_pulse_f': {
            'operation': 'measurement',
            'length': readout_len,
            'waveforms': {
                'I': 'If_wf',
                'Q': 'Qf_wf'
            },
            'integration_weights': {
                'integW_cos': 'integW_cos',
                'integW_sin': 'integW_sin'
            },
            'digital_marker': 'ON'
        },

    },

    'waveforms': {

        'zero_wf': {
            'type': 'constant',
            'sample': 0.0
        },

        'ro_wf_q1a': {
            'type': 'constant',
            'sample': 0.11
        },

        'Ig_wf': {
            'type': 'arbitrary',
            'samples': [float(arg / divide_signal_factor) for arg in Ig_]
        },

        'Qg_wf': {
            'type': 'arbitrary',
            'samples': [float(arg / divide_signal_factor) for arg in Qg_]
        },

        'Ie_wf': {
            'type': 'arbitrary',
            'samples': [float(arg / divide_signal_factor) for arg in Ie_]
        },

        'Qe_wf': {
            'type': 'arbitrary',
            'samples': [float(arg / divide_signal_factor) for arg in Qe_]
        },

        'If_wf': {
            'type': 'arbitrary',
            'samples': [float(arg / divide_signal_factor) for arg in If_]
        },

        'Qf_wf': {
            'type': 'arbitrary',
            'samples': [float(arg / divide_signal_factor) for arg in Qf_]
        },

    },

    'digital_waveforms': {

        'ON': {
            'samples': [(1, 0)]
        },
    },

    'integration_weights': {

        'integW_cos': {
            'cosine': [1.0] * 120,
            'sine': [0.0] * 120,
        },

        'integW_sin': {
            'cosine': [0.0] * 120,
            'sine': [1.0] * 120,
        },

    },

    'mixers': {
        'mixer_WG1': [
            {'intermediate_frequency': q1a_res_IF, 'lo_frequency': WG1_lo, 'correction': [1, 0, 0, 1]},
        ],

        'mixer_WG2': [
            {'intermediate_frequency': q2a_res_IF, 'lo_frequency': WG2_lo, 'correction': [1, 0, 0, 1]},
        ],
    }
}
