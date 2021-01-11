import numpy as np
from qm.qua import *

π = np.pi
π_2 = np.pi/2

def gauss(amplitude, mu, sigma, delf, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    # Detuning correction Eqn. (4) in Chen et al. PRL, 116, 020501 (2016)
    gauss_wave = gauss_wave * np.exp(2 * np.pi * delf * t)
    return [float(x) for x in gauss_wave]


def gauss_der(amplitude, mu, sigma, delf, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_der_wave = amplitude * (-2 * (t - mu)) * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    # Detuning correction Eqn. (4) in Chen et al. PRL, 116, 020501 (2016)
    gauss_der_wave = gauss_der_wave * np.exp(2 * np.pi * delf * t)
    return [float(x) for x in gauss_der_wave]


# X90 definition

x90amp = 0.4
x90std = 0.2
x90mean = 0
x90duration = 1000
x90detuning = 0
x90waveform = gauss(x90amp, x90mean, x90std, x90detuning, x90duration)  # Assume you have calibration for a X90 pulse
lmda = 0.5  # Define scaling parameter for Drag Scheme
alpha = -1  # Define anharmonicity parameter
x90der_waveform = gauss_der(x90amp, x90mean, x90std, x90detuning, x90duration)

# Y180 definition
y180waveform = gauss(2 * x90amp, x90mean, x90std, x90detuning,
                     x90duration)  # Assume you have calibration for a X90 pulse
y180der_waveform = gauss_der(2 * x90amp, x90mean, x90std, x90detuning, x90duration)

config = {
    'version': 1,
    'controllers': {
        "con1": {
            'type': 'opx1',
            'analog_outputs': {
                1: {'offset': +0.0},  # Control qubit I component
                2: {'offset': +0.0},  # Control qubit Q component
                3: {'offset': +0.0},  # Readout resonator I component
                4: {'offset': +0.0},  # Readout resonator Q component
                5: {'offset': +0.0},  # Tunable coupler I component (flux-tunable transmon)
                6: {'offset': +0.0},  # Tunable coupler Q component

            },
            'analog_inputs': {
                1: {'offset': +0.0},  # Input of the readout resonator coupled to control qubit

            }
        }
    },

    'elements': {
        "control": { #Control qubit, hidden qubit is off the configuration, everything is played indirectly via TC below
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                'lo_frequency': 5.10e7,
                'mixer': 'mixer_qubit'
            },
            'intermediate_frequency': 0,  # 6.19e9,
            'operations': {
                'X90': "X90_pulse",
                'Y90': "Y90_pulse",
                'Y180': "Y180_pulse"
            },
        },
        "RR": { #Readout resonator, connected to control qubit
            'mixInputs': {
                'I': ('con1', 3),
                'Q': ('con1', 4),
                'lo_frequency': 6.00e7,
                'mixer': 'mixer_res'
            },
            'intermediate_frequency': 0,
            'operations': {
                'ReadoutOp': 'readout_pulse',
            },
            'time_of_flight': 180,  # Measurement parameters
            'smearing': 0,
            'outputs': {
                'out1': ('con1', 1)
            }

        },
        "TC": {  #Tunable coupler, connection between control and hidden qubit (used for 2 qubit gates)
            "mixInputs": {
                "I": ("con1", 5),
                "Q": ("con1", 6),
                'lo_frequency': 5.10e7,
                'mixer': 'mixer_TC'
            },
            'intermediate_frequency': 0,  # 5.2122e9,
            'operations': {
                'iSWAP': "X90_pulse",
                'CPHASE': "Y90_pulse",
            },

        }

    },
    "pulses": {
        'readout_pulse': {  # Readout pulse
            'operation': 'measurement',
            'length': 200,
            'waveforms': {
                'I': 'exc_wf',  # Decide what pulse to apply for each component
                'Q': 'zero_wf'
            },
            'integration_weights': {
                'integW1': 'integW1',
                'integW2': 'integW2',
            },
            'digital_marker': 'marker1'
        },

        "X90_pulse": {
            'operation': 'control',
            'length': 1000,
            'waveforms': {
                'I': 'x90_wf',
                'Q': 'x90_der_wf'
            }
        },
        "Y90_pulse": {
            'operation': 'control',
            'length': 1000,
            'waveforms': {
                'I': 'x90_der_wf',
                'Q': 'x90_wf'
            }
        },
        "Y180_pulse": {
            'operation': 'control',
            'length': 1000,
            'waveforms': {
                'I': 'y180_der_wf',
                'Q': 'y180_wf'
            }
        },

    },
    "waveforms": {
        'const_wf': {
            'type': 'constant',
            'sample': 0.2
        },
        'zero_wf': {
            'type': 'constant',
            'sample': 0.
        },
        'x90_wf': {
            'type': 'arbitrary',
            'samples': x90waveform
        },
        'x90_der_wf': {
            'type': 'arbitrary',
            'samples': x90der_waveform
        },

        'y180_wf': {
            'type': 'arbitrary',
            'samples': y180waveform
        },
        'y180_der_wf': {
            'type': 'arbitrary',
            'samples': y180der_waveform
        },
        'exc_wf': {
            'type': 'constant',
            'sample': 0.479
        },
    },
    'digital_waveforms': {
        'marker1': {
            'samples': [(1, 4), (0, 2), (1, 1), (1, 0)]
        }
    },
    'integration_weights': {  # Define integration weights for measurement demodulation
        'integW1': {
            'cosine': [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                       4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                       4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                       4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            'sine': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        'integW2': {
            'cosine': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'sine': [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                     4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                     4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                     4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
        }
    },

    'mixers': {  # Potential corrections to be brought related to the IQ mixing scheme
        'mixer_res': [
            {'intermediate_frequency': 0, 'lo_frequency': 6.00e7, 'correction': [1.0, 0.0, 0.0, 1.0]}
        ],
        'mixer_qubit': [
            {'intermediate_frequency': 0, 'lo_frequency': 5.10e7, 'correction': [1.0, 0.0, 0.0, 1.0]}
        ],
        'mixer_TC': [
            {'intermediate_frequency': 0, 'lo_frequency': 6.00e7, 'correction': [1.0, 0.0, 0.0, 1.0]}
        ],
    }
}


# QUA macros (pulse definition of useful quantum gates)
def Hadamard(tgt):
    U2(tgt, 0, π)


def U2(tgt, 𝜙=0, 𝜆=0):
    Rz(𝜆, tgt)
    Y90(tgt)
    Rz(𝜙, tgt)


def U3(tgt, 𝜃=0, 𝜙=0, 𝜆=0):
    Rz(𝜆 - π / 2, tgt)
    X90(tgt)
    Rz(π - 𝜃, tgt)
    X90(tgt)
    Rz(𝜙 - π / 2, tgt)


def Rz(𝜆, tgt):
    frame_rotation(-𝜆, tgt)


def Rx(𝜆, tgt):
    # U3(tgt, 𝜆, -π / 2, π / 2)
    play("X90_pulse" * amp(𝜆 / (π / 2)), tgt)


def Ry(𝜆, tgt):
    # U3(tgt, 𝜆, 0, 0)
    play("Y90_pulse" * amp(𝜆 / (π / 2)), tgt)


def X90(tgt):
    play('X90', tgt)


def Y90(tgt):
    play('Y90', tgt)


def Y180(tgt):
    play('Y180', tgt)


def process(index,
        tgt):  # QUA macro for applying arbitrary process, here a X rotation of angle π/4, followed by a Y rotation of π/2
    Rx(π / 4, tgt)
    Ry(π / 2, tgt)


def iSWAP(tgt):
    play("iSWAP", tgt)


def CPHASE(tgt):
    play("CPHASE", tgt)


def state_saving(I, Q, state_estimate, stream):  # Do state estimation protocol in QUA, and save the associated state
    # Define coef a & b defining the line separating states 0 & 1 in the IQ Plane (calibration required), here a & b are arbitrary
    a = declare(fixed, value=IO1)
    b = declare(fixed, value=IO2)
    with if_(Q - a * I - b > 0):
        assign(state_estimate, 1)
    with else_():
        assign(state_estimate, 0)
    save(state_estimate, stream)


def measure_and_reset_state(tgt, RR, I, Q, A, stream_A): #to be redefined
    measure('meas_pulse', RR, None, ('integW1', I), ('integW2', Q))
    state_saving(I, Q, A, stream_A) #To be redefined to match the joint qubit readout
    with if_(A == 1):
        play("X90_pulse" * amp(2), tgt)

def play_pulse(pulse):
    if pulse == "Rx_π_2":
        X90("control")
    elif pulse == "Ry_π_2":
        Y90("control")
    elif pulse == "iSWAP":
        iSWAP("TC")
    elif pulse == "Rx_π":
        Rx(π, "control")
    else:
        return None

def play_readout(op):
    if op == "Rx_π_2":
        X90("control")
    elif op == "Ry_π_2":
        Y90("control")
    elif op == "iSWAP":
        iSWAP("TC")
    elif op == "CPHASE":
        CPHASE("TC")
    elif op == "readout":
        measure_and_reset_state("control", "RR")
        # this macro has to be redefined there to get the joint state of hidden and control