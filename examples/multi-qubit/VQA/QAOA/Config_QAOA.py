import itertools

import numpy as np
from qm.qua import *

π = np.pi


def gauss(amplitude, mu, sigma, delf, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    # Detuning correction Eqn. (4) in Chen et al. PRL, 116, 020501 (2016)
    gauss_wave = gauss_wave * np.exp(2 * np.pi * delf * t)
    return [float(x) for x in gauss_wave]


def gauss_der(amplitude, mu, sigma, delf, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_der_wave = (
        amplitude * (-2 * (t - mu)) * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    )
    # Detuning correction Eqn. (4) in Chen et al. PRL, 116, 020501 (2016)
    gauss_der_wave = gauss_der_wave * np.exp(2 * np.pi * delf * t)
    return [float(x) for x in gauss_der_wave]


# X90 definition

x90amp = 0.4
x90std = 0.2
x90mean = 0
x90duration = 80
x90detuning = 0
x90waveform = gauss(
    x90amp, x90mean, x90std, x90detuning, x90duration
)  # Assume you have calibration for a X90 pulse
lmda = 0.5  # Define scaling parameter for Drag Scheme
alpha = -1  # Define anharmonicity parameter
x90der_waveform = gauss_der(x90amp, x90mean, x90std, x90detuning, x90duration)

CR_duration = 120
LO_freq = 1e9
qubit_IF = 0
rr_IF = 0
IBMconfig = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},
                2: {"offset": +0.0},
                3: {"offset": +0.0},
                4: {"offset": +0.0},
                5: {"offset": +0.0},
                6: {"offset": +0.0},
                7: {"offset": +0.0},
                8: {"offset": +0.0},
            },
            "analog_inputs": {
                1: {"offset": +0.0},
                2: {"offset": +0.0},
            },
        },
        "con2": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},
                2: {"offset": +0.0},
                3: {"offset": +0.0},
                4: {"offset": +0.0},
                5: {"offset": +0.0},
                6: {"offset": +0.0},
                7: {"offset": +0.0},
                8: {"offset": +0.0},
            },
            "analog_inputs": {
                1: {"offset": +0.0},
                2: {"offset": +0.0},
            },
        },
    },
    "elements": {
        "q0": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": LO_freq,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": qubit_IF,
            "operations": {
                "X90": "X90_pulse",
                "Y90": "Y90_pulse",
                "Y180": "Y180_pulse",
                "CR_with_q1": "CR_01",
                "CR_with_q2": "CR_02",
                "CR_with_q3": "CR_03",
            },
        },
        "rr0": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": LO_freq,
                "mixer": "mixer_res",
            },
            "intermediate_frequency": rr_IF,
            "operations": {
                "meas_pulse": "meas_pulse_in",
            },
            "time_of_flight": 180,  # Measurement parameters
            "smearing": 0,
            "outputs": {"out1": ("con1", 1)},
        },
        "q1": {
            "mixInputs": {
                "I": ("con1", 5),
                "Q": ("con1", 6),
                "lo_frequency": LO_freq,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": qubit_IF,
            "operations": {
                "X90": "X90_pulse",
                "Y90": "Y90_pulse",
                "Y180": "Y180_pulse",
                "CR_with_q0": "CR_10",
                "CR_with_q2": "CR_12",
                "CR_with_q3": "CR_13",
            },
        },
        "rr1": {
            "mixInputs": {
                "I": ("con1", 7),
                "Q": ("con1", 8),
                "lo_frequency": LO_freq,
                "mixer": "mixer_res",
            },
            "intermediate_frequency": rr_IF,
            "operations": {
                "meas_pulse": "meas_pulse_in",
            },
            "time_of_flight": 180,  # Measurement parameters
            "smearing": 0,
            "outputs": {"out1": ("con1", 2)},
        },
        "q2": {
            "mixInputs": {
                "I": ("con2", 1),
                "Q": ("con2", 2),
                "lo_frequency": LO_freq,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": qubit_IF,
            "operations": {
                "X90": "X90_pulse",
                "Y90": "Y90_pulse",
                "Y180": "Y180_pulse",
                "CR_with_q0": "CR_20",
                "CR_with_q1": "CR_21",
                "CR_with_q3": "CR_23",
            },
        },
        "rr2": {
            "mixInputs": {
                "I": ("con2", 3),
                "Q": ("con2", 4),
                "lo_frequency": LO_freq,
                "mixer": "mixer_res",
            },
            "intermediate_frequency": rr_IF,
            "operations": {
                "meas_pulse": "meas_pulse_in",
            },
            "time_of_flight": 180,  # Measurement parameters
            "smearing": 0,
            "outputs": {"out1": ("con2", 1)},
        },
        "q3": {
            "mixInputs": {
                "I": ("con2", 5),
                "Q": ("con2", 6),
                "lo_frequency": LO_freq,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": qubit_IF,
            "operations": {
                "X90": "X90_pulse",
                "Y90": "Y90_pulse",
                "Y180": "Y180_pulse",
                "CR_with_q0": "CR_30",
                "CR_with_q1": "CR_31",
                "CR_with_q2": "CR_32",
            },
        },
        "rr3": {
            "mixInputs": {
                "I": ("con2", 7),
                "Q": ("con2", 8),
                "lo_frequency": LO_freq,
                "mixer": "mixer_res",
            },
            "intermediate_frequency": rr_IF,
            "operations": {
                "meas_pulse": "meas_pulse_in",
            },
            "time_of_flight": 180,  # Measurement parameters
            "smearing": 0,
            "outputs": {"out1": ("con2", 2)},
        },
    },
    "pulses": {
        "meas_pulse_in": {  # Readout pulse
            "operation": "measurement",
            "length": 200,
            "waveforms": {
                "I": "exc_wf",
                "Q": "zero_wf",
            },
            "integration_weights": {
                "integW1": "integW1",
                "integW2": "integW2",
            },
            "digital_marker": "marker1",
        },
        "X90_pulse": {
            "operation": "control",
            "length": x90duration,
            "waveforms": {"I": "x90_wf", "Q": "x90_der_wf"},
        },
        "Y90_pulse": {
            "operation": "control",
            "length": x90duration,
            "waveforms": {"I": "x90_der_wf", "Q": "x90_wf"},
        },
        "Y180_pulse": {
            "operation": "control",
            "length": x90duration,
            "waveforms": {"I": "y180_der_wf", "Q": "y180_wf"},
        },

        **{
            f"CR_{i}{j}": {
                "operation": "control",
                "length": CR_duration,
                "waveforms": {"I": "CR_wf", "Q": "zero_wf"},
            }
            for i,j in itertools.product(range(4), range(4))
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "x90_wf": {"type": "arbitrary", "samples": x90waveform},
        "CR_wf": {"type": "arbitrary", "samples": [0.2] * CR_duration},
        "x90_der_wf": {"type": "arbitrary", "samples": x90der_waveform},
        "y180_wf": {"type": "arbitrary", "samples": list(2 * np.array(x90waveform))},
        "y180_der_wf": {"type": "arbitrary", "samples": list(2 * np.array(x90der_waveform))},
        "exc_wf": {"type": "constant", "sample": 0.479},
    },
    "digital_waveforms": {"marker1": {"samples": [(1, 4), (0, 2), (1, 1), (1, 0)]}},
    "integration_weights": {  # Define integration weights for measurement demodulation
        "integW1": {"cosine": [4.0] * 27, "sine": [0.0] * 27},
        "integW2": {"cosine": [0.0] * 27, "sine": [4.0] * 27},
    },
    "mixers": {  # Potential corrections to be brought related to the IQ mixing scheme
        "mixer_res": [
            {
                "intermediate_frequency": rr_IF,
                "lo_frequency": LO_freq,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
        ],
        "mixer_qubit": [
            {
                "intermediate_frequency": qubit_IF,
                "lo_frequency": LO_freq,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
        ],
    },
}


def cost_function_C(
    x, G
):  # Cost function for MaxCut problem, needs to be adapted to the considered optimization problem

    E = G.edges()
    if len(x) != len(G.nodes()):
        return np.nan

    C = 0
    for edge in E:
        e1 = edge[0]
        e2 = edge[1]
        w = G[e1][e2]["weight"]
        C += w * x[e1] * (1 - x[e2]) + w * x[e2] * (1 - x[e1])

    return C


# QUA macros (pulse definition of quantum gates)
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


def CR(ctrl, tgt):
    """https://arxiv.org/abs/2004.06755"""
    return None


def CNOT(ctrl, tgt):  # To be defined
    frame_rotation(π / 2, ctrl)
    X90(tgt)
    CR(tgt, ctrl)


def Rz(𝜆, tgt):
    frame_rotation(-𝜆, tgt)


def Ry(𝜆, tgt):
    U3(tgt, 𝜆, 0, 0)


def CU1(𝜆, ctrl, tgt):
    Rz(𝜆 / 2.0, ctrl)
    CNOT(ctrl, tgt)
    Rz(-𝜆 / 2.0, tgt)
    CNOT(ctrl, tgt)
    Rz(𝜆 / 2.0, tgt)


def Rx(𝜆, tgt):
    U3(tgt, 𝜆, -π / 2, π / 2)


def X90(tgt):
    play("X90", tgt)


def Y90(tgt):
    play("Y90", tgt)


def Y180(tgt):
    play("Y180", tgt)


def SWAP(qubit1, qubit2):
    CNOT(qubit1, qubit2)
    CNOT(qubit2, qubit1)
    CNOT(qubit1, qubit2)


# Stream processing QUA macros
def raw_saving(I, Q, I_stream, Q_stream):
    # Saving command
    save(I, I_stream)
    save(Q, Q_stream)
