import numpy as np
from qm.qua import *

π = np.pi
π_2 = np.pi / 2


def gauss(amplitude, mu, sigma, delf, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma**2))
    # Detuning correction Eqn. (4) in Chen et al. PRL, 116, 020501 (2016)
    gauss_wave = gauss_wave * np.exp(2 * np.pi * delf * t)
    return [float(x) for x in gauss_wave]


def gauss_der(amplitude, mu, sigma, delf, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_der_wave = amplitude * (-2 * (t - mu)) * np.exp(-((t - mu) ** 2) / (2 * sigma**2))
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
y180waveform = gauss(
    2 * x90amp, x90mean, x90std, x90detuning, x90duration
)  # Assume you have calibration for a X90 pulse
y180der_waveform = gauss_der(2 * x90amp, x90mean, x90std, x90detuning, x90duration)

state_prep = {
    "00": [],
    "00-i10": ["Rx_π_2"],
    "-i10": ["Rx_π"],
    "00+10": ["Ry_π_2"],
    "01": ["Rx_π", "iSWAP"],
    "-i11": ["Rx_π", "iSWAP", "Rx_π"],
    "01-i11": ["Rx_π", "iSWAP", "Rx_π_2"],
    "01+11": ["Rx_π", "iSWAP", "Ry_π_2"],
    "00+01": ["Rx_π_2", "iSWAP"],
    "-i10-i11": ["Rx_π_2", "iSWAP", "Rx_π"],
    "00+01-i10-i11": ["Rx_π_2", "iSWAP", "Rx_π_2"],
    "00+01+10+11": ["Rx_π_2", "iSWAP", "Ry_π_2"],
    "00+i01": ["Ry_π_2", "iSWAP"],
    "-i10+11": ["Ry_π_2", "iSWAP", "Rx_π"],
    "00+i01-i10+11": ["Ry_π_2", "iSWAP", "Rx_π_2"],
    "00+i01+10+i11": ["Ry_π_2", "iSWAP", "Ry_π_2"],
}
config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},  # Control qubit I component
                2: {"offset": +0.0},  # Control qubit Q component
                3: {"offset": +0.0},  # Readout resonator I component
                4: {"offset": +0.0},  # Readout resonator Q component
                5: {"offset": +0.0},  # Tunable coupler I component (flux-tunable transmon)
                6: {"offset": +0.0},  # Tunable coupler Q component
            },
            "analog_inputs": {
                1: {"offset": +0.0},  # Input of the readout resonator coupled to control qubit
            },
        }
    },
    "elements": {
        "control": {
            # Control qubit, hidden qubit is off the configuration, everything is played indirectly via TC below
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": 5.10e7,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": 0,  # 6.19e9,
            "operations": {
                "X90": "X90_pulse",
                "Y90": "Y90_pulse",
                "Y180": "Y180_pulse",
            },
        },
        "RR": {  # Readout resonator, connected to control qubit
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": 6.00e7,
                "mixer": "mixer_res",
            },
            "intermediate_frequency": 0,
            "operations": {
                "ReadoutOp": "readout_pulse",
            },
            "time_of_flight": 180,  # Measurement parameters
            "smearing": 0,
            "outputs": {"out1": ("con1", 1)},
        },
        "TC": {  # Tunable coupler, connection between control and hidden qubit (used for 2 qubit gates)
            "mixInputs": {
                "I": ("con1", 5),
                "Q": ("con1", 6),
                "lo_frequency": 5.0e7,
                "mixer": "mixer_TC",
            },
            "intermediate_frequency": 0,  # 5.2122e9,
            "operations": {
                "iSWAP": "X90_pulse",
                "CPHASE": "Y90_pulse",
            },
        },
    },
    "pulses": {
        "readout_pulse": {  # Readout pulse
            "operation": "measurement",
            "length": 200,
            "waveforms": {
                "I": "exc_wf",  # Decide what pulse to apply for each component
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
            "length": 1000,
            "waveforms": {"I": "x90_wf", "Q": "x90_der_wf"},
        },
        "Y90_pulse": {
            "operation": "control",
            "length": 1000,
            "waveforms": {"I": "x90_der_wf", "Q": "x90_wf"},
        },
        "Y180_pulse": {
            "operation": "control",
            "length": 1000,
            "waveforms": {"I": "y180_der_wf", "Q": "y180_wf"},
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "x90_wf": {"type": "arbitrary", "samples": x90waveform},
        "x90_der_wf": {"type": "arbitrary", "samples": x90der_waveform},
        "y180_wf": {"type": "arbitrary", "samples": y180waveform},
        "y180_der_wf": {"type": "arbitrary", "samples": y180der_waveform},
        "exc_wf": {"type": "constant", "sample": 0.479},
    },
    "digital_waveforms": {"marker1": {"samples": [(1, 4), (0, 2), (1, 1), (1, 0)]}},
    "integration_weights": {  # Define integration weights for measurement demodulation
        "integW1": {"cosine": [4.0] * 25, "sine": [0.0] * 25},
        "integW2": {
            "cosine": [0.0] * 25,
            "sine": [4.0] * 25,
        },
    },
    "mixers": {  # Potential corrections to be brought related to the IQ mixing scheme
        "mixer_res": [
            {
                "intermediate_frequency": 0,
                "lo_frequency": 6.00e7,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
        ],
        "mixer_qubit": [
            {
                "intermediate_frequency": 0,
                "lo_frequency": 5.10e7,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
        ],
        "mixer_TC": [
            {
                "intermediate_frequency": 0,
                "lo_frequency": 5.00e7,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
        ],
    },
}

state_prep = {
    "00": [],
    "00-i10": ["Rx_π_2"],
    "-i10": ["Rx_π"],
    "00+10": ["Ry_π_2"],
    "01": ["Rx_π", "iSWAP"],
    "-i11": ["Rx_π", "iSWAP", "Rx_π"],
    "01-i11": ["Rx_π", "iSWAP", "Rx_π_2"],
    "01+11": ["Rx_π", "iSWAP", "Ry_π_2"],
    "00+01": ["Rx_π_2", "iSWAP"],
    "-i10-i11": ["Rx_π_2", "iSWAP", "Rx_π"],
    "00+01-i10-i11": ["Rx_π_2", "iSWAP", "Rx_π_2"],
    "00+01+10+11": ["Rx_π_2", "iSWAP", "Ry_π_2"],
    "00+i01": ["Ry_π_2", "iSWAP"],
    "-i10+11": ["Ry_π_2", "iSWAP", "Rx_π"],
    "00+i01-i10+11": ["Ry_π_2", "iSWAP", "Rx_π_2"],
    "00+i01+10+i11": ["Ry_π_2", "iSWAP", "Ry_π_2"],
}

tomography_set = {
    "ID__σ_x": ["CPHASE", "iSWAP", "Rx_π_2"],
    "ID__σ_y": ["CPHASE", "iSWAP", "Ry_π_2"],
    "ID__σ_z": ["iSWAP"],
    "σ_z__ID": [],
    "σ_z__σ_x": ["iSWAP", "Rx_π_2"],
    "σ_z__σ_y": ["iSWAP", "Ry_π_2"],
    "σ_z__σ_z": ["Rx_π_2", "CPHASE", "Rx_π_2"],
    "σ_y__ID": ["Rx_π_2"],
    "σ_y__σ_x": ["Rx_π_2", "iSWAP", "Rx_π_2"],
    "σ_y__σ_y": ["Rx_π_2", "iSWAP", "Ry_π_2"],
    "σ_y__σ_z": ["CPHASE", "Rx_π_2"],
    "σ_x__ID": ["Ry_π_2"],
    "σ_x__σ_x": ["Ry_π_2", "iSWAP", "Rx_π_2"],
    "σ_x__σ_y": ["Ry_π_2", "iSWAP", "Ry_π_2"],
    "σ_x__σ_z": ["CPHASE", "Ry_π_2"],
}
processes = {
    "1": ["Rx_π_2"],
    "2": ["Ry_π_2"],
    "3": ["iSWAP"],
    "4": ["CPHASE"]
    # Add any arbitrary process we'd like to characterize here, as a sequence of previous elementary gates
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
    play("X90" * amp(𝜆 / (π / 2)), tgt)


def Ry(𝜆, tgt):
    # U3(tgt, 𝜆, 0, 0)
    play("Y90" * amp(𝜆 / (π / 2)), tgt)


def X90(tgt):  # Should actually be calibrated to do a -π/2 rotation
    play("X90", tgt)


def Y90(tgt):  # Should actually be calibrated to do a -π/2 rotation
    play("Y90", tgt)


def Y180(tgt):
    play("Y180", tgt)


def iSWAP(tgt):
    play("iSWAP", tgt)


def CPHASE(tgt):
    play("CPHASE", tgt)


def state_saving(
    I, Q, state_c, state_h, stream_c, stream_h
):  # Do state estimation protocol in QUA, and save the associated state
    """Define coefficients a & b, c defining the line separaation plane
    between states 00, 01, 10, 11 in IQ Plane (prior calibration required),
    here a & b are arbitrary"""
    # a = declare(fixed, value=1.)
    # b = declare(fixed, value=2.)
    # c = declare(fixed, value=3.)
    a = 1
    b = 2
    c = 3

    assign(state_c, ((a < Q) & (Q < b)) | (Q > c))
    assign(state_h, (Q > b))
    # with if_(Q < a):
    #     assign(state_c, 0)
    #     assign(state_h, 0)
    # with if_((a < Q) & (Q < b)):
    #     assign(state_c, 1)
    #     assign(state_h, 0)
    # with if_((b < Q) & (Q < c)):
    #     assign(state_c, 0)
    #     assign(state_h, 1)
    # with else_():
    #     assign(state_c, 1)
    #     assign(state_h, 1)
    save(state_c, stream_c)
    save(state_h, stream_h)


def measure_and_reset_state(RR, I, Q, A, B, stream_A, stream_B):
    measure("ReadoutOp", RR, None, ("integW1", I), ("integW2", Q))
    state_saving(I, Q, A, B, stream_A, stream_B)  # To be redefined to match the joint qubit readout
    # Active reset
    with if_((A == 1) & (B == 1)):
        Rx(π, "control")
        iSWAP("TC")
        Rx(π, "control")

    with if_((A == 1) & (B == 0)):
        Rx(π, "control")
    with if_((A == 0) & (B == 1)):
        iSWAP("TC")
        Rx(π, "control")


def play_pulse(pulse):
    if pulse == "Rx_π_2":
        X90("control")
    elif pulse == "Ry_π_2":
        Y90("control")
    elif pulse == "iSWAP":
        iSWAP("TC")
    elif pulse == "Rx_π":
        Rx(π, "control")
    elif pulse == "CPHASE":
        CPHASE("TC")
    else:
        return None
