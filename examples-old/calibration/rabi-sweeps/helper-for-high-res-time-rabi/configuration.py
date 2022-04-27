import numpy as np


def delayed_gauss(amp, length, sigma):
    gauss_arg = np.linspace(-sigma, sigma, length)
    delay = 16 - length - 4
    if delay < 0:
        return amp * np.exp(-(gauss_arg**2) / 2)

    return np.r_[np.zeros(delay), amp * np.exp(-(gauss_arg**2) / 2), np.zeros(4)]


# Definition of the sample for Gaussian pulse
gauss_wf_16ns = delayed_gauss(0.2, 16, 3)
gauss_wf_8ns = delayed_gauss(0.2, 8, 3)
gauss_wf_4ns = delayed_gauss(0.2, 4, 2)

# Setting up the configuration of the experimental setup
# Embedded in a Python dictionary
config = {
    "version": 1,
    "controllers": {  # Define the QM device and its outputs, in this case:
        "con1": {  # 2 analog outputs for the in-phase and out-of phase components
            "type": "opx1",  # of the qubit (I & Q)
            "analog_outputs": {
                1: {"offset": 0},
                2: {"offset": 0},
            },
        }
    },
    "elements": {  # Define the elements composing the quantum system, i.e the qubit
        "qubit": {
            "mixInputs": {
                "I": ("con1", 1),  # Connect the component to one output of the OPX
                "Q": ("con1", 2),
                "lo_frequency": 5.10e7,
                "mixer": "mixer_qubit",  # Associate a mixer entity to control the IQ mixing process
            },
            "intermediate_frequency": 5.15e3,  # Resonant frequency of the qubit
            "operations": {  # Define the set of operations doable on the qubit, each operation is related
                "gauss_pulse_1ns_res": "gauss_pulse_in1",  # to a pulse
                "gauss_pulse_2ns_res": "gauss_pulse_in2",  # to a pulse
                "gauss_pulse_4ns_res": "gauss_pulse_in4",  # to a pulse
            },
        },
    },
    "pulses": {  # Pulses definition
        "gauss_pulse_in1": {
            "operation": "control",
            "length": 16,
            "waveforms": {"I": "gauss_wf1", "Q": "zero_wf"},
        },
        "gauss_pulse_in2": {
            "operation": "control",
            "length": 16,
            "waveforms": {"I": "gauss_wf2", "Q": "zero_wf"},
        },
        "gauss_pulse_in4": {
            "operation": "control",
            "length": 16,
            "waveforms": {"I": "gauss_wf4", "Q": "zero_wf"},
        },
    },
    "waveforms": {  # Specify the envelope type of the pulses defined above
        "zero_wf": {"type": "constant", "sample": 0.0},
        "gauss_wf1": {"type": "arbitrary", "samples": gauss_wf_4ns.tolist()},
        "gauss_wf2": {"type": "arbitrary", "samples": gauss_wf_8ns.tolist()},
        "gauss_wf4": {"type": "arbitrary", "samples": gauss_wf_16ns.tolist()},
    },
    "digital_waveforms": {"marker1": {"samples": [(1, 4), (0, 2), (1, 1), (1, 0)]}},
    "integration_weights": {  # Define integration weights for measurement demodulation
        "integW1": {
            "cosine": [
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
            ],
            "sine": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        },
        "integW2": {
            "cosine": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "sine": [
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
            ],
        },
    },
    "mixers": {  # Potential corrections to be brought related to the IQ mixing scheme
        "mixer_qubit": [
            {
                "intermediate_frequency": 5.15e3,
                "lo_frequency": 5.10e7,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
        ],
    },
}
