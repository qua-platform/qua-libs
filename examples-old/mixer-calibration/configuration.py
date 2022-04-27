import numpy as np

pulse_len = 100
qubit_IF = 50e6
qubit_LO = 2e9


def IQ_imbalance_correction(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},  # I
                2: {"offset": +0.0},  # Q
            },
        }
    },
    "elements": {
        "qubit": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": qubit_LO,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": qubit_IF,
            "operations": {
                "test_pulse": "constPulse",
            },
            "time_of_flight": 180,
            "smearing": 0,
        },
    },
    "pulses": {
        "constPulse": {
            "operation": "control",
            "length": pulse_len,
            "waveforms": {"I": "const_wf", "Q": "const_wf"},
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.45},
        # It is good practice to output from the OPX an amplitude which is close to the maximum, in order to have the
        # maximum dynamic range. If the amplitude needs to be lower than 0.1 (in order to get rid of spurs), then it is
        # better to add external attenuators and increase the amplitude.
    },
    "mixers": {
        "mixer_qubit": [
            {
                "intermediate_frequency": qubit_IF,
                "lo_frequency": qubit_LO,
                "correction": IQ_imbalance_correction(0, 0),
            }
        ],
    },
}
