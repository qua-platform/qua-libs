import numpy as np
from scipy.signal.windows import gaussian
readout_len = 400
q0_IF = 0
q1_IF = 0
rr0_IF = 50e6
rr1_IF = 50e6
qubit_LO = 6.345e9
rr_LO = 4.755e9
coupler_LO = 4.755e9

CR_amp = 0.2
CR_flat_duration = 80
CR_total_duration = 120
CR_sigma = 20

pi_amp = 0.2
pi_duration = 40
pi_std = 20


def gauss(amplitude, mu, sigma, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    return [float(x) for x in gauss_wave]


def flat_top_gaussian(amp, pulse_length, flat_length, sigma):
    gauss_length = pulse_length - flat_length
    gauss = (amp * gaussian(gauss_length, sigma)).tolist()
    return gauss[:gauss_length//2] + [amp] * flat_length + gauss[gauss_length//2:]


def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g ** 2) * (2 * c ** 2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},  # q0_I
                2: {"offset": +0.0},  # q0_Q
                3: {"offset": +0.0},  # q1_I
                4: {"offset": +0.0},  # q1_Q
                5: {"offset": +0.0},  # rr1-I
                6: {"offset": +0.0},  # rr1-Q
                7: {"offset": +0.0},  # rr2-I
                8: {"offset": +0.0},  # rr2-Q
            },
            "analog_inputs": {
                1: {"offset": +0.0},
                2: {"offset": +0.0},
            },
        },
    },
    "elements": {
        **{
            qubit[0]: {
                "mixInputs": {
                    "I": ("con1", qubit[1]),
                    "Q": ("con1", qubit[2]),
                    "lo_frequency": qubit_LO,
                    "mixer": "mixer_qubit",
                },
                "intermediate_frequency": qubit[3],
                "operations": {
                    "I": f"IPulse",
                    "X/2": f"X/2Pulse_{i}",
                    "X": f"XPulse_{i}",
                    "-X/2": f"-X/2Pulse_{i}",
                    "Y/2": f"Y/2Pulse_{i}",
                    "Y": f"YPulse_{i}",
                    "-Y/2": f"-Y/2Pulse_{i}",
                    "CR_pi_over_4": f"CRPulse_{i}",
                },
            }
            for i, qubit in enumerate([("q0", 1, 2, q0_IF), ("q1", 3, 4, q1_IF)])
        },
        "rr0": {
            "mixInputs": {
                "I": ("con1", 5),
                "Q": ("con1", 6),
                "lo_frequency": rr_LO,
                "mixer": "mixer_RR",
            },
            "intermediate_frequency": rr0_IF,
            "operations": {
                "readout": "readout_pulse_0",
            },
            "outputs": {"out1": ("con1", 1)
                        },
            "time_of_flight": 28,
            "smearing": 0,
        },
        "rr1": {
            "mixInputs": {
                "I": ("con1", 7),
                "Q": ("con1", 8),
                "lo_frequency": rr_LO,
                "mixer": "mixer_RR",
            },
            "intermediate_frequency": rr1_IF,
            "operations": {
                "readout": "readout_pulse_1",
            },
            "outputs": {"out1": ("con1", 2)
                        },
            "time_of_flight": 28,
            "smearing": 0,
        },
    },
    "pulses": {
        "IPulse": {
            "operation": "control",
            "length": pi_duration,
            "waveforms": {"I": "zero_wf", "Q": "zero_wf"},
        },
        **{
            f"CRPulse_{i}": {
                "operation": "control",
                "length": CR_total_duration,
                "waveforms": {"I": "cr_I_wf", "Q": "cr_Q_wf"},
            }
            for i in [0, 1]
        },

        **{
            f"XPulse_{i}": {
                "operation": "control",
                "length": pi_duration,
                "waveforms": {"I": "pi_wf", "Q": "zero_wf"},
            }
            for i in [0,1]
        },

        **{
            f"X/2Pulse_{i}": {
                "operation": "control",
                "length": pi_duration,
                "waveforms": {"I": "pi/2_wf", "Q": "zero_wf"},
            }
            for i in [0,1]
        },

        **{
            f"-X/2Pulse_{i}": {
                "operation": "control",
                "length": pi_duration,
                "waveforms": {"I": "-pi/2_wf", "Q": "zero_wf"},
            }
            for i in [0,1]
        },

        **{
            f"YPulse_{i}": {
                "operation": "control",
                "length": pi_duration,
                "waveforms": {"I": "zero_wf", "Q": "pi_wf"},
            }
            for i in [0,1]
        },

        **{
            f"Y/2Pulse_{i}": {
                "operation": "control",
                "length": pi_duration,
                "waveforms": {"I": "zero_wf", "Q": "pi/2_wf"},
            }
            for i in [0,1]
        },

        **{
            f"-Y/2Pulse_{i}": {
                "operation": "control",
                "length": pi_duration,
                "waveforms": {"I": "zero_wf", "Q": "-pi/2_wf"},
            }
            for i in [0,1]
        },

        **{
            f"readout_pulse_{i}": {
                "operation": "measurement",
                "length": readout_len,
                "waveforms": {"I": "readout_wf", "Q": "zero_wf"},
                "integration_weights": {
                    "integW1": "integW1",
                    "integW2": "integW2",
                },
                "digital_marker": "ON",
            }
            for i in [0,1]
        },

    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
        "cr_I_wf": {"type": "arbitrary", "samples": flat_top_gaussian(CR_amp, CR_total_duration, CR_flat_duration,
                                                                      CR_sigma)},
        "cr_Q_wf": {"type": "arbitrary", "samples": flat_top_gaussian(CR_amp, CR_total_duration, CR_flat_duration,
                                                                      CR_sigma)},
        "pi_wf": {"type": "arbitrary", "samples": gauss(pi_amp, 0, pi_std, pi_duration)},
        "-pi/2_wf": {"type": "arbitrary", "samples": gauss(-pi_amp/2, 0, pi_std, pi_duration)},
        "pi/2_wf": {"type": "arbitrary", "samples": gauss(pi_amp/2, 0, pi_std, pi_duration)},
        "zero_wf": {"type": "constant", "sample": 0},
        "readout_wf": {"type": "constant", "sample": 0.3},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "integW1": {
            "cosine": [1.0] * int(readout_len / 4),
            "sine": [0.0] * int(readout_len / 4),
        },
        "integW2": {
            "cosine": [0.0] * int(readout_len / 4),
            "sine": [1.0] * int(readout_len / 4),
        },
    },
    "mixers": {
        "mixer_qubit": [
            {
                "intermediate_frequency": q_IF,
                "lo_frequency": qubit_LO,
                "correction": IQ_imbalance(0.0, 0.0),
            }
            for q_IF in (q0_IF, q1_IF)],
        "mixer_RR": [
            {
                "intermediate_frequency": rr_IF,
                "lo_frequency": rr_LO,
                "correction": IQ_imbalance(0.0, 0.0),
            }
            for rr_IF in (rr0_IF, rr1_IF)],
    },
}
