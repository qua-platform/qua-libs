import numpy as np
import matplotlib.pyplot as plt


#############################
# simulation helpers #
#############################


def gauss(amplitude, mu, sigma, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma**2))
    return np.array([float(x) for x in gauss_wave])


def simulate_pulse(IF_freq, chi, k, Ts, Td, input_I, input_Q):
    I = [0]
    Q = [0]
    # solve numerically a simplified version of the readout resonator
    for t in range(Ts):
        I.append(I[-1] + (input_I[t] / 2 - k * I[-1] + Q[-1] * chi))
        Q.append(Q[-1] + (input_Q[t] / 2 - k * Q[-1] - I[-1] * chi))

    for t in range(Td - 1):
        I.append(I[-1] + (-k * I[-1] + Q[-1] * chi))
        Q.append(Q[-1] + (-k * Q[-1] - I[-1] * chi))

    I = np.array(I)
    Q = np.array(Q)
    t = np.arange(len(I))

    S = I * np.cos(2 * np.pi * IF_freq * t * 1e-9) + Q * np.sin(2 * np.pi * IF_freq * t * 1e-9)

    return t, I, Q, S


lo_freq_cavity = 8.0e9
cavity_IF = 180e6
lo_freq_qubit = 7.4e9
qubit_IF = 60e6
lo_freq_rr = 9.3e9
rr_IF = 60e6

readout_len = 380
IF_freq = rr_IF
Td = 200
Ts = readout_len - Td
power = 0.2
const_I = [power] * Ts
const_Q = [power] * Ts

alpha = 1 + 1j
sigma_displace = 4
power_displace = alpha / np.sqrt(2 * np.pi) / sigma_displace
displace_len = 8 * sigma_displace
displace_I = gauss(
    np.real(alpha),
    displace_len / 2 - 3 * int(sigma_displace + 1),
    sigma_displace,
    displace_len,
)
displace_Q = gauss(
    np.imag(alpha),
    displace_len / 2 - 3 * int(sigma_displace + 1),
    sigma_displace,
    displace_len,
)

gauss_sigma = 6
pulse_len = 8 * gauss_sigma
gauss_pulse = gauss(0.2, -(pulse_len / 2 - 3 * gauss_sigma), gauss_sigma, pulse_len)

k = 0.04
chi = 0.023

[tdis_, Idis_, Qdis_, Sdis_] = simulate_pulse(cavity_IF, -1 * chi, k, displace_len - 1, 0, displace_I, displace_Q)

[tg_, Ig_, Qg_, Sg_] = simulate_pulse(rr_IF, -1 * chi, k, Ts, Td, const_I, const_Q)
[te_, Ie_, Qe_, Se_] = simulate_pulse(rr_IF, 1 * chi, k, Ts, Td, const_I, const_Q)

divide_signal_factor = 10
config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": 0},
                2: {"offset": 0},
                3: {"offset": 0},
                4: {"offset": 0},
                5: {"offset": 0},
                6: {"offset": 0},
            },
            "digital_outputs": {
                1: {},
            },
            "analog_inputs": {
                1: {"offset": 0},
                2: {"offset": 0},
            },
        },
    },
    "elements": {
        "cavity_I": {
            "singleInput": {
                "port": ("con1", 3),
                # 'lo_frequency': lo_freq_cavity,
                # 'mixer': 'mixer_cavity',
            },
            "intermediate_frequency": cavity_IF,
            "operations": {
                "displace_I": "displace_pulse_I",
            },
            "time_of_flight": 188,
            "smearing": 0,
        },
        "cavity_Q": {
            "singleInput": {
                "port": ("con1", 4),
                # 'lo_frequency': lo_freq_cavity,
                # 'mixer': 'mixer_cavity',
            },
            "intermediate_frequency": cavity_IF,
            "operations": {
                "displace_Q": "displace_pulse_Q",
            },
            "time_of_flight": 188,
            "smearing": 0,
        },
        "rr": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": lo_freq_rr,
                "mixer": "mixer_rr",
            },
            "intermediate_frequency": rr_IF,
            "operations": {
                "readout": "readout_pulse",
                "readout_g": "readout_pulse_g",
                "readout_e": "readout_pulse_e",
            },
            "outputs": {
                "out1": ("con1", 1),
                "out2": ("con1", 2),
            },
            "time_of_flight": 188,
            "smearing": 0,
        },
        "qubit": {
            "mixInputs": {
                "I": ("con1", 5),
                "Q": ("con1", 6),
                "lo_frequency": lo_freq_qubit,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": qubit_IF,
            "operations": {
                "x_pi/2": "x_pi/2_pulse",
            },
            "time_of_flight": 188,
            "smearing": 0,
        },
    },
    "pulses": {
        "readout_pulse": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {"I": "Ig_wf", "Q": "Qg_wf"},
            "integration_weights": {
                "integW_cos": "integW_cos",
                "integW_sin": "integW_sin",
            },
            "digital_marker": "ON",
        },
        "readout_pulse_g": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {"I": "Ig_wf", "Q": "Qg_wf"},
            "integration_weights": {
                "integW_cos": "integW_cos",
                "integW_sin": "integW_sin",
            },
            "digital_marker": "ON",
        },
        "readout_pulse_e": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {"I": "Ie_wf", "Q": "Qe_wf"},
            "integration_weights": {
                "integW_cos": "integW_cos",
                "integW_sin": "integW_sin",
            },
            "digital_marker": "ON",
        },
        "x_pi/2_pulse": {
            "operation": "control",
            "length": pulse_len,
            "waveforms": {"I": "gauss_wf", "Q": "zero_wf"},
        },
        "displace_pulse_I": {
            "operation": "control",
            "length": displace_len,
            "waveforms": {
                "single": "Idis_wf",
            },
        },
        "displace_pulse_Q": {
            "operation": "control",
            "length": displace_len,
            "waveforms": {"single": "Qdis_wf"},
        },
    },
    "waveforms": {
        "zero_wf": {"type": "constant", "sample": 0.0},
        "const_wf": {"type": "constant", "sample": 0.1},
        "gauss_wf": {"type": "arbitrary", "samples": gauss_pulse},
        "Ig_wf": {
            "type": "arbitrary",
            "samples": [float(arg / divide_signal_factor) for arg in Ig_],
        },
        "Qg_wf": {
            "type": "arbitrary",
            "samples": [float(arg / divide_signal_factor) for arg in Qg_],
        },
        "Ie_wf": {
            "type": "arbitrary",
            "samples": [float(arg / divide_signal_factor) for arg in Ie_],
        },
        "Qe_wf": {
            "type": "arbitrary",
            "samples": [float(arg / divide_signal_factor) for arg in Qe_],
        },
        "Idis_wf": {"type": "arbitrary", "samples": [float(arg) for arg in displace_I]},
        "Qdis_wf": {"type": "arbitrary", "samples": [float(arg) for arg in displace_Q]},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "integW_cos": {
            "cosine": [1.0] * 120,
            "sine": [0.0] * 120,
        },
        "integW_sin": {
            "cosine": [0.0] * 120,
            "sine": [1.0] * 120,
        },
    },
    "mixers": {
        "mixer_cavity": [
            {
                "intermediate_frequency": cavity_IF,
                "lo_frequency": lo_freq_cavity,
                "correction": [1, 0, 0, 1],
            },
        ],
        "mixer_rr": [
            {
                "intermediate_frequency": rr_IF,
                "lo_frequency": lo_freq_rr,
                "correction": [1, 0, 0, 1],
            },
        ],
        "mixer_qubit": [
            {
                "intermediate_frequency": qubit_IF,
                "lo_frequency": lo_freq_qubit,
                "correction": [1, 0, 0, 1],
            },
        ],
    },
}
