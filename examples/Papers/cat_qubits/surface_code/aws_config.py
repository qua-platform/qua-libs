import numpy as np


def gauss(amplitude, mu, sigma, delf, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    # Detuning correction Eqn. (4) in Chen et al. PRL, 116, 020501 (2016)
    gauss_wave = gauss_wave * np.exp(2 * np.pi * delf * t)
    return [float(x) for x in gauss_wave]


def gauss_der(amplitude, mu, sigma, delf, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_der_wave = (
        amplitude
        * (-2 * (t - mu))
        * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
        / (2 * sigma ** 2)
    )
    # Detuning correction Eqn. (4) in Chen et al. PRL, 116, 020501 (2016)
    gauss_der_wave = gauss_der_wave * np.exp(2 * np.pi * delf * t)
    return [float(x) for x in gauss_der_wave]


# X90 definition
x90amp = 0.2
xamp = 0.4
SQstd = 20
SQmean = 0
SQduration = 80
SQdetuning = 0
lmda = 0.5  # Define scaling parameter for Drag Scheme
alpha = -1  # Define anharmonicity parameter

dis_amp = 0.2
dis_std = 0.2
dis_mean = 0
dis_duration = 1000
dis_detuning = 0

displace_wf = gauss(dis_amp, dis_mean, dis_std, dis_detuning, dis_duration)

probe_amp = 0.2
probe_std = 0.2
probe_mean = 0
probe_duration = 1000
probe_detuning = 0

probe_wf = gauss(dis_amp, dis_mean, dis_std, dis_detuning, dis_duration)
omega_LO_q = 1e9
omega_LO_b = 1e9
omega_LO_a = 1e9
omega_LO_ATS = 1e9
omega_LO_RR = 1e9
# Index 0 : Ancilla, Index 1 : Readout mode, Index 2 to 5 : Data

omega_alpha = 1.3e8  # Data mode frequency
omega_gamma = 1e8  # Data mode frequency
omega_beta = 1.5e8  # Ancilla mode frequency
omega_rho = 1.4e8  # Readout mode frequency
cat_frequencies = {
    "alpha": omega_alpha,  # Data qubit α
    "beta": omega_beta,  # Ancilla qubit β
    "gamma": omega_gamma,  # Data qubit γ
    "rho": omega_rho,  # Phononic readout mode frequency
}

omega_b1 = 60e6
omega_b2 = 65e6

delta_beta = 5e5
delta_alpha = 5e5
delta_gamma = 5e5

omega_d1_beta = omega_b1 - delta_beta
omega_d1_alpha = omega_b1 - delta_alpha
omega_d1_gamma = omega_b1 - delta_gamma

omega_p1_alpha = 2 * omega_alpha - omega_b1 + delta_alpha
omega_p1_beta = 2 * omega_beta - omega_b1 + delta_beta
omega_p1_gamma = 2 * omega_gamma - omega_b1 + delta_gamma
omega_d2_beta = omega_b2 - delta_beta
omega_d2_alpha = omega_b2 - delta_alpha
omega_d2_gamma = omega_b2 - delta_gamma

omega_p2_alpha = 2 * omega_alpha - omega_b2 + delta_alpha
omega_p2_beta = 2 * omega_beta - omega_b2 + delta_beta
omega_p2_gamma = 2 * omega_gamma - omega_b2 + delta_gamma

omega_CNOT1_alpha = (omega_beta - omega_d1_alpha) / 2
omega_CNOT2_alpha = (omega_beta - omega_d2_alpha) / 2

omega_CNOT1_gamma = (omega_beta - omega_d1_gamma) / 2
omega_CNOT2_gamma = (omega_beta - omega_d2_gamma) / 2

buffer_frequencies = [
    omega_d1_gamma,
    omega_d2_gamma,
    omega_d1_beta,
    omega_d2_beta,
    omega_d1_alpha,
    omega_d2_alpha,
]
ATS_frequencies = [
    omega_p1_gamma,
    omega_p2_gamma,
    omega_p1_alpha,
    omega_p2_alpha,
    omega_p1_beta,
    omega_p2_beta,
    omega_CNOT1_gamma,
    omega_CNOT2_gamma,
    omega_CNOT1_alpha,
    omega_CNOT2_alpha,
]


omega_RR = 40e6
omega_q = 30e6
chi_qa = 0.023  # Coupling constant between transmon and cat qubit cavity

g_readout_ancilla = 2 * np.pi * 1e6
t_swap = int(np.pi / 2 / g_readout_ancilla * 1e9 / 4) * 4  # cycle unit
deflation_duration = 100
readout_duration = 200
pump_duration = 400
drive_duration = 400

const_high = 0.24
const_low = 0.1
config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},  # Transmon I component
                2: {"offset": +0.0},  # Transmon qubit Q component
                3: {"offset": +0.0},  # Readout resonator I component
                4: {"offset": +0.0},  # Readout resonator Q component
                5: {"offset": +0.0},  # ATS 1 (left) I
                6: {"offset": +0.0},  # ATS 1 (left) Q
                7: {"offset": +0.0},  # Buffer drive 1 (left) I
                8: {"offset": +0.0},  # Buffer drive 1 (left) Q
                9: {"offset": +0.0},  # ATS 2 (right) I
                10: {"offset": +0.0},  # ATS 2 (right) Q
            },
            "analog_inputs": {
                1: {"offset": +0.0},  # Input of RR
                2: {"offset": +0.0},  # Input of RR
            },
        },
        "con2": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},  # Buffer drive 2 (right) I
                2: {"offset": +0.0},  # Buffer drive 2 (right) Q
            },
            "analog_inputs": {
                1: {"offset": +0.0},
            },
        },
    },
    "elements": {
        "t0": {  # Transmon qubit
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": omega_LO_q,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": omega_q,
            "operations": {
                "X90": "X90_pulse",
                "X": "X_pulse",
            },
        },
        "rr0": {  # Readout resonator, coupled to transmon
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": omega_LO_RR,
                "mixer": "mixer_res",
            },
            "intermediate_frequency": omega_RR,
            "operations": {
                "Readout_Op": "readout_pulse",
            },
            "time_of_flight": 28,  # Measurement parameters
            "smearing": 0,
            "outputs": {"out1": ("con1", 1), "out2": ("con1", 2)},
        },
        "gamma_1_ATS": {
            "mixInputs": {
                "I": ("con1", 5),
                "Q": ("con1", 6),
                "lo_frequency": omega_LO_ATS,
                "mixer": "mixer_ATS",
            },
            "intermediate_frequency": omega_p1_gamma,
            "operations": {"Pump_Op": "Pump_pulse", "swap": "swap_pulse"},
        },
        "gamma_1_buffer": {
            "mixInputs": {
                "I": ("con1", 7),
                "Q": ("con1", 8),
                "lo_frequency": omega_LO_b,
                "mixer": "mixer_buffer",
            },
            "intermediate_frequency": omega_d1_gamma,
            "operations": {"drive": "linear_pulse"},
        },
        "gamma_2_ATS": {
            "mixInputs": {
                "I": ("con1", 9),
                "Q": ("con1", 10),
                "lo_frequency": omega_LO_ATS,
                "mixer": "mixer_ATS",
            },
            "intermediate_frequency": omega_p2_gamma,
            "operations": {"Pump_Op": "Pump_pulse", "swap": "swap_pulse"},
        },
        "gamma_2_buffer": {
            "mixInputs": {
                "I": ("con2", 1),
                "Q": ("con2", 2),
                "lo_frequency": omega_LO_b,
                "mixer": "mixer_buffer",
            },
            "intermediate_frequency": omega_d2_gamma,
            "operations": {"drive": "linear_pulse"},
        },
        "alpha_1_ATS": {
            "mixInputs": {
                "I": ("con1", 5),
                "Q": ("con1", 6),
                "lo_frequency": omega_LO_ATS,
                "mixer": "mixer_ATS",
            },
            "intermediate_frequency": omega_p1_alpha,
            "operations": {"Pump_Op": "Pump_pulse", "swap": "swap_pulse"},
        },
        "alpha_1_buffer": {
            "mixInputs": {
                "I": ("con1", 7),
                "Q": ("con1", 8),
                "lo_frequency": omega_LO_b,
                "mixer": "mixer_buffer",
            },
            "intermediate_frequency": omega_d1_alpha,
            "operations": {"drive": "linear_pulse"},
        },
        "alpha_2_ATS": {
            "mixInputs": {
                "I": ("con1", 9),
                "Q": ("con1", 10),
                "lo_frequency": omega_LO_ATS,
                "mixer": "mixer_ATS",
            },
            "intermediate_frequency": omega_p2_alpha,
            "operations": {"Pump_Op": "Pump_pulse", "swap": "swap_pulse"},
        },
        "alpha_2_buffer": {
            "mixInputs": {
                "I": ("con2", 1),
                "Q": ("con2", 2),
                "lo_frequency": omega_LO_b,
                "mixer": "mixer_buffer",
            },
            "intermediate_frequency": omega_d2_alpha,
            "operations": {"drive": "linear_pulse"},
        },
        "beta_1_ATS": {
            "mixInputs": {
                "I": ("con1", 5),
                "Q": ("con1", 6),
                "lo_frequency": omega_LO_ATS,
                "mixer": "mixer_ATS",
            },
            "intermediate_frequency": omega_p1_beta,
            "operations": {"Pump_Op": "Pump_pulse", "swap": "swap_pulse"},
        },
        "beta_1_buffer": {
            "mixInputs": {
                "I": ("con1", 7),
                "Q": ("con1", 8),
                "lo_frequency": omega_LO_b,
                "mixer": "mixer_buffer",
            },
            "intermediate_frequency": omega_d1_beta,
            "operations": {"drive": "linear_pulse", "deflation": "deflation_pulse"},
        },
        "beta_2_ATS": {
            "mixInputs": {
                "I": ("con1", 9),
                "Q": ("con1", 10),
                "lo_frequency": omega_LO_ATS,
                "mixer": "mixer_ATS",
            },
            "intermediate_frequency": omega_p2_beta,
            "operations": {"Pump_Op": "Pump_pulse", "swap": "swap_pulse"},
        },
        "beta_2_buffer": {
            "mixInputs": {
                "I": ("con2", 1),
                "Q": ("con2", 2),
                "lo_frequency": omega_LO_b,
                "mixer": "mixer_buffer",
            },
            "intermediate_frequency": omega_d2_beta,
            "operations": {"drive": "linear_pulse", "deflation": "deflation_pulse"},
        },
        "CNOT_gamma_1_ATS": {
            "mixInputs": {
                "I": ("con1", 5),
                "Q": ("con1", 6),
                "lo_frequency": omega_LO_ATS,
                "mixer": "mixer_ATS",
            },
            "intermediate_frequency": omega_CNOT1_gamma,
            "operations": {"Pump_Op": "Pump_pulse", "swap": "swap_pulse"},
        },
        "CNOT_gamma_2_ATS": {
            "mixInputs": {
                "I": ("con1", 9),
                "Q": ("con1", 10),
                "lo_frequency": omega_LO_ATS,
                "mixer": "mixer_ATS",
            },
            "intermediate_frequency": omega_CNOT2_gamma,
            "operations": {"Pump_Op": "Pump_pulse", "swap": "swap_pulse"},
        },
        "CNOT_alpha_1_ATS": {
            "mixInputs": {
                "I": ("con1", 5),
                "Q": ("con1", 6),
                "lo_frequency": omega_LO_ATS,
                "mixer": "mixer_ATS",
            },
            "intermediate_frequency": omega_CNOT1_alpha,
            "operations": {"Pump_Op": "Pump_pulse", "swap": "swap_pulse"},
        },
        "CNOT_alpha_2_ATS": {
            "mixInputs": {
                "I": ("con1", 9),
                "Q": ("con1", 10),
                "lo_frequency": omega_LO_ATS,
                "mixer": "mixer_ATS",
            },
            "intermediate_frequency": omega_CNOT2_alpha,
            "operations": {"Pump_Op": "Pump_pulse", "swap": "swap_pulse"},
        },
    },
    "pulses": {
        "readout_pulse": {  # Readout pulse for readout resonator coupled to transmon
            "operation": "measurement",
            "length": readout_duration,
            "waveforms": {"I": "const_wf_high", "Q": "zero_wf"},
            "integration_weights": {
                "integW_cos": "integW_cos",
                "integW_sin": "integW_sin",
            },
            "digital_marker": "marker1",
        },
        "X90_pulse": {
            "operation": "control",
            "length": SQduration,
            "waveforms": {"I": "x90_wf", "Q": "x90_der_wf"},
        },
        "X_pulse": {
            "operation": "control",
            "length": SQduration,
            "waveforms": {"I": "x_wf", "Q": "x_der_wf"},
        },
        "Pump_pulse": {
            "operation": "control",
            "length": pump_duration,
            "waveforms": {"I": "const_wf", "Q": "zero_wf"},
        },
        "swap_pulse": {
            "operation": "control",
            "length": t_swap,
            "waveforms": {"I": "const_wf", "Q": "zero_wf"},
        },
        "Displace_pulse": {
            "operation": "control",
            "length": 1000,
            "waveforms": {"I": "disp_wf", "Q": "disp_wf"},
        },
        "linear_pulse": {
            "operation": "control",
            "length": drive_duration,
            "waveforms": {
                "I": "const_wf_high",
                "Q": "zero_wf",
            },
        },
        "deflation_pulse": {
            "operation": "control",
            "length": deflation_duration,
            "waveforms": {
                "I": "tanh_down_wf",  # Decide what pulse to apply for each component
                "Q": "zero_wf",
            },
        },
        "constPulse": {
            "operation": "control",
            "length": 1000,
            "waveforms": {"single": "const_wf"},
        },
    },
    "waveforms": {
        "const_wf_low": {"type": "constant", "sample": const_low},
        "const_wf_high": {"type": "constant", "sample": const_high},
        "const_wf": {"type": "constant", "sample": 0.2},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "x_wf": {
            "type": "arbitrary",
            "samples": gauss(xamp, SQmean, SQstd, SQdetuning, SQduration),
        },
        "x_der_wf": {
            "type": "arbitrary",
            "samples": gauss_der(xamp, SQmean, SQstd, SQdetuning, SQduration),
        },
        "x90_wf": {
            "type": "arbitrary",
            "samples": gauss(x90amp, SQmean, SQstd, SQdetuning, SQduration),
        },
        "x90_der_wf": {
            "type": "arbitrary",
            "samples": gauss_der(x90amp, SQmean, SQstd, SQdetuning, SQduration),
        },
        "disp_wf": {"type": "arbitrary", "samples": displace_wf},
        "probe_wf": {"type": "arbitrary", "samples": probe_wf},
        "exc_wf": {"type": "constant", "sample": 0.479},
        "tanh_down_wf": {
            "type": "arbitrary",
            "samples": [
                0.15 * np.tanh(-x) + 0.15
                for x in np.linspace(-3, 3, deflation_duration)
            ],
        },
    },
    "digital_waveforms": {"marker1": {"samples": [(1, 4), (0, 2), (1, 1), (1, 0)]}},
    "integration_weights": {  # Define integration weights for measurement demodulation
        "integW1": {"cosine": [4.0] * 25, "sine": [0.0] * 25},
        "integW2": {"cosine": [0.0] * 25, "sine": [4.0] * 25},
        "integW_cos": {"cosine": [1.0] * 120, "sine": [0.0] * 120},
        "integW_sin": {"cosine": [0.0] * 120, "sine": [1.0] * 120},
    },
    "mixers": {
        "mixer_res": [
            {
                "intermediate_frequency": omega_RR,
                "lo_frequency": omega_LO_RR,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
        ],
        "mixer_buffer": [
            {
                "intermediate_frequency": omega_B,
                "lo_frequency": omega_LO_b,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
            for omega_B in buffer_frequencies
        ],
        "mixer_qubit": [
            {
                "intermediate_frequency": omega_q,
                "lo_frequency": omega_LO_q,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
        ],
        "mixer_ATS": [
            {
                "intermediate_frequency": omega_ATS,
                "lo_frequency": omega_LO_ATS,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
            for omega_ATS in ATS_frequencies
        ],
    },
}
