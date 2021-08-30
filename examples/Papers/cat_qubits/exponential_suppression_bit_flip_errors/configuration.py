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
        * (-(t - mu) / sigma ** 2)
        * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
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

dis_amp = 0.4
dis_std = 20
dis_mean = 0
dis_duration = 80
dis_detuning = 0

displace_wf = gauss(dis_amp, dis_mean, dis_std, dis_detuning, dis_duration)

probe_amp = 0.4
probe_std = 20
probe_mean = 0
probe_duration = 1000
probe_detuning = 0

probe_wf = gauss(dis_amp, dis_mean, dis_std, dis_detuning, dis_duration)
omega_LO_q = 1e9
omega_LO_b = 1e9
omega_LO_a = 1e9
omega_LO_ATS = 1e9
omega_LO_RR = 1e9
omega_a = 70e6
omega_b = 40e6
omega_p = 2 * omega_a - omega_b
omega_RR = 50e6
omega_q = 60e6
chi_qa = 0.023  # Coupling constant between transmon and cat qubit cavity

deflation_duration = 200

optimal_go_to_g_duration_storage = 200
optimal_go_to_g_duration_transmon = 200

readout_duration = 200
pump_duration = 400
drive_duration = 400

const_high = 0.48
const_low = 0.2

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},  # Transmon I
                2: {"offset": +0.0},  # Transmon Q
                3: {"offset": +0.0},  # Readout resonator I
                4: {"offset": +0.0},  # Readout resonator Q
                5: {"offset": +0.0},  # fluxline transmon
                6: {"offset": +0.0},
                7: {"offset": +0.0},  # storage drive I
                8: {"offset": +0.0},  # storage drive Q
                9: {"offset": +0.0},  # ATS pump I
                10: {"offset": +0.0},  # ATS pump Q
            },
            "analog_inputs": {
                1: {"offset": +0.0},  # Input of RR
                2: {"offset": +0.0},  # Input of RR
            },
        },
        "con2": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},  # Buffer drive I
                2: {"offset": +0.0},  # Buffer drive Q
            },
            "analog_inputs": {
                1: {"offset": +0.0},  # Buffer drive readout I
                2: {"offset": +0.0},  # Buffer drive readout Q
            },
        },
    },
    "elements": {
        "transmon": {
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
                "g1_to_g0_opt_con": "g1_to_g0_opt_con_pulse_transmon",
                "e1_to_g0_opt_con": "e1_to_g0_opt_con_pulse_transmon",
            },
        },
        "RR": {  # Readout resonator, connected to transmon
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
        "buffer": {
            "mixInputs": {
                "I": ("con2", 1),
                "Q": ("con2", 2),
                "lo_frequency": omega_LO_b,
                "mixer": "mixer_buffer",
            },
            "intermediate_frequency": omega_b,
            "operations": {"drive": "drive_pulse", "deflation": "deflation_pulse"},
            "time_of_flight": 180,  # Measurement parameters
            "smearing": 0,
            "outputs": {"out3": ("con2", 1), "out4": ("con2", 2)},
        },
        "storage": {
            "mixInputs": {
                "I": ("con1", 7),
                "Q": ("con1", 8),
                "lo_frequency": omega_LO_a,
                "mixer": "mixer_cat",
            },
            "intermediate_frequency": omega_a,
            "operations": {
                "Displace_Op": "Displace_pulse",
                "linear": "linear_pulse",
                "g1_to_g0_opt_con": "g1_to_g0_opt_con_pulse_storage",
                "e1_to_g0_opt_con": "e1_to_g0_opt_con_pulse_storage",
                "reset_fock_0": "reset_fock_0",
                "reset_fock_1": "reset_fock_1",
                "reset_fock_2": "reset_fock_2",
                "reset_fock_3": "reset_fock_3",
                "reset_fock_4": "reset_fock_4",
                "reset_fock_5": "reset_fock_5",
                "reset_fock_6": "reset_fock_6",
                "reset_fock_7": "reset_fock_7",
            },
        },
        "ATS": {
            "mixInputs": {
                "I": ("con1", 9),
                "Q": ("con1", 10),
                "lo_frequency": omega_LO_ATS,
                "mixer": "mixer_ATS",
            },
            "intermediate_frequency": omega_p,
            "operations": {
                "pump": "Pump_pulse",
                "deflation": "deflation_pulse",
                "inflation": "inflation_pulse",
                "constOp": "constPulseIQ",
            },
        },
        "ATS_linear_drive": {
            "mixInputs": {
                "I": ("con1", 9),
                "Q": ("con1", 10),
                "lo_frequency": omega_LO_ATS,
                "mixer": "mixer_ATS2",
            },
            "intermediate_frequency": omega_a,
            "operations": {
                "linear": "linear_pulse",
                "deflation": "deflation_pulse",
                "inflation": "inflation_pulse",
                "constOp": "constPulseIQ",
            },
        },
        "fluxline": {
            "singleInput": {
                "port": ("con1", 5),
            },
            "intermediate_frequency": 0,
            "operations": {"constOp": "constPulse"},
        },
    },
    "pulses": {
        "readout_pulse": {  # Readout pulse for readout resonator coupled to transmon
            "operation": "measurement",
            "length": readout_duration,
            "waveforms": {
                "I": "const_wf_high",  # Decide what pulse to apply for each component
                "Q": "zero_wf",
            },
            "integration_weights": {
                "integW_cos": "integW_cos",
                "integW_sin": "integW_sin",
                "optimal_integW_1": "optimal_integW_1",
                "optimal_integW_2": "optimal_integW_2",
                "optimal_integW_1_minus-sin": "integW_minus_sin",
            },
            "digital_marker": "marker1",
        },
        "X_pulse": {
            "operation": "control",
            "length": SQduration,
            "waveforms": {"I": "x_wf", "Q": "x_der_wf"},
        },
        "X90_pulse": {
            "operation": "control",
            "length": SQduration,
            "waveforms": {"I": "x90_wf", "Q": "x90_der_wf"},
        },
        "Pump_pulse": {
            "operation": "control",
            "length": pump_duration,
            "waveforms": {"I": "const_wf_low", "Q": "zero_wf"},
        },
        "linear_pulse": {
            "operation": "control",
            "length": 80,
            "waveforms": {"I": "const_wf_low", "Q": "zero_wf"},
        },
        "Displace_pulse": {
            "operation": "control",
            "length": 80,
            "waveforms": {"I": "disp_wf", "Q": "disp_wf"},
        },
        "reset_fock_5": {
            "operation": "control",
            "length": 80,
            "waveforms": {"I": "disp_wf", "Q": "disp_wf"},
        },
        "reset_fock_6": {
            "operation": "control",
            "length": 80,
            "waveforms": {"I": "disp_wf", "Q": "disp_wf"},
        },
        "reset_fock_7": {
            "operation": "control",
            "length": 80,
            "waveforms": {"I": "disp_wf", "Q": "disp_wf"},
        },
        "reset_fock_0": {
            "operation": "control",
            "length": 80,
            "waveforms": {"I": "disp_wf", "Q": "disp_wf"},
        },
        "reset_fock_1": {
            "operation": "control",
            "length": 80,
            "waveforms": {"I": "disp_wf", "Q": "disp_wf"},
        },
        "reset_fock_2": {
            "operation": "control",
            "length": 80,
            "waveforms": {"I": "disp_wf", "Q": "disp_wf"},
        },
        "reset_fock_3": {
            "operation": "control",
            "length": 80,
            "waveforms": {"I": "disp_wf", "Q": "disp_wf"},
        },
        "reset_fock_4": {
            "operation": "control",
            "length": 80,
            "waveforms": {"I": "disp_wf", "Q": "disp_wf"},
        },
        "deflation_pulse": {
            "operation": "control",
            "length": deflation_duration,
            "waveforms": {
                "I": "tanh_down_wf",  # Decide what pulse to apply for each component
                "Q": "zero_wf",
            },
        },
        "inflation_pulse": {
            "operation": "control",
            "length": deflation_duration,
            "waveforms": {
                "I": "tanh_up_wf",  # Decide what pulse to apply for each component
                "Q": "zero_wf",
            },
        },
        "drive_pulse": {
            "operation": "measurement",
            "length": drive_duration,
            "waveforms": {
                "I": "const_wf_high",  # Decide what pulse to apply for each component
                "Q": "zero_wf",
            },
            "integration_weights": {
                "integW_cos": "integW_cos",
                "integW_sin": "integW_sin",
                "integW_minus_sin": "integW_minus_sin",
                "optimal_integW_1": "optimal_integW_1",
                "optimal_integW_2": "optimal_integW_2",
            },
            "digital_marker": "marker1",
        },
        "constPulse": {
            "operation": "control",
            "length": 1000,
            "waveforms": {"single": "const_wf_low"},
        },
        "constPulseIQ": {
            "operation": "control",
            "length": 1000,
            "waveforms": {"I": "const_wf_low", "Q": "zero_wf"},
        },
        "g1_to_g0_opt_con_pulse_storage": {
            "operation": "control",
            "length": optimal_go_to_g_duration_storage,
            "waveforms": {
                "I": "g1_to_g0_opt_con_storage_wf_I",
                "Q": "g1_to_g0_opt_con_storage_wf_Q",
            },
        },
        "g1_to_g0_opt_con_pulse_transmon": {
            "operation": "control",
            "length": optimal_go_to_g_duration_transmon,
            "waveforms": {
                "I": "g1_to_g0_opt_con_transmon_wf_I",
                "Q": "g1_to_g0_opt_con_transmon_wf_Q",
            },
        },
        "e1_to_g0_opt_con_pulse_storage": {
            "operation": "control",
            "length": optimal_go_to_g_duration_storage,
            "waveforms": {
                "I": "e1_to_g0_opt_con_storage_wf_I",
                "Q": "e1_to_g0_opt_con_storage_wf_Q",
            },
        },
        "e1_to_g0_opt_con_pulse_transmon": {
            "operation": "control",
            "length": optimal_go_to_g_duration_transmon,
            "waveforms": {
                "I": "e1_to_g0_opt_con_transmon_wf_I",
                "Q": "e1_to_g0_opt_con_transmon_wf_Q",
            },
        },
    },
    "waveforms": {
        "zero_wf": {"type": "constant", "sample": 0.0},
        "const_wf_low": {"type": "constant", "sample": const_low},
        "const_wf_high": {"type": "constant", "sample": const_high},
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
        "disp_wf": {
            "type": "arbitrary",
            "samples": gauss(dis_amp, dis_mean, dis_std, dis_detuning, dis_duration),
        },
        "tanh_down_wf": {
            "type": "arbitrary",
            "samples": [
                const_high / 2 * np.tanh(-x) + const_high / 2
                for x in np.linspace(-3, 3, deflation_duration)
            ],
        },
        "tanh_up_wf": {
            "type": "arbitrary",
            "samples": [
                const_high / 2 * np.tanh(x) + const_high / 2
                for x in np.linspace(-3, 3, deflation_duration)
            ],
        },
        "g1_to_g0_opt_con_storage_wf_I": {
            "type": "arbitrary",
            "samples": gauss(
                dis_amp * 0.4,
                dis_mean,
                dis_std,
                dis_detuning,
                optimal_go_to_g_duration_storage,
            ),
        },
        "g1_to_g0_opt_con_transmon_wf_I": {
            "type": "arbitrary",
            "samples": gauss(
                dis_amp * 0.4,
                dis_mean,
                dis_std,
                dis_detuning,
                optimal_go_to_g_duration_transmon,
            ),
        },
        "g1_to_g0_opt_con_storage_wf_Q": {
            "type": "arbitrary",
            "samples": gauss(
                dis_amp * 0.4,
                dis_mean,
                dis_std,
                dis_detuning,
                optimal_go_to_g_duration_storage,
            ),
        },
        "g1_to_g0_opt_con_transmon_wf_Q": {
            "type": "arbitrary",
            "samples": gauss(
                dis_amp * 0.4,
                dis_mean,
                dis_std,
                dis_detuning,
                optimal_go_to_g_duration_transmon,
            ),
        },
        "e1_to_g0_opt_con_storage_wf_I": {
            "type": "arbitrary",
            "samples": gauss(
                dis_amp * 0.4,
                dis_mean,
                dis_std,
                dis_detuning,
                optimal_go_to_g_duration_storage,
            ),
        },
        "e1_to_g0_opt_con_transmon_wf_I": {
            "type": "arbitrary",
            "samples": gauss(
                dis_amp * 0.4,
                dis_mean,
                dis_std,
                dis_detuning,
                optimal_go_to_g_duration_transmon,
            ),
        },
        "e1_to_g0_opt_con_storage_wf_Q": {
            "type": "arbitrary",
            "samples": gauss(
                dis_amp * 0.4,
                dis_mean,
                dis_std,
                dis_detuning,
                optimal_go_to_g_duration_storage,
            ),
        },
        "e1_to_g0_opt_con_transmon_wf_Q": {
            "type": "arbitrary",
            "samples": gauss(
                dis_amp * 0.4,
                dis_mean,
                dis_std,
                dis_detuning,
                optimal_go_to_g_duration_transmon,
            ),
        },
    },
    "digital_waveforms": {"marker1": {"samples": [(1, 0)]}},
    "integration_weights": {
        "optimal_integW_1": {
            "cosine": [1.0] * (readout_duration // 4),
            "sine": [0.0] * (readout_duration // 4),
        },
        "optimal_integW_2": {
            "cosine": [0.0] * (readout_duration // 4),
            "sine": [-1.0] * (readout_duration // 4),
        },
        "integW_cos": {
            "cosine": [1.0] * (readout_duration // 4),
            "sine": [0.0] * (readout_duration // 4),
        },
        "integW_sin": {
            "cosine": [0.0] * (readout_duration // 4),
            "sine": [1.0] * (readout_duration // 4),
        },
        "integW_minus_sin": {
            "cosine": [0.0] * (readout_duration // 4),
            "sine": [-1.0] * (readout_duration // 4),
        },
    },
    "mixers": {  # Potential corrections to be brought related to the IQ mixing scheme
        "mixer_res": [
            {
                "intermediate_frequency": omega_RR,
                "lo_frequency": omega_LO_RR,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
        ],
        "mixer_buffer": [
            {
                "intermediate_frequency": omega_b,
                "lo_frequency": omega_LO_b,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
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
                "intermediate_frequency": omega_p,
                "lo_frequency": omega_LO_ATS,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
        ],
        "mixer_ATS2": [
            {
                "intermediate_frequency": omega_a,
                "lo_frequency": omega_LO_ATS,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
        ],
        "mixer_cat": [
            {
                "intermediate_frequency": omega_a,
                "lo_frequency": omega_LO_a,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
        ],
    },
}
