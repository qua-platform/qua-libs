import numpy as np


def gauss(amplitude, mu, sigma, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma**2))
    return [float(x) for x in gauss_wave]


def IQ_imbalance_corr(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


resonator_TOF = 332
readout_pulse_length = 500
load_pulse_length = 172
resonator_freq = 6.6e9
drive_freq = 5e9
Tpihalf = 32

resonator_IF = 50e6
resonator_LO = resonator_freq - resonator_IF

drive_IF = 31.25e6 * 0
drive_LO = drive_freq - drive_IF

readout_amp = 0.1  # meas pulse amplitude
loadpulse_amp = 0.3  # prepulse to fast load cavity /!\ <0.5 V

drive_gauss_pulse_length = Tpihalf
drive_amp = 0.1
pi_half_amp = 0.1
pi_half_mu = 0
pi_half_sigma = drive_gauss_pulse_length / 6

resonator_I0 = 0.0
resonator_Q0 = 0.0
resonator_g = 0.0
resonator_phi = 0.0

drive_I0 = 0.0
drive_Q0 = 0.0
drive_g = 0.0
drive_phi = 0.0

resonator_correction_matrix = IQ_imbalance_corr(resonator_g, resonator_phi)
drive_correction_matrix = IQ_imbalance_corr(drive_g, drive_phi)

Input1_offset = 0.0
Input2_offset = 0.0

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": resonator_I0},  # resonator I
                2: {"offset": resonator_Q0},  # resonator Q
                3: {"offset": drive_I0},  # drive I
                4: {"offset": drive_Q0},  # drive Q
                5: {"offset": 0},
            },
            "digital_outputs": {
                1: {},  # resonator digital marker
                2: {},  # drive digital marker
                3: {},  # drive LO synthesizer trigger
                4: {},  # drive LO microwave source trigger --------------------------
            },
            "analog_inputs": {
                1: {"offset": Input1_offset},  # I readout from resonator demod
                2: {"offset": Input2_offset},  # Q readout from resonator demod
            },
        },
    },
    "elements": {
        "resonator": {  # resonator element
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": resonator_LO,
                "mixer": "resonator_mixer",
            },
            "intermediate_frequency": resonator_IF,
            "operations": {
                "readout": "readout_pulse",  # play('readout', 'resonator'),
                "chargecav": "load_pulse",
            },
            "digitalInputs": {
                "switchR": {
                    "port": ("con1", 1),
                    "delay": 144,
                    "buffer": 0,
                }
            },
            "outputs": {
                "out1": ("con1", 1),
                "out2": ("con1", 2),
            },
            "time_of_flight": resonator_TOF,
            "smearing": 0,
        },
        "drive": {  # drive element
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": drive_LO,
                "mixer": "drive_mixer",
            },
            "digitalInputs": {
                "switchE": {
                    "port": ("con1", 3),
                    "delay": 144,
                    "buffer": 0,
                }
            },
            "intermediate_frequency": drive_IF,
            "operations": {
                "pi_half": "pi_half_pulse",
            },
        },
    },
    "pulses": {
        "readout_pulse": {
            "operation": "measurement",
            "length": readout_pulse_length,
            "waveforms": {
                "I": "readout_wf",
                "Q": "zero_wf",
            },
            "integration_weights": {
                "cos": "integW_cosine",
                "sin": "integW_sine",
            },
            "digital_marker": "ON",  # Put ON instead of Modulate if you want raw adc time traces
        },
        "load_pulse": {
            "operation": "control",
            "length": load_pulse_length,
            "waveforms": {
                "I": "loadpulse_wf",
                "Q": "zero_wf",
            },
            "digital_marker": "ON",
        },
        "pi_half_pulse": {
            "operation": "control",
            "length": drive_gauss_pulse_length,
            "waveforms": {
                "I": "pi_half_wf",
                "Q": "zero_wf",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "readout_wf": {
            "type": "constant",
            "sample": readout_amp,
        },
        "loadpulse_wf": {
            "type": "constant",
            "sample": loadpulse_amp,
        },
        "zero_wf": {
            "type": "constant",
            "sample": 0.0,
        },
        "drive_wf": {
            "type": "constant",
            "sample": drive_amp,
        },
        "pi_half_wf": {
            "type": "arbitrary",
            "samples": gauss(
                pi_half_amp,
                pi_half_mu,
                pi_half_sigma,
                drive_gauss_pulse_length,
            ),
        },
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
        "OFF": {"samples": [(0, 0)]},  # [(value, length)]
        "Modulate": {
            "samples": [(1, readout_pulse_length / 20), (0, readout_pulse_length / 20)] * 10  # [(value, length)]
        },
    },
    "integration_weights": {
        "integW_cosine": {
            "cosine": [1.0] * int(readout_pulse_length / 4),
            "sine": [0.0] * int(readout_pulse_length / 4),
        },
        "integW_sine": {
            "cosine": [0.0] * int(readout_pulse_length / 4),
            "sine": [1.0] * int(readout_pulse_length / 4),
        },
    },
    "mixers": {
        "resonator_mixer": [
            {
                "intermediate_frequency": resonator_IF,
                "lo_frequency": resonator_LO,
                "correction": resonator_correction_matrix,
            }
        ],
        "drive_mixer": [
            {
                "intermediate_frequency": drive_IF,
                "lo_frequency": drive_LO,
                "correction": drive_correction_matrix,
            }
        ],
    },
}
