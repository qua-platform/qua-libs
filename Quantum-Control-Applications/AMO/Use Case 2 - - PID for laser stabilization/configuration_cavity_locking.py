import numpy as np
from qualang_tools.units import unit

u = unit()

######################
# AUXILIARY FUNCTIONS:
######################


def gauss(amplitude, mu, sigma, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma**2))
    return [float(x) for x in gauss_wave]


######################
# Network parameters #
######################
qop_ip = "192.168.0.129"  # Write the QM router IP address
cluster_name = "Cluster_Bordeaux"  # Write your cluster_name if version >= QOP220
qop_port = None  # Write the QOP port if version < QOP220


################
# CONFIGURATION:
################
# Phase modulation
phase_modulation_IF = 5 * u.MHz
phase_mod_amplitude = 0.02
phase_mod_len = 10 * u.us

# AOM
AOM_IF = 0
const_amplitude = 0.1
const_len = 100

# Noise
noise_IF = 20e3
noise_amplitude = 0.04
noise_len = 100

# Photo-diode
readout_len = phase_mod_len
time_of_flight = 192


config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                6: {"offset": 0.0},
                7: {"offset": 0.0},
                8: {"offset": 0.0},
            },
            "digital_outputs": {
                10: {},
            },
            "analog_inputs": {
                2: {"offset": 0.0},
            },
        }
    },
    "elements": {
        "phase_modulator": {
            "singleInput": {
                "port": ("con1", 6),
            },
            "intermediate_frequency": phase_modulation_IF,
            "operations": {
                "cw": "phase_mod_pulse",
            },
        },
        "filter_cavity_1": {
            "singleInput": {
                "port": ("con1", 6),
            },
        },
        "filter_cavity_2": {
            "singleInput": {
                "port": ("con1", 7),
            },
        },
        "filter_cavity_3": {
            "singleInput": {
                "port": ("con1", 8),
            },
        },
        "detector_DC": {
            "singleInput": {
                "port": ("con1", 6),
            },
            "operations": {
                "readout": "DC_readout_pulse",
            },
            "outputs": {
                "out1": ("con1", 2),
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
        },
        "detector_AC": {
            "singleInput": {
                "port": ("con1", 6),
            },
            "intermediate_frequency": phase_modulation_IF,
            "operations": {
                "readout": "AC_readout_pulse",
            },
            "outputs": {
                "out1": ("con1", 2),
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
        },
    },
    "pulses": {
        "phase_mod_pulse": {
            "operation": "control",
            "length": phase_mod_len,
            "waveforms": {
                "single": "phase_mod_wf",
            },
        },
        "cw_pulse": {
            "operation": "control",
            "length": const_len,
            "waveforms": {
                "single": "const_wf",
            },
        },
        "noise_pulse": {
            "operation": "control",
            "length": noise_len,
            "waveforms": {
                "single": "noise_wf",
            },
        },
        "DC_readout_pulse": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {
                "single": "zero_wf",
            },
            "integration_weights": {
                "constant": "constant_weights",
            },
            "digital_marker": "ON",
        },
        "AC_readout_pulse": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {
                "single": "zero_wf",
            },
            "integration_weights": {
                "constant": "constant_weights",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "phase_mod_wf": {"type": "constant", "sample": phase_mod_amplitude},
        "const_wf": {"type": "constant", "sample": const_amplitude},
        "noise_wf": {"type": "constant", "sample": noise_amplitude},
        "zero_wf": {"type": "constant", "sample": 0.0},
    },
    "digital_waveforms": {"ON": {"samples": [(1, 0)]}},
    "integration_weights": {
        "constant_weights": {
            "cosine": [(1.0, readout_len)],
            "sine": [(0.0, readout_len)],
        },
    },
}
