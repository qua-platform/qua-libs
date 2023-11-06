import numpy as np

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

# AOM
AOM_IF = 0
const_amplitude = 0.1
const_len = 100

# Noise
noise_IF = 20e3
noise_amplitude = 0.04
noise_len = 100

# Photo-diode
readout_len = 100
time_of_flight = 192


config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                9: {"offset": 0.0},
                2: {"offset": 0.0},
            },
            "digital_outputs": {
                1: {},
            },
            "analog_inputs": {
                1: {"offset": 0.0},
            },
        }
    },
    "elements": {
        "AOM": {
            "singleInput": {
                "port": ("con1", 9),
            },
            "intermediate_frequency": AOM_IF,
            "operations": {
                "cw": "cw_pulse",
            },
        },
        "noise": {
            "singleInput": {
                "port": ("con1", 9),
            },
            "intermediate_frequency": noise_IF,
            "operations": {
                "cw": "noise_pulse",
            },
        },
        "photo-diode": {
            "singleInput": {
                "port": ("con1", 2),
            },
            "operations": {
                "readout": "readout_pulse",
            },
            "digitalInputs": {
                "trigger": {
                    "port": ("con1", 1),
                    "delay": 136,
                    "buffer": 0,
                }
            },
            "outputs": {
                "out1": ("con1", 1),
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
        },
    },
    "pulses": {
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
        "readout_pulse": {
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
