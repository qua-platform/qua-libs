import numpy as np
from scipy import signal

simulate = False

# Frequencies
qubit_freq = 80e6  # in units of Hz
aux_freq = 200e6  # in units of Hz
LO_freq = 1e9  # in units of Hz
LO_aux_freq = 1e9  # in units of Hz

# Pulses lengths
charge_init_len = 300  # in ns
spin_init_len = 300  # in ns
long_spin_init_len = 3000  # in ns
EX_len = 3000  # in ns
SCC_len = 300  # in ns
ionization_len = 300  # in ns
charge_read_len = 300  # in ns
PL_len = 300  # in ns
meas_len = 5000  # in ns
full_meas_len = 200  # in ns

# MW parameters
mw_amp = 0.4  # in units of volts
mw_len = 500  # in units of ns

# Pi pulse parameters
pi_len = 60  # in units of ns
pi_amp = 0.3  # in units of volts
pi_wf = (
    pi_amp * (signal.windows.gaussian(pi_len, pi_len / 5) - signal.windows.gaussian(pi_len, pi_len / 5)[-1])
).tolist()  # waveform

# Pi_half pulse parameters
pi_half_len = 60  # in units of ns
pi_half_amp = 0.15  # in units of volts
pi_half_wf = (
    pi_half_amp
    * (
        signal.windows.gaussian(pi_half_len, pi_half_len / 5)
        - signal.windows.gaussian(pi_half_len, pi_half_len / 5)[-1]
    )
).tolist()  # waveform

# Gaussian pulse parameters
gauss_amp = 0.3  # The gaussian is used when calibrating pi and pi_half pulses
gauss_len = 20  # The gaussian is used when calibrating pi and pi_half pulses
gauss_wf = (
    gauss_amp
    * (signal.windows.gaussian(gauss_len, gauss_len / 5) - signal.windows.gaussian(gauss_len, gauss_len / 5)[-1])
).tolist()  # waveform

# Pi_aux pulse parameters
pi_aux_len = 60  # in units of volts
pi_aux_amp = 0.3  # in units of ns
pi_aux_wf = (
    pi_amp * (signal.windows.gaussian(pi_len, pi_len / 5) - signal.windows.gaussian(pi_len, pi_len / 5)[-1])
).tolist()  # waveform

# Integration weights
optimized_weights = [1.0] * int(304)


def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    n = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(n * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": 0.0},  # qubit I
                2: {"offset": 0.0},  # qubit Q
                3: {"offset": 0.0},  # qubit aux I
                4: {"offset": 0.0},  # qubit aux Q
                5: {"offset": 0.0},  # Electric field to qubit
            },
            "digital_outputs": {
                1: {},  # 705 nm AOM
                2: {},  # E12 AOM
                3: {},  # EX AOM
                4: {},  # 1151 nm AOM
                5: {},  # SNSPD marker
            },
            "analog_inputs": {
                1: {"offset": 0},  # SNSPD
            },
        }
    },
    "elements": {
        "qubit": {
            "thread": "a",
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": LO_freq,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": qubit_freq,
            "operations": {
                "mw": "cw_pulse",
                "gauss": "gauss_pulse",
                "pi": "pi_pulse",
                "pi_half": "pi_half_pulse",
            },
        },  # shares threads with 'qubit_aux' (uses 2)
        "qubit_aux": {
            "thread": "a",
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": LO_aux_freq,
                "mixer": "mixer_qubit_aux",
            },
            "intermediate_frequency": aux_freq,
            "operations": {
                "gauss": "gauss_pulse",
                "pi_aux": "pi_aux_pulse",
            },
        },  # shares threads with 'qubit' (uses 2)
        "SNSPD": {
            "thread": "b",
            # blanked input
            "singleInput": {"port": ("con1", 5)},
            #
            "digitalInputs": {
                "marker": {
                    "port": ("con1", 5),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "photon_count": "photon_count_pulse",
                "full_readout": "full_readout_pulse",
            },
            "outputs": {"out1": ("con1", 1)},
            "time_of_flight": 24,
            "smearing": 0,
            "outputPulseParameters": {
                "signalThreshold": 300,
                "signalPolarity": "Descending",
                "derivativeThreshold": 50,
                "derivativePolarity": "Descending",
            },
        },  # has its own unique thread
        "E_field": {
            "thread": "c",
            "singleInput": {"port": ("con1", 5)},
            "operations": {},
        },  # has its own unique thread
        "laser_705nm": {
            "thread": "d",
            "digitalInputs": {
                "705nm_AOM": {
                    "port": ("con1", 1),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "charge_init": "charge_init_pulse",
            },
        },  # shares the thread with other lasers
        "laser_E12": {
            "thread": "d",
            "digitalInputs": {
                "E12_AOM": {
                    "port": ("con1", 2),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "spin_init": "spin_init_pulse",
                "long_spin_init": "long_spin_init_pulse",
            },
        },  # shares the thread with other lasers
        "laser_EX": {
            "thread": "d",
            "digitalInputs": {
                "EX_AOM": {
                    "port": ("con1", 3),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "EX": "EX_pulse",
            },
        },  # shares the thread with other lasers
        "laser_EX1151nm": {
            "thread": "d",
            "digitalInputs": {
                "EX_AOM": {
                    "port": ("con1", 3),
                    "delay": 0,
                    "buffer": 0,
                },
                "1151nm_AOM": {
                    "port": ("con1", 4),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "SCC": "SCC_pulse",
            },
        },  # shares the thread with other lasers
        "laser_EX121151nm": {
            "thread": "d",
            "digitalInputs": {
                "EX_AOM": {
                    "port": ("con1", 3),
                    "delay": 0,
                    "buffer": 0,
                },
                "E12_AOM": {
                    "port": ("con1", 2),
                    "delay": 0,
                    "buffer": 0,
                },
                "1151nm_AOM": {
                    "port": ("con1", 4),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "ionization": "ionization_pulse",
            },
        },  # shares the thread with other lasers
        "laser_EX12": {
            "thread": "d",
            "digitalInputs": {
                "EX_AOM": {
                    "port": ("con1", 3),
                    "delay": 0,
                    "buffer": 0,
                },
                "E12_AOM": {
                    "port": ("con1", 2),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "charge_read": "charge_read_pulse",
            },
        },  # shares the thread with other lasers
        "laser_EX705nm": {
            "thread": "d",
            "digitalInputs": {
                "EX_AOM": {
                    "port": ("con1", 3),
                    "delay": 0,
                    "buffer": 0,
                },
                "705nm_AOM": {
                    "port": ("con1", 1),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "PL": "PL_pulse",
            },
        },  # shares the thread with other lasers
    },
    "pulses": {
        "cw_pulse": {
            "operation": "control",
            "length": mw_len,
            "waveforms": {"I": "const_wf", "Q": "zero_wf"},
        },
        "gauss_pulse": {
            "operation": "control",
            "length": gauss_len,
            "waveforms": {"I": "gauss_wf", "Q": "zero_wf"},
        },
        "pi_pulse": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {"I": "pi_wf", "Q": "zero_wf"},
        },
        "pi_half_pulse": {
            "operation": "control",
            "length": pi_half_len,
            "waveforms": {"I": "pi_half_wf", "Q": "zero_wf"},
        },
        "pi_aux_pulse": {
            "operation": "control",
            "length": pi_aux_len,
            "waveforms": {"I": "pi_aux_wf", "Q": "zero_wf"},
        },
        "charge_init_pulse": {
            "operation": "control",
            "length": charge_init_len,
            "digital_marker": "ON",
        },
        "spin_init_pulse": {
            "operation": "control",
            "length": spin_init_len,
            "digital_marker": "ON",
        },
        "long_spin_init_pulse": {
            "operation": "control",
            "length": long_spin_init_len,
            "digital_marker": "ON",
        },
        "EX_pulse": {"operation": "control", "length": EX_len, "digital_marker": "ON"},
        "SCC_pulse": {
            "operation": "control",
            "length": SCC_len,
            "digital_marker": "ON",
        },
        "ionization_pulse": {
            "operation": "control",
            "length": ionization_len,
            "digital_marker": "ON",
        },
        "charge_read_pulse": {
            "operation": "control",
            "length": charge_read_len,
            "digital_marker": "ON",
        },
        "PL_pulse": {"operation": "control", "length": PL_len, "digital_marker": "ON"},
        "photon_count_pulse": {
            "operation": "measurement",
            "length": meas_len,
            "waveforms": {"single": "zero_wf"},
            "digital_marker": "ON",
        },
        "full_readout_pulse": {
            "operation": "measurement",
            "length": full_meas_len,
            "waveforms": {"single": "zero_wf"},
            "integration_weights": {
                "300ns_integration_weights": "300ns_integration_weights",
                "optimized_integration_weights": "optimized_integration_weights",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": mw_amp},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "gauss_wf": {"type": "arbitrary", "samples": gauss_wf},
        "pi_wf": {"type": "arbitrary", "samples": pi_wf},
        "pi_half_wf": {"type": "arbitrary", "samples": pi_half_wf},
        "pi_aux_wf": {"type": "arbitrary", "samples": pi_aux_wf},
    },
    "digital_waveforms": {"ON": {"samples": [(1, 0)]}},  # [(on/off, ns)]
    "integration_weights": {
        "300ns_integration_weights": {
            "cosine": [(1.0, 300)],
            "sine": [(0.0, 300)],
        },
        "optimized_integration_weights": {
            "cosine": [(1.0, len(optimized_weights))],
            "sine": [(0.0, len(optimized_weights))],
        },
    },
    "mixers": {
        "mixer_qubit": [
            {
                "intermediate_frequency": qubit_freq,
                "lo_frequency": LO_freq,
                "correction": IQ_imbalance(0, 0),
            },
        ],
        "mixer_qubit_aux": [
            {
                "intermediate_frequency": aux_freq,
                "lo_frequency": LO_aux_freq,
                "correction": IQ_imbalance(0, 0),
            },
        ],
    },
}
