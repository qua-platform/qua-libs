from scipy.signal.windows import gaussian
import numpy as np

# Used to correct for IQ mixer imbalances
def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


# Frequencies
resonator_if = -300e6  # in Hz
ensemble_if = -300e6  # in Hz
# ensemble_if = 37e6  # in Hz

# LOs are used in plots. They can also be used marking mixer elements in the config.
# On top of that they can also be used for setting LO sources (this make sure that everything is in sync)
ensemble_LO = 2.8e9  # in Hz

# Readout parameters
const_amp = 0.18  # in V
const_len = 360  # in ns

short_readout_len = 80  # in ns
short_readout_amp = 0.4  # in V

readout_len = 1000  # in ns
readout_amp = 0.4  # in V

long_readout_len = 70000  # in ns
long_readout_amp = 0.1  # in V

time_of_flight = 160  # Time it takes the pulses to go through the RF chain, including the device.

# Qubit parameters:
saturation_amp = 0.2  # in V
saturation_len = 50000  # Needs to be several T1 so that the final state is an equal population of |0> and |1>

# Pi pulse parameters
pi_len = 60  # in units of ns
pi_amp = 0.3  # in units of volts
pi_wf = (pi_amp * (gaussian(pi_len, pi_len / 5) - gaussian(pi_len, pi_len / 5)[-1])).tolist()  # waveform
minus_pi_wf = ((-1) * pi_amp * (gaussian(pi_len, pi_len / 5) - gaussian(pi_len, pi_len / 5)[-1])).tolist()  # waveform

# Pi_half pulse parameters
pi_half_len = 60  # in units of ns
pi_half_amp = 0.15  # in units of volts
pi_half_wf = (
    pi_half_amp * (gaussian(pi_half_len, pi_half_len / 5) - gaussian(pi_half_len, pi_half_len / 5)[-1])
).tolist()  # waveform
minus_pi_half_wf = (
    (-1) * pi_half_amp * (gaussian(pi_half_len, pi_half_len / 5) - gaussian(pi_half_len, pi_half_len / 5)[-1])
).tolist()  # waveform

# Subtracted Gaussian pulse parameters
gauss_amp = 0.3  # The gaussian is used when calibrating pi and pi_half pulses
gauss_len = 20  # The gaussian is used when calibrating pi and pi_half pulses
gauss_wf = (
    gauss_amp * (gaussian(gauss_len, gauss_len / 5) - gaussian(gauss_len, gauss_len / 5)[-1])
).tolist()  # waveform

# Note: a subtracted Gaussian pulse has a more narrow spectral density than a regular gaussian
# it becomes useful in short pulses to reduce leakage to higher energy states

# Flux:
square_flux_amp = 0.3
minus_square_flux_amp = -0.3
triangle_flux_amp = 0.3
triangle_wf = [triangle_flux_amp * i / 7 for i in range(8)] + [triangle_flux_amp * (1 - i / 7) for i in range(8)]

initialization_len = 1000  # in ns
activation_len = 500  # in ns

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": -0.0185},  # qubit I
                2: {"offset": +0.005},  # qubit Q
                3: {"offset": +0.0},  # Resonator I
                4: {"offset": +0.0},  # Resonator Q
                5: {"offset": +0.0},  # Slow Flux
                6: {"offset": +0.0},  # Slow Flux
            },
            "digital_outputs": {
                1: {},
                2: {},
                3: {},
                4: {},
                5: {},
                10: {},
            },
            "analog_inputs": {
                1: {"offset": 0.06478248046875001, "gain_db": 14},  # I from down conversion
                2: {"offset": 0.22755350048828127, "gain_db": 14},  # Q from down conversion
            },
        },
    },
    "elements": {
        "ensemble": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": ensemble_LO,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": ensemble_if,
            "operations": {
                "const": "const_pulse",
                "saturation": "saturation_pulse",
                "gaussian": "gaussian_pulse",
                "pi": "pi_pulse",
                "pi_half": "pi_half_pulse",
                "X": "Xpi_pulse",
                "-X": "-Xpi_pulse",
                "X/2": "Xpi_half_pulse",
                "-X/2": "-Xpi_half_pulse",
                "Y": "Ypi_pulse",
                "-Y": "-Ypi_pulse",
                "Y/2": "Ypi_half_pulse",
                "-Y/2": "-Ypi_half_pulse",
            },
        },
        "ensemble2": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": ensemble_LO,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": ensemble_if,
            "operations": {
                "const": "const_pulse",
                "saturation": "saturation_pulse",
                "gaussian": "gaussian_pulse",
                "pi": "pi_pulse",
                "pi_half": "pi_half_pulse",
                "X": "Xpi_pulse",
                "-X": "-Xpi_pulse",
                "X/2": "Xpi_half_pulse",
                "-X/2": "-Xpi_half_pulse",
                "Y": "Ypi_pulse",
                "-Y": "-Ypi_pulse",
                "Y/2": "Ypi_half_pulse",
                "-Y/2": "-Ypi_half_pulse",
            },
        },
        "resonator": {
            "singleInput": {"port": ("con1", 6)},
            "intermediate_frequency": resonator_if,
            "operations": {
                "short_readout": "short_readout_pulse",
                "readout": "readout_pulse",
                "long_readout": "long_readout_pulse",
            },
            "digitalInputs": {
                "laser": {
                    "port": ("con1", 10),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
            "outputs": {
                "out1": ("con1", 1),
                "out2": ("con1", 2),
            },
        },
        "oscillator": {
            "singleInput": {"port": ("con1", 5)},
            "operations": {
                "offset": "square_pulse",
                "minus_offset": "minus_square_pulse",
                "triangle": "triangle_pulse",
            },
        },
        "green_laser": {
            "digitalInputs": {
                "laser": {
                    "port": ("con1", 4),
                    "delay": 136,
                    "buffer": 0,
                },
            },
            "operations": {
                "initialization": "initialization_pulse",
            },
        },
        "switch_1": {
            "digitalInputs": {
                "activate": {
                    "port": ("con1", 1),
                    "delay": 136,
                    "buffer": 50,
                },
            },
            "operations": {
                "activate": "activate_pulse",
            },
        },
        "switch_2": {
            "digitalInputs": {
                "activate": {
                    "port": ("con1", 2),
                    "delay": 136,
                    "buffer": 50,
                },
            },
            "operations": {
                "activate": "activate_pulse",
            },
        },
        "switch_receiver": {
            "digitalInputs": {
                "activate": {
                    "port": ("con1", 3),
                    "delay": 136,
                    "buffer": 50,
                },
            },
            "operations": {
                "activate": "activate_pulse",
            },
        },
    },
    "pulses": {
        "initialization_pulse": {
            "operation": "control",
            "length": initialization_len,
            "digital_marker": "ON",
        },
        "activate_pulse": {
            "operation": "control",
            "length": activation_len,
            "digital_marker": "ON",
        },
        "const_pulse": {
            "operation": "control",
            "length": const_len,  # in ns
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
        },
        "saturation_pulse": {
            "operation": "control",
            "length": saturation_len,  # in ns
            "waveforms": {"I": "saturation_wf", "Q": "zero_wf"},
        },
        "gaussian_pulse": {
            "operation": "control",
            "length": gauss_len,  # in ns
            "waveforms": {"I": "gaussian_wf", "Q": "zero_wf"},
        },
        "pi_pulse": {
            "operation": "control",
            "length": pi_len,  # in ns
            "waveforms": {
                "I": "pi_wf",
                "Q": "zero_wf",
            },
        },
        "pi_half_pulse": {
            "operation": "control",
            "length": pi_half_len,  # in ns
            "waveforms": {
                "I": "pi_half_wf",
                "Q": "zero_wf",
            },
        },
        "Xpi_pulse": {
            "operation": "control",
            "length": pi_len,  # in ns
            "waveforms": {
                "I": "pi_wf",
                "Q": "zero_wf",
            },
        },
        "-Xpi_pulse": {
            "operation": "control",
            "length": pi_len,  # in ns
            "waveforms": {
                "I": "-pi_wf",
                "Q": "zero_wf",
            },
        },
        "Xpi_half_pulse": {
            "operation": "control",
            "length": pi_half_len,  # in ns
            "waveforms": {
                "I": "pi_half_wf",
                "Q": "zero_wf",
            },
        },
        "-Xpi_half_pulse": {
            "operation": "control",
            "length": pi_half_len,  # in ns
            "waveforms": {
                "I": "-pi_half_wf",
                "Q": "zero_wf",
            },
        },
        "Ypi_pulse": {
            "operation": "control",
            "length": pi_len,  # in ns
            "waveforms": {
                "I": "zero_wf",
                "Q": "pi_wf",
            },
        },
        "-Ypi_pulse": {
            "operation": "control",
            "length": pi_len,  # in ns
            "waveforms": {
                "I": "zero_wf",
                "Q": "-pi_wf",
            },
        },
        "Ypi_half_pulse": {
            "operation": "control",
            "length": pi_half_len,  # in ns
            "waveforms": {
                "I": "zero_wf",
                "Q": "pi_half_wf",
            },
        },
        "-Ypi_half_pulse": {
            "operation": "control",
            "length": pi_half_len,  # in ns
            "waveforms": {
                "I": "zero_wf",
                "Q": "-pi_half_wf",
            },
        },
        "square_pulse": {"operation": "control", "length": 16, "waveforms": {"single": "square_wf"}},  # in ns
        "minus_square_pulse": {
            "operation": "control",
            "length": 16,  # in ns
            "waveforms": {"single": "minus_square_wf"},
        },
        "triangle_pulse": {"operation": "control", "length": 16, "waveforms": {"single": "triangle_wf"}},  # in ns
        "short_readout_pulse": {
            "operation": "measurement",
            "length": short_readout_len,  # in ns
            "waveforms": {"single": "zero_wf"},
            "digital_marker": "ON",
            "integration_weights": {
                "cos": "short_cos_weights",
                "sin": "short_sin_weights",
                "minus_sin": "short_minus_sin_weights",
            },
        },
        "readout_pulse": {
            "operation": "measurement",
            "length": readout_len,  # in ns
            "waveforms": {"single": "zero_wf"},
            "digital_marker": "ON",
            "integration_weights": {
                "cos": "cos_weights",
                "sin": "sin_weights",
                "minus_sin": "minus_sin_weights",
            },
        },
        "long_readout_pulse": {
            "operation": "measurement",
            "length": long_readout_len,  # in ns
            "waveforms": {"single": "zero_wf"},
            "digital_marker": "ON",
            "integration_weights": {
                "cos": "long_cos_weights",
                "sin": "long_sin_weights",
                "minus_sin": "long_minus_sin_weights",
            },
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": const_amp},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "saturation_wf": {"type": "constant", "sample": saturation_amp},
        "short_readout_wf": {"type": "constant", "sample": short_readout_amp},
        "readout_wf": {"type": "constant", "sample": readout_amp},
        "long_readout_wf": {"type": "constant", "sample": long_readout_amp},
        "square_wf": {"type": "constant", "sample": square_flux_amp},
        "minus_square_wf": {"type": "constant", "sample": minus_square_flux_amp},
        "triangle_wf": {"type": "arbitrary", "samples": triangle_wf},
        "gaussian_wf": {"type": "arbitrary", "samples": gauss_wf},
        "pi_wf": {"type": "arbitrary", "samples": pi_wf},
        "-pi_wf": {"type": "arbitrary", "samples": minus_pi_wf},
        "pi_half_wf": {"type": "arbitrary", "samples": pi_half_wf},
        "-pi_half_wf": {"type": "arbitrary", "samples": minus_pi_half_wf},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},  # commonly used for measurement pulses, e.g., in a readout pulse
    },
    "integration_weights": {
        "short_cos_weights": {
            "cosine": [(1.0, short_readout_len)],  # Previous format for versions before 1.20: [1.0] * readout_len
            "sine": [(0.0, short_readout_len)],
        },
        "short_sin_weights": {
            "cosine": [(0.0, short_readout_len)],
            "sine": [(1.0, short_readout_len)],
        },
        "short_minus_sin_weights": {
            "cosine": [(0.0, short_readout_len)],
            "sine": [(-1.0, short_readout_len)],
        },
        "cos_weights": {
            "cosine": [(1.0, readout_len)],  # Previous format for versions before 1.20: [1.0] * readout_len
            "sine": [(0.0, readout_len)],
        },
        "sin_weights": {
            "cosine": [(0.0, readout_len)],
            "sine": [(1.0, readout_len)],
        },
        "minus_sin_weights": {
            "cosine": [(0.0, readout_len)],
            "sine": [(-1.0, readout_len)],
        },
        "long_cos_weights": {
            "cosine": [(1.0, long_readout_len)],  # Previous format for versions before 1.20: [1.0] * readout_len
            "sine": [(0.0, long_readout_len)],
        },
        "long_sin_weights": {
            "cosine": [(0.0, long_readout_len)],
            "sine": [(1.0, long_readout_len)],
        },
        "long_minus_sin_weights": {
            "cosine": [(0.0, long_readout_len)],
            "sine": [(-1.0, long_readout_len)],
        },
    },
    "mixers": {
        "mixer_qubit": [
            {
                "intermediate_frequency": ensemble_if,
                "lo_frequency": ensemble_LO,
                "correction": IQ_imbalance(-0.052, 0.058),
            },
        ],
    },
}
