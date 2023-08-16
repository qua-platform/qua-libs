import numpy as np
from scipy.signal.windows import gaussian
from qualang_tools.config.waveform_tools import drag_gaussian_pulse_waveforms


#######################
# AUXILIARY FUNCTIONS #
#######################


# IQ imbalance matrix
def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


#############
# VARIABLES #
#############

qop_ip = "127.0.0.1"
qop_port = 80

# Qubits
qubit_LO = 7.4e9
qubit_IF = qubit_LO - 7.27e9 - 3.6e6 + 0.325e6 - 0.234e6 - 0.0124e6
mixer_qubit_g = 0.02
mixer_qubit_phi = np.pi * 0.525

qubit_T1 = 10e3

saturation_len = 1000
saturation_amp = 0.1
const_len = 100
const_amp = 50e-3 * 4
pi_amp = 0.047766347173328

gauss_len = 200
gauss_sigma = gauss_len / 5
gauss_amp = 0.25
gauss_wf = gauss_amp * gaussian(gauss_len, gauss_sigma)

x180_len = 400
x180_sigma = x180_len / 5
x180_amp = 0.15
x180_wf, x180_der_wf = np.array(drag_gaussian_pulse_waveforms(x180_amp, x180_len, x180_sigma, alpha=0, delta=1))
# No DRAG when alpha=0, it's just a gaussian.

x90_len = x180_len
x90_sigma = x90_len / 5
x90_amp = x180_amp / 2
x90_wf, x90_der_wf = np.array(drag_gaussian_pulse_waveforms(x90_amp, x90_len, x90_sigma, alpha=0, delta=1))
# No DRAG when alpha=0, it's just a gaussian.

# Resonator
resonator_IF = 60e6
resonator_LO = 4.845200000000000e9
mixer_resonator_g = 0.0
mixer_resonator_phi = 0.0

time_of_flight = 300

short_readout_len = 2000
short_readout_amp = 0.25
readout_len = 500
readout_amp = 0.5
long_readout_len = 50000
long_readout_amp = 0.1

integration_start = 4
integration_stop = short_readout_len - integration_start

# Flux line
flux_line_IF = 0e6
const_flux_len = 200
const_flux_amp = 0.45

# IQ Plane Angle
rotation_angle = (28 / 180) * np.pi

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": -2e-3},  # I qubit
                2: {"offset": -28e-3},  # Q qubit
                3: {"offset": 0.0},  # I resonator (SSB mixer)
                4: {"offset": 0.234 + 0.012},  # flux line
            },
            "digital_outputs": {
                1: {},
            },
            "analog_inputs": {
                1: {"offset": 0.0, "gain_db": 0},  # I from down-conversion
                2: {"offset": 0.0, "gain_db": 0},  # Q from down-conversion
            },
        },
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
                "cw": "const_pulse",
                "saturation": "saturation_pulse",
                "gauss": "gaussian_pulse",
                "pi": "pi_pulse",
                "pi_half": "pi_half_pulse",
                "X90": "X_90_pulse",
                "Y90": "Y_90_pulse",
            },
        },
        "resonator": {
            "singleInput": {
                "port": ("con1", 3),
            },
            "intermediate_frequency": resonator_IF,
            "operations": {
                "cw": "const_single_pulse",
                "short_readout": "short_readout_pulse",
                "readout": "readout_pulse",
                "long_readout": "long_readout_pulse",
            },
            "outputs": {
                "out1": ("con1", 1),
                "out2": ("con1", 2),
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
        },
        "flux_line": {
            "singleInput": {
                "port": ("con1", 4),
            },
            "intermediate_frequency": flux_line_IF,
            "operations": {
                "cw": "const_flux_pulse",
            },
        },
        "flux_line_sticky": {
            "singleInput": {
                "port": ("con1", 4),
            },
            "hold_offset": {"duration": 100},  # in clock cycles
            "intermediate_frequency": flux_line_IF,
            "operations": {
                "cw": "const_flux_pulse",
            },
        },
    },
    "pulses": {
        "const_single_pulse": {
            "operation": "control",
            "length": const_len,
            "waveforms": {
                "single": "const_wf",
            },
        },
        "const_flux_pulse": {
            "operation": "control",
            "length": const_flux_len,
            "waveforms": {
                "single": "const_flux_wf",
            },
        },
        "const_pulse": {
            "operation": "control",
            "length": const_len,
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
        },
        "pi_pulse": {
            "operation": "control",
            "length": const_len,
            "waveforms": {
                "I": "pi_wf",
                "Q": "zero_wf",
            },
        },
        "pi_half_pulse": {
            "operation": "control",
            "length": const_len,
            "waveforms": {
                "I": "pi_half_wf",
                "Q": "zero_wf",
            },
        },
        "X_90_pulse": {
            "operation": "control",
            "length": const_len,
            "waveforms": {
                "I": "pi_half_wf",
                "Q": "zero_wf",
            },
        },
        "Y_90_pulse": {
            "operation": "control",
            "length": const_len,
            "waveforms": {
                "I": "zero_wf",
                "Q": "pi_half_wf",
            },
        },
        "saturation_pulse": {
            "operation": "control",
            "length": saturation_len,
            "waveforms": {"I": "saturation_drive_wf", "Q": "zero_wf"},
        },
        "gaussian_pulse": {
            "operation": "control",
            "length": gauss_len,
            "waveforms": {
                "I": "gauss_wf",
                "Q": "zero_wf",
            },
        },
        "short_readout_pulse": {
            "operation": "measurement",
            "length": short_readout_len,
            "waveforms": {
                "single": "short_readout_wf",
            },
            "integration_weights": {
                "cos": "short_cosine_weights",
                "sin": "short_sine_weights",
                "minus_sin": "short_minus_sine_weights",
                "rotated_cos": "short_rotated_cosine_weights",
                "rotated_sin": "short_rotated_sine_weights",
                "rotated_minus_sin": "short_rotated_minus_sine_weights",
            },
            "digital_marker": "ON",
        },
        "readout_pulse": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {
                "single": "readout_wf",
            },
            "integration_weights": {
                "cos": "cosine_weights",
                "sin": "sine_weights",
                "minus_sin": "minus_sine_weights",
                "rotated_cos": "rotated_cosine_weights",
                "rotated_sin": "rotated_sine_weights",
                "rotated_minus_sin": "rotated_minus_sine_weights",
            },
            "digital_marker": "ON",
        },
        "long_readout_pulse": {
            "operation": "measurement",
            "length": long_readout_len,
            "waveforms": {
                "single": "long_readout_wf",
            },
            "integration_weights": {
                "cos": "long_cosine_weights",
                "sin": "long_sine_weights",
                "minus_sin": "long_minus_sine_weights",
                "rotated_cos": "long_rotated_cosine_weights",
                "rotated_sin": "long_rotated_sine_weights",
                "rotated_minus_sin": "long_rotated_minus_sine_weights",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": const_amp},
        "pi_wf": {"type": "constant", "sample": pi_amp},
        "pi_half_wf": {"type": "constant", "sample": pi_amp / 2},
        "const_flux_wf": {"type": "constant", "sample": const_flux_amp},
        "saturation_drive_wf": {"type": "constant", "sample": saturation_amp},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "gauss_wf": {"type": "arbitrary", "samples": gauss_wf.tolist()},
        "short_readout_wf": {"type": "constant", "sample": short_readout_amp},
        "readout_wf": {"type": "constant", "sample": readout_amp},
        "long_readout_wf": {"type": "constant", "sample": long_readout_amp},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "short_cosine_weights": {
            "cosine": [(0.0, integration_start), (1.0, short_readout_len - integration_start)],
            "sine": [(0.0, short_readout_len)],
        },
        "short_sine_weights": {
            "cosine": [(0.0, short_readout_len)],
            "sine": [(0.0, integration_start), (1.0, short_readout_len - integration_start)],
        },
        "short_minus_sine_weights": {
            "cosine": [(0.0, short_readout_len)],
            "sine": [(0.0, integration_start), (-1.0, short_readout_len - integration_start)],
        },
        "short_rotated_cosine_weights": {
            "cosine": [(0.0, integration_start), (np.cos(rotation_angle), short_readout_len - integration_start)],
            "sine": [(0.0, integration_start), (-np.sin(rotation_angle), short_readout_len - integration_start)],
        },
        "short_rotated_sine_weights": {
            "cosine": [(0.0, integration_start), (np.sin(rotation_angle), short_readout_len - integration_start)],
            "sine": [(0.0, integration_start), (np.cos(rotation_angle), short_readout_len - integration_start)],
        },
        "short_rotated_minus_sine_weights": {
            "cosine": [(0.0, integration_start), (-np.sin(rotation_angle), short_readout_len - integration_start)],
            "sine": [(0.0, integration_start), (-np.cos(rotation_angle), short_readout_len - integration_start)],
        },
        "cosine_weights": {
            "cosine": [(1.0, readout_len)],
            "sine": [(0.0, readout_len)],
        },
        "sine_weights": {
            "cosine": [(0.0, readout_len)],
            "sine": [(1.0, readout_len)],
        },
        "minus_sine_weights": {
            "cosine": [(0.0, readout_len)],
            "sine": [(-1.0, readout_len)],
        },
        "rotated_cosine_weights": {
            "cosine": [(np.cos(rotation_angle), readout_len)],
            "sine": [(-np.sin(rotation_angle), readout_len)],
        },
        "rotated_sine_weights": {
            "cosine": [(np.sin(rotation_angle), readout_len)],
            "sine": [(np.cos(rotation_angle), readout_len)],
        },
        "rotated_minus_sine_weights": {
            "cosine": [(-np.sin(rotation_angle), readout_len)],
            "sine": [(-np.cos(rotation_angle), readout_len)],
        },
        "long_cosine_weights": {
            "cosine": [(1.0, long_readout_len)],
            "sine": [(0.0, long_readout_len)],
        },
        "long_sine_weights": {
            "cosine": [(0.0, long_readout_len)],
            "sine": [(1.0, long_readout_len)],
        },
        "long_minus_sine_weights": {
            "cosine": [(0.0, long_readout_len)],
            "sine": [(-1.0, long_readout_len)],
        },
        "long_rotated_cosine_weights": {
            "cosine": [(np.cos(rotation_angle), long_readout_len)],
            "sine": [(-np.sin(rotation_angle), long_readout_len)],
        },
        "long_rotated_sine_weights": {
            "cosine": [(np.sin(rotation_angle), long_readout_len)],
            "sine": [(np.cos(rotation_angle), long_readout_len)],
        },
        "long_rotated_minus_sine_weights": {
            "cosine": [(-np.sin(rotation_angle), long_readout_len)],
            "sine": [(-np.cos(rotation_angle), long_readout_len)],
        },
    },
    "mixers": {
        "mixer_qubit": [
            {
                "intermediate_frequency": qubit_IF,
                "lo_frequency": qubit_LO,
                "correction": IQ_imbalance(mixer_qubit_g, mixer_qubit_phi),
            }
        ],
        "mixer_resonator": [
            {
                "intermediate_frequency": resonator_IF,
                "lo_frequency": resonator_LO,
                "correction": IQ_imbalance(mixer_resonator_g, mixer_resonator_phi),
            }
        ],
    },
}
