from scipy.signal.windows import gaussian
import numpy as np


# Used to correct for IQ mixer imbalances
def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


#############
# VARIABLES #
#############

qop_ip = "127.0.0.1"
cluster_name = "my_cluster"
qop_port = 80

# Set octave_config to None if no octave are present
octave_config = None

# Frequencies
blue_sideband = 140e6  # 287e6#253e6
red_sideband = 280e6  # 287e6#253e6

# For res # Q1 = 33.75e6 # Q2 = 89e6  # Q3 = 142.3e6 # Q4 = 183.75e6
resonator1_if = 184.55e6  # 34.85e6 #142.15e6 #36.65e6#36.8e6
qubit1_if = 111.69e6  # 58.687e6#115.4142e6 #63.870931e6 #83.247e6 #-52.285e6 #280.93e6
qubit1_if_ef = -168.429e6

resonator2_if = 89.5e6
qubit2_if = -43.764e6
qubit2_if_ef = -168.429e6

# LOs are used in plots. They can also be used marking mixer elements in the config.
# On top of that they can also be used for setting LO sources (this make sure that everything is in sync)
qubit_LO = 4.9e9
resonator_LO = 6.25e9

# Drive amps
q1_ge_amp = 0.418954  # 0.219941 #0.239310#0.190015
q1_ef_amp = 0.561779 * 0.5

q2_ge_amp = 0.384762
q2_ef_amp = 0.561779 * 0.5

# Readout parameters
const_amp = 0.3
const_len = 500

readout_len = 2000
readout_amp = 0.3

# Time of flight must be a multiple of 4 and greater or equal than 24 (got 12)
time_of_flight = 272  # Time it takes the pulses to go through the RF chain, including the device.

# Qubit parameters:
# saturation_amp = 0.2
# saturation_len = 50000  # Needs to be several T1 so that the final state is an equal population of |0> and |1>

## for e to f
saturation_amp = 0.2
saturation_len = 10000

# #ge
# Pi pulse parameters
pi_len = 80  # 80  # in units of ns
pi_amp = 0.4  # in units of volts
pi_wf = (pi_amp * (gaussian(pi_len, pi_len / 5) - gaussian(pi_len, pi_len / 5)[-1])).tolist()

# Pi_half pulse parameters
pi_half_len = 80  # in units of ns
pi_half_amp = 0.4 * 0.5  # in units of volts
pi_half_wf = (
    pi_half_amp * (gaussian(pi_half_len, pi_half_len / 5) - gaussian(pi_half_len, pi_half_len / 5)[-1])
).tolist()

# #ef
# Pi pulse parameters
pi_len_ef = 80  # 80  # in units of ns
pi_amp_ef = 0.4  # in units of volts
pi_wf_ef = (pi_amp_ef * (gaussian(pi_len_ef, pi_len_ef / 5) - gaussian(pi_len_ef, pi_len_ef / 5)[-1])).tolist()
# Pi_half pulse parameters
pi_half_len_ef = 80  # 80  # in units of ns
pi_half_amp_ef = 0.4 * 0.5  # in units of volts
pi_half_wf_ef = (
    pi_half_amp_ef * (gaussian(pi_half_len_ef, pi_half_len_ef / 5) - gaussian(pi_half_len_ef, pi_half_len_ef / 5)[-1])
).tolist()

# Gaussian pulse parameters
gauss_amp = 0.4  # The gaussian is used when calibrating pi and pi_half pulses
gauss_len = 80  # The gaussian is used when calibrating pi and pi_half pulses
gauss_wf = (gauss_amp * (gaussian(gauss_len, gauss_len / 5) - gaussian(gauss_len, gauss_len / 5)[-1])).tolist()

# Flux:
bs_a = 0.8
rs_a = 0.1
square_flux_amp = 0.4  # 0.12
minus_square_flux_amp = -square_flux_amp

# Rotation angle:
rotation_angle = (-73 / 180) * np.pi

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                2: {"offset": +0.0},  # Fast Flux red sideband
                3: {"offset": +0.0031},  # Fast Flux I blue sideband
                4: {"offset": -0.0074},  # Fast Flux Q blue sideband
                5: {"offset": +0.0088},  # qubit I
                6: {"offset": +0.0027},  # qubit Q
                9: {"offset": -0.0366},  # Resonator I
                10: {"offset": +0.0044},  # Resonator Q
            },
            "digital_outputs": {},
            "analog_inputs": {
                1: {"offset": -0.13987294807128908 - 0.11194140625, "gain_db": 10},  # I from down conversion
                2: {"offset": -0.21447197199707035 - 0.09380615234375, "gain_db": 10},  # Q from down conversion
            },
        },
    },
    "elements": {
        "qubit1": {
            "mixInputs": {
                "I": ("con1", 5),
                "Q": ("con1", 6),
                "lo_frequency": qubit_LO,
                "mixer": "mixer_qubit1",
            },
            "intermediate_frequency": qubit1_if,
            "operations": {
                "const": "const_pulse",
                "saturation": "saturation_pulse",
                "gaussian": "gaussian_pulse",
                "pi": "pi_pulse",
                "pi_half": "pi_half_pulse",
            },
        },
        "qubit1_ef": {
            "mixInputs": {
                "I": ("con1", 5),
                "Q": ("con1", 6),
                "lo_frequency": qubit_LO,
                "mixer": "mixer_qubit1ef",
            },
            "intermediate_frequency": qubit1_if_ef,
            "operations": {
                "const": "const_pulse",
                "saturation": "saturation_pulse",
                "gaussian": "gaussian_pulse",
                "pi": "pi_pulse_ef",
                "pi_half": "pi_half_pulse_ef",
            },
        },
        "qubit2": {
            "mixInputs": {
                "I": ("con1", 5),
                "Q": ("con1", 6),
                "lo_frequency": qubit_LO,
                "mixer": "mixer_qubit2",
            },
            "intermediate_frequency": qubit2_if,
            "operations": {
                "const": "const_pulse",
                "saturation": "saturation_pulse",
                "gaussian": "gaussian_pulse",
                "pi": "pi_pulse",
                "pi_half": "pi_half_pulse",
            },
        },
        "qubit2_ef": {
            "mixInputs": {
                "I": ("con1", 5),
                "Q": ("con1", 6),
                "lo_frequency": qubit_LO,
                "mixer": "mixer_qubit2ef",
            },
            "intermediate_frequency": qubit2_if_ef,
            "operations": {
                "const": "const_pulse",
                "saturation": "saturation_pulse",
                "gaussian": "gaussian_pulse",
                "pi": "pi_pulse_ef",
                "pi_half": "pi_half_pulse_ef",
            },
        },
        "resonator1": {
            "mixInputs": {
                "I": ("con1", 9),
                "Q": ("con1", 10),
                "lo_frequency": resonator_LO,
                "mixer": "mixer_resonator1",
            },
            "intermediate_frequency": resonator1_if,
            "operations": {
                "const": "const_pulse",
                "readout": "readout_pulse",
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
            "outputs": {
                "out1": ("con1", 1),
                "out2": ("con1", 2),
            },
        },
        "resonator2": {
            "mixInputs": {
                "I": ("con1", 9),
                "Q": ("con1", 10),
                "lo_frequency": resonator_LO,
                "mixer": "mixer_resonator2",
            },
            "intermediate_frequency": resonator2_if,
            "operations": {
                "const": "const_pulse",
                "readout": "readout_pulse",
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
            "outputs": {
                "out1": ("con1", 1),
                "out2": ("con1", 2),
            },
        },
        "flux_line_bs": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": qubit_LO,
                "mixer": "mixer_flux1",
            },
            "intermediate_frequency": blue_sideband,
            "operations": {
                "const": "const_pulse",
            },
        },
        "flux_line_rs": {
            "singleInput": {"port": ("con1", 2)},
            "intermediate_frequency": red_sideband,
            "operations": {
                "offset": "square_pulse",
                "-offset": "-square_pulse",
            },
        },
    },
    "pulses": {
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
        "pi_pulse_ef": {
            "operation": "control",
            "length": pi_len_ef,  # in ns
            "waveforms": {
                "I": "pi_wf_ef",
                "Q": "zero_wf",
            },
        },
        "pi_half_pulse_ef": {
            "operation": "control",
            "length": pi_half_len_ef,  # in ns
            "waveforms": {
                "I": "pi_half_wf_ef",
                "Q": "zero_wf",
            },
        },
        "square_pulse": {"operation": "control", "length": 16, "waveforms": {"single": "square_wf"}},  # in ns
        "-square_pulse": {"operation": "control", "length": 16, "waveforms": {"single": "-square_wf"}},  # in ns
        "readout_pulse": {
            "operation": "measurement",
            "length": readout_len,  # in ns
            "waveforms": {"I": "readout_wf", "Q": "zero_wf"},
            "digital_marker": "ON",
            "integration_weights": {
                "cos": "cos_weights",
                "sin": "sin_weights",
                "minus_sin": "minus_sin_weights",
                "rotated_cos": "rotated_cos_weights",
                "rotated_sin": "rotated_sin_weights",
                "rotated_minus_sin": "rotated_minus_sin_weights",
            },
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": const_amp},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "saturation_wf": {"type": "constant", "sample": saturation_amp},
        "readout_wf": {"type": "constant", "sample": readout_amp},
        "square_wf": {"type": "constant", "sample": square_flux_amp},
        "-square_wf": {"type": "constant", "sample": minus_square_flux_amp},
        "gaussian_wf": {"type": "arbitrary", "samples": gauss_wf},
        "pi_wf": {"type": "arbitrary", "samples": pi_wf},
        "pi_half_wf": {"type": "arbitrary", "samples": pi_half_wf},
        "pi_wf_ef": {"type": "arbitrary", "samples": pi_wf_ef},
        "pi_half_wf_ef": {"type": "arbitrary", "samples": pi_half_wf_ef},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},  # commonly used for measurement pulses, e.g., in a readout pulse
    },
    "integration_weights": {
        "cos_weights": {
            "cosine": [(1.0, readout_len)],
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
        "rotated_cos_weights": {
            "cosine": [(np.cos(rotation_angle), readout_len)],
            "sine": [(-np.sin(rotation_angle), readout_len)],
        },
        "rotated_sin_weights": {
            "cosine": [(np.sin(rotation_angle), readout_len)],
            "sine": [(np.cos(rotation_angle), readout_len)],
        },
        "rotated_minus_sin_weights": {
            "cosine": [(-np.sin(rotation_angle), readout_len)],
            "sine": [(-np.cos(rotation_angle), readout_len)],
        },
    },
    "mixers": {
        "mixer_flux1": [
            {
                "intermediate_frequency": blue_sideband,
                "lo_frequency": qubit_LO,
                "correction": IQ_imbalance(0.004, 0.088),
            },
        ],
        "mixer_qubit1": [
            {"intermediate_frequency": qubit1_if, "lo_frequency": qubit_LO, "correction": IQ_imbalance(0.01, 0.082)},
        ],
        "mixer_qubit1ef": [
            {
                "intermediate_frequency": qubit1_if_ef,
                "lo_frequency": qubit_LO,
                "correction": IQ_imbalance(0.006, 0.061),
            },
        ],
        "mixer_qubit2": [
            {"intermediate_frequency": qubit2_if, "lo_frequency": qubit_LO, "correction": IQ_imbalance(0.005, 0.05)},
        ],
        "mixer_qubit2ef": [
            {"intermediate_frequency": qubit2_if_ef, "lo_frequency": qubit_LO, "correction": IQ_imbalance(0.005, 0.05)},
        ],
        "mixer_resonator1": [
            {
                "intermediate_frequency": resonator1_if,
                "lo_frequency": resonator_LO,
                "correction": IQ_imbalance(0.006, 0.017),
            },
        ],
        "mixer_resonator2": [
            {
                "intermediate_frequency": resonator2_if,
                "lo_frequency": resonator_LO,
                "correction": IQ_imbalance(0.006, 0.017),
            },
        ],
    },
}
