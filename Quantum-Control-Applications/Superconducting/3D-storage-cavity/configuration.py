"""
Octave configuration working for QOP222 and qm-qua==1.1.5 and newer.
"""

from pathlib import Path

import numpy as np
import plotly.io as pio
from qualang_tools.config.waveform_tools import drag_gaussian_pulse_waveforms
from qualang_tools.units import unit
from set_octave import OctaveUnit, octave_declaration

pio.renderers.default = "browser"
#######################
# AUXILIARY FUNCTIONS #
#######################
u = unit(coerce_to_integer=True)

######################
# Network parameters #
######################
qop_ip = "172.16.33.101"  # Write the QM router IP address
cluster_name = "Cluster_81"  # Write your cluster_name if version >= QOP220
qop_port = None  # Write the QOP port if version < QOP220

# Path to save data
save_dir = Path().absolute() / "QM" / "INSTALLATION" / "data"

############################
# Set octave configuration #
############################

# The Octave port is 11xxx, where xxx are the last three digits of the Octave internal IP that can be accessed from
# the OPX admin panel if you QOP version is >= QOP220. Otherwise, it is 50 for Octave1, then 51, 52 and so on.
octave_1 = OctaveUnit("octave1", "192.168.88.250", port=80, con="con1")
# octave_2 = OctaveUnit("octave2", qop_ip, port=11051, con="con1")

# If the control PC or local network is connected to the internal network of the QM router (port 2 onwards)
# or directly to the Octave (without QM the router), use the local octave IP and port 80.
# octave_ip = "192.168.88.X"
# octave_1 = OctaveUnit("octave1", octave_ip, port=80, con="con1")

# Add the octaves
octaves = [octave_1]
# Configure the Octaves
octave_config = octave_declaration(octaves)

#####################
# OPX configuration #
#####################

#############################################
#                  Storage                  #
#############################################
storage_LO = 5 * u.GHz
storage_IF = 100 * u.MHz  # correspond to Fock state n=0


storage_T1 = int(1 * u.ms)
storage_thermalization_time = 5 * storage_T1
t_parity = 300 * u.ns

off_pump_len = 16 * u.ns
storage_off_pump_amp = 0.1
# Continuous wave
storage_const_len = 200
storage_const_amp = 0.02

# Fock state n=1 parameters
# beta1_wave
storage_beta1_len = 104
storage_beta1_amp = storage_const_amp

# beta2_wave
storage_beta2_len = 52
storage_beta2_amp = -storage_const_amp

# Parameters for storage cavity T2
# beta3_wave
storage_beta3_len = 72
storage_beta3_amp = 0.015

#############################################
#                  Qubits                   #
#############################################
qubit_LO = 3.9 * u.GHz
qubit_IF = 100 * u.MHz
qubit_IF_n1 = 98 * u.MHz  # correspond to Fock state n=1

qubit_T1 = int(40 * u.us)
thermalization_time = 10 * qubit_T1

# Continuous wave
const_len = 100
const_amp = 0.1
# Saturation_pulse
saturation_len = 10 * u.us
saturation_amp = 0.1
# Square pi pulse
square_pi_len = 100
square_pi_amp = 0.1
# Drag pulses
drag_coef = 0
anharmonicity = -200 * u.MHz
AC_stark_detuning = 0 * u.MHz


x180_len = 72
x180_sigma = x180_len / 3
x180_amp = 0.1
x180_wf, x180_der_wf = np.array(
    drag_gaussian_pulse_waveforms(x180_amp, x180_len, x180_sigma, drag_coef, anharmonicity, AC_stark_detuning)
)
x180_I_wf = x180_wf
x180_Q_wf = x180_der_wf
# No DRAG when alpha=0, it's just a gaussian.

x180_len_long = 5000
x180_sigma_long = x180_len_long / 3
x180_amp_long = 0.001
x180_wf_long, x180_der_wf_long = np.array(
    drag_gaussian_pulse_waveforms(
        x180_amp_long, x180_len_long, x180_sigma_long, drag_coef, anharmonicity, AC_stark_detuning
    )
)
x180_I_wf_long = x180_wf_long
x180_Q_wf_long = x180_der_wf_long
# No DRAG when alpha=0, it's just a gaussian.


x360_len_long = x180_len_long
x360_sigma_long = x360_len_long / 3

x360_amp_long = x180_amp_long * 2
x360_wf_long, x360_der_wf_long = np.array(
    drag_gaussian_pulse_waveforms(
        x360_amp_long, x360_len_long, x360_sigma_long, drag_coef, anharmonicity, AC_stark_detuning
    )
)
x360_I_wf_long = x360_wf_long
x360_Q_wf_long = x360_der_wf_long

x90_len = x180_len
x90_sigma = x90_len / 3
x90_amp = x180_amp / 2
x90_wf, x90_der_wf = np.array(
    drag_gaussian_pulse_waveforms(x90_amp, x90_len, x90_sigma, drag_coef, anharmonicity, AC_stark_detuning)
)
x90_I_wf = x90_wf
x90_Q_wf = x90_der_wf
# No DRAG when alpha=0, it's just a gaussian.

minus_x90_len = x180_len
minus_x90_sigma = minus_x90_len / 3
minus_x90_amp = -x90_amp
minus_x90_wf, minus_x90_der_wf = np.array(
    drag_gaussian_pulse_waveforms(
        minus_x90_amp,
        minus_x90_len,
        minus_x90_sigma,
        drag_coef,
        anharmonicity,
        AC_stark_detuning,
    )
)
minus_x90_I_wf = minus_x90_wf
minus_x90_Q_wf = minus_x90_der_wf
# No DRAG when alpha=0, it's just a gaussian.

y180_len = x180_len
y180_sigma = y180_len / 5
y180_amp = x180_amp
y180_wf, y180_der_wf = np.array(
    drag_gaussian_pulse_waveforms(y180_amp, y180_len, y180_sigma, drag_coef, anharmonicity, AC_stark_detuning)
)
y180_I_wf = (-1) * y180_der_wf
y180_Q_wf = y180_wf
# No DRAG when alpha=0, it's just a gaussian.

y90_len = x180_len
y90_sigma = y90_len / 5
y90_amp = y180_amp / 2
y90_wf, y90_der_wf = np.array(
    drag_gaussian_pulse_waveforms(y90_amp, y90_len, y90_sigma, drag_coef, anharmonicity, AC_stark_detuning)
)
y90_I_wf = (-1) * y90_der_wf
y90_Q_wf = y90_wf
# No DRAG when alpha=0, it's just a gaussian.

minus_y90_len = y180_len
minus_y90_sigma = minus_y90_len / 5
minus_y90_amp = -y90_amp
minus_y90_wf, minus_y90_der_wf = np.array(
    drag_gaussian_pulse_waveforms(
        minus_y90_amp,
        minus_y90_len,
        minus_y90_sigma,
        drag_coef,
        anharmonicity,
        AC_stark_detuning,
    )
)
minus_y90_I_wf = (-1) * minus_y90_der_wf
minus_y90_Q_wf = minus_y90_wf
# No DRAG when alpha=0, it's just a gaussian.

#############################################
#                Resonators                 #
#############################################
resonator_LO = 6.9 * u.GHz
resonator_IF = 150 * u.MHz

resonator_off_pump_amp = 0.45

readout_len = 1000
readout_amp = 0.05

time_of_flight = 28
depletion_time = 2 * u.us

opt_weights = False
if opt_weights:
    weights = np.load("optimal_weights.npz")
    opt_weights_real = [(x, weights["division_length"] * 4) for x in weights["weights_real"]]
    opt_weights_minus_imag = [(x, weights["division_length"] * 4) for x in weights["weights_minus_imag"]]
    opt_weights_imag = [(x, weights["division_length"] * 4) for x in weights["weights_imag"]]
    opt_weights_minus_real = [(x, weights["division_length"] * 4) for x in weights["weights_minus_real"]]
else:
    opt_weights_real = [(1.0, readout_len)]
    opt_weights_minus_imag = [(0.0, readout_len)]
    opt_weights_imag = [(0.0, readout_len)]
    opt_weights_minus_real = [(-1.0, readout_len)]
# IQ Plane
rotation_angle = (0 / 180) * np.pi
ge_threshold = 1.0

#############################################
#                  Config                   #
#############################################
config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": 0.0},  # I resonator
                2: {"offset": 0.0},  # Q resonator
                3: {"offset": 0.0},  # I qubit
                4: {"offset": 0.0},  # Q qubit
                5: {"offset": 0.0},  # I qubit pump
                6: {"offset": 0.0},  # Q qubit pump
                7: {"offset": 0.0},  # I storage
                8: {"offset": 0.0},  # Q storage
            },
            "digital_outputs": {},
            "analog_inputs": {
                1: {"offset": 0.00835, "gain_db": 0},  # I from down-conversion
                2: {"offset": 0.017176, "gain_db": 0},  # Q from down-conversion
            },
        },
    },
    "elements": {
        "qubit": {
            "RF_inputs": {"port": ("octave1", 3)},
            "intermediate_frequency": qubit_IF,
            "operations": {
                "cw": "const_pulse",
                "saturation": "saturation_pulse",
                "pi": "square_pi_pulse",
                "pi_half": "square_pi_half_pulse",
                "x90": "x90_pulse",
                "x180": "x180_pulse",
                "x180_long": "x180_pulse_long",
                "-x90": "-x90_pulse",
                "y90": "y90_pulse",
                "y180": "y180_pulse",
                "-y90": "-y90_pulse",
                "x360_long": "x360_pulse_long",
            },
        },
        "storage": {
            "RF_inputs": {"port": ("octave1", 4)},
            "intermediate_frequency": storage_IF,
            "operations": {
                "cw": "storage_const_pulse",
                "beta1": "storage_beta1_pulse",
                "beta2": "storage_beta2_pulse",
                "beta3": "storage_beta3_pulse",
                "off_pump": "storage_off_pump_pulse",
            },
        },
        "resonator": {
            "RF_inputs": {"port": ("octave1", 1)},
            "RF_outputs": {"port": ("octave1", 1)},
            "intermediate_frequency": resonator_IF,
            "operations": {"cw": "const_pulse", "readout": "readout_pulse", "off_pump": "resonator_off_pump_pulse"},
            "time_of_flight": time_of_flight,
            "smearing": 0,
        },
    },
    "octaves": {
        "octave1": {
            "RF_outputs": {
                1: {
                    "LO_frequency": resonator_LO,
                    "LO_source": "internal",
                    "output_mode": "always_on",
                    "gain": 0,
                },
                3: {
                    "LO_frequency": qubit_LO,
                    "LO_source": "internal",
                    "output_mode": "always_on",
                    "gain": 0,
                },
                4: {
                    "LO_frequency": storage_LO,
                    "LO_source": "internal",
                    "output_mode": "always_on",
                    "gain": 0,
                },
            },
            "RF_inputs": {
                1: {
                    "LO_frequency": resonator_LO,
                    "LO_source": "internal",
                },
            },
            "connectivity": "con1",
        }
    },
    "pulses": {
        "const_pulse": {
            "operation": "control",
            "length": const_len,
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
        },
        "square_pi_pulse": {
            "operation": "control",
            "length": square_pi_len,
            "waveforms": {
                "I": "square_pi_wf",
                "Q": "zero_wf",
            },
        },
        "square_pi_half_pulse": {
            "operation": "control",
            "length": square_pi_len,
            "waveforms": {
                "I": "square_pi_half_wf",
                "Q": "zero_wf",
            },
        },
        "saturation_pulse": {
            "operation": "control",
            "length": saturation_len,
            "waveforms": {"I": "saturation_drive_wf", "Q": "zero_wf"},
        },
        "x90_pulse": {
            "operation": "control",
            "length": x90_len,
            "waveforms": {
                "I": "x90_I_wf",
                "Q": "x90_Q_wf",
            },
        },
        "x180_pulse": {
            "operation": "control",
            "length": x180_len,
            "waveforms": {
                "I": "x180_I_wf",
                "Q": "x180_Q_wf",
            },
        },
        "x180_pulse_long": {
            "operation": "control",
            "length": x180_len_long,
            "waveforms": {
                "I": "x180_I_wf_long",
                "Q": "x180_Q_wf_long",
            },
        },
        "x360_pulse_long": {
            "operation": "control",
            "length": x360_len_long,
            "waveforms": {
                "I": "x360_I_wf_long",
                "Q": "x360_Q_wf_long",
            },
        },
        "-x90_pulse": {
            "operation": "control",
            "length": minus_x90_len,
            "waveforms": {
                "I": "minus_x90_I_wf",
                "Q": "minus_x90_Q_wf",
            },
        },
        "y90_pulse": {
            "operation": "control",
            "length": y90_len,
            "waveforms": {
                "I": "y90_I_wf",
                "Q": "y90_Q_wf",
            },
        },
        "y180_pulse": {
            "operation": "control",
            "length": y180_len,
            "waveforms": {
                "I": "y180_I_wf",
                "Q": "y180_Q_wf",
            },
        },
        "-y90_pulse": {
            "operation": "control",
            "length": minus_y90_len,
            "waveforms": {
                "I": "minus_y90_I_wf",
                "Q": "minus_y90_Q_wf",
            },
        },
        "storage_const_pulse": {
            "operation": "control",
            "length": storage_const_len,
            "waveforms": {
                "I": "storage_const_wf",
                "Q": "zero_wf",
            },
        },
        "storage_beta1_pulse": {
            "operation": "control",
            "length": storage_beta1_len,
            "waveforms": {
                "I": "storage_beta1_wf",
                "Q": "zero_wf",
            },
        },
        "storage_beta2_pulse": {
            "operation": "control",
            "length": storage_beta2_len,
            "waveforms": {
                "I": "storage_beta2_wf",
                "Q": "zero_wf",
            },
        },
        "storage_beta3_pulse": {
            "operation": "control",
            "length": storage_beta3_len,
            "waveforms": {
                "I": "storage_beta3_wf",
                "Q": "zero_wf",
            },
        },
        "storage_off_pump_pulse": {
            "operation": "control",
            "length": off_pump_len,
            "waveforms": {
                "I": "storage_off_pump_pulse_wf",
                "Q": "zero_wf",
            },
        },
        "resonator_off_pump_pulse": {
            "operation": "control",
            "length": off_pump_len,
            "waveforms": {
                "I": "resonator_off_pump_pulse_wf",
                "Q": "zero_wf",
            },
        },
        "readout_pulse": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {
                "I": "readout_wf",
                "Q": "zero_wf",
            },
            "integration_weights": {
                "cos": "cosine_weights",
                "sin": "sine_weights",
                "minus_sin": "minus_sine_weights",
                "rotated_cos": "rotated_cosine_weights",
                "rotated_sin": "rotated_sine_weights",
                "rotated_minus_sin": "rotated_minus_sine_weights",
                "opt_cos": "opt_cosine_weights",
                "opt_sin": "opt_sine_weights",
                "opt_minus_sin": "opt_minus_sine_weights",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": const_amp},
        "storage_const_wf": {"type": "constant", "sample": storage_const_amp},
        "storage_beta1_wf": {"type": "constant", "sample": storage_beta1_amp},
        "storage_beta2_wf": {"type": "constant", "sample": storage_beta2_amp},
        "storage_beta3_wf": {"type": "constant", "sample": storage_beta3_amp},
        "storage_off_pump_pulse_wf": {"type": "constant", "sample": storage_off_pump_amp},
        "resonator_off_pump_pulse_wf": {"type": "constant", "sample": resonator_off_pump_amp},
        "saturation_drive_wf": {"type": "constant", "sample": saturation_amp},
        "square_pi_wf": {"type": "constant", "sample": square_pi_amp},
        "square_pi_half_wf": {"type": "constant", "sample": square_pi_amp / 2},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "x90_I_wf": {"type": "arbitrary", "samples": x90_I_wf.tolist()},
        "x90_Q_wf": {"type": "arbitrary", "samples": x90_Q_wf.tolist()},
        "x180_I_wf": {"type": "arbitrary", "samples": x180_I_wf.tolist()},
        "x180_Q_wf": {"type": "arbitrary", "samples": x180_Q_wf.tolist()},
        "x180_I_wf_long": {"type": "arbitrary", "samples": x180_I_wf_long.tolist()},
        "x180_Q_wf_long": {"type": "arbitrary", "samples": x180_Q_wf_long.tolist()},
        "x360_I_wf_long": {"type": "arbitrary", "samples": x360_I_wf_long.tolist()},
        "x360_Q_wf_long": {"type": "arbitrary", "samples": x360_Q_wf_long.tolist()},
        "minus_x90_I_wf": {"type": "arbitrary", "samples": minus_x90_I_wf.tolist()},
        "minus_x90_Q_wf": {"type": "arbitrary", "samples": minus_x90_Q_wf.tolist()},
        "y90_Q_wf": {"type": "arbitrary", "samples": y90_Q_wf.tolist()},
        "y90_I_wf": {"type": "arbitrary", "samples": y90_I_wf.tolist()},
        "y180_Q_wf": {"type": "arbitrary", "samples": y180_Q_wf.tolist()},
        "y180_I_wf": {"type": "arbitrary", "samples": y180_I_wf.tolist()},
        "minus_y90_Q_wf": {"type": "arbitrary", "samples": minus_y90_Q_wf.tolist()},
        "minus_y90_I_wf": {"type": "arbitrary", "samples": minus_y90_I_wf.tolist()},
        "readout_wf": {"type": "constant", "sample": readout_amp},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
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
        "opt_cosine_weights": {
            "cosine": opt_weights_real,
            "sine": opt_weights_minus_imag,
        },
        "opt_sine_weights": {
            "cosine": opt_weights_imag,
            "sine": opt_weights_real,
        },
        "opt_minus_sine_weights": {
            "cosine": opt_weights_minus_imag,
            "sine": opt_weights_minus_real,
        },
        "rotated_cosine_weights": {
            "cosine": [(np.cos(rotation_angle), readout_len)],
            "sine": [(np.sin(rotation_angle), readout_len)],
        },
        "rotated_sine_weights": {
            "cosine": [(-np.sin(rotation_angle), readout_len)],
            "sine": [(np.cos(rotation_angle), readout_len)],
        },
        "rotated_minus_sine_weights": {
            "cosine": [(np.sin(rotation_angle), readout_len)],
            "sine": [(-np.cos(rotation_angle), readout_len)],
        },
    },
}
