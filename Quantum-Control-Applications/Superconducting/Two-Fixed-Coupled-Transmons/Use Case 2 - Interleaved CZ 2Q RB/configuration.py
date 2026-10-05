"""
Two-Fixed-Coupled-Transmons — Use Case 3: Interleaved CZ 2Q RB
================================================================
Minimal OPX+ configuration for CZ-based two-qubit randomized benchmarking.

Elements:
    q1_xy  — qubit 1 XY drive (ports 3-4)
    q2_xy  — qubit 2 XY drive (ports 7-8)
    rr1    — readout resonator for q1 (ports 5-6, shared)
    rr2    — readout resonator for q2 (ports 5-6, shared)
    tc12   — tunable coupler flux line for CZ gate (port 9, single-ended)

Port mapping (OPX+ con1):
    Analog outputs 3-4: q1_xy I/Q
    Analog outputs 5-6: readout feedline I/Q (multiplexed rr1 + rr2)
    Analog outputs 7-8: q2_xy I/Q
    Analog output  9:   tc12 coupler flux line (CZ gate)
    Analog inputs  1-2: readout return I/Q
"""

from pathlib import Path

import numpy as np
from qualang_tools.config.waveform_tools import drag_gaussian_pulse_waveforms
from qualang_tools.units import unit

#######################
# AUXILIARY FUNCTIONS #
#######################
u = unit(coerce_to_integer=True)


def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


######################
# Network parameters #
######################
qop_ip = "172.16.33.101"
cluster_name = "CS_2"
qop_port = None
octave_config = None

#############
# Save Path #
#############
save_dir = Path(__file__).parent.resolve() / "Data"
save_dir.mkdir(exist_ok=True)

default_additional_files = {
    Path(__file__).name: Path(__file__).name,
    "optimal_weights.npz": "optimal_weights.npz",
}

#####################
# OPX configuration #
#####################

#############################################
#                  Qubits                   #
#############################################
qubit_LO_q1 = 8.00 * u.GHz
qubit_LO_q2 = 7.50 * u.GHz
qubit_IF_q1 = -100 * u.MHz
qubit_IF_q2 = -200 * u.MHz

# Mixer parameters
mixer_qubit_g_q1 = 0.00
mixer_qubit_g_q2 = 0.00
mixer_qubit_phi_q1 = 0.0
mixer_qubit_phi_q2 = 0.0

# Tunable coupler / flux parking
coupling_off_flux = 0.0

# Relaxation time
qubit1_T1 = int(30 * u.us)
qubit2_T1 = int(30 * u.us)
thermalization_time = 5 * max(qubit1_T1, qubit2_T1)

# CW pulse parameter
const_len = 1000
const_amp = 0.25

# Flux line
const_flux_len = 200
const_flux_amp = 0.45

# Pi pulse parameters (DRAG Gaussian)
pi_len = 40
pi_sigma = pi_len / 5
pi_amp_q1 = 0.22
pi_amp_q2 = 0.22

# DRAG coefficients
drag_coef_q1 = 1.0
drag_coef_q2 = 1.0
anharmonicity_q1 = -200 * u.MHz
anharmonicity_q2 = -180 * u.MHz
AC_stark_detuning_q1 = 0 * u.MHz
AC_stark_detuning_q2 = 0 * u.MHz

# DRAG waveforms
x180_wf_q1, x180_der_wf_q1 = np.array(
    drag_gaussian_pulse_waveforms(pi_amp_q1, pi_len, pi_sigma, drag_coef_q1, anharmonicity_q1, AC_stark_detuning_q1)
)
x180_I_wf_q1 = x180_wf_q1
x180_Q_wf_q1 = x180_der_wf_q1
x180_wf_q2, x180_der_wf_q2 = np.array(
    drag_gaussian_pulse_waveforms(pi_amp_q2, pi_len, pi_sigma, drag_coef_q2, anharmonicity_q2, AC_stark_detuning_q2)
)
x180_I_wf_q2 = x180_wf_q2
x180_Q_wf_q2 = x180_der_wf_q2

x90_wf_q1, x90_der_wf_q1 = np.array(
    drag_gaussian_pulse_waveforms(pi_amp_q1 / 2, pi_len, pi_sigma, drag_coef_q1, anharmonicity_q1, AC_stark_detuning_q1)
)
x90_I_wf_q1 = x90_wf_q1
x90_Q_wf_q1 = x90_der_wf_q1
x90_wf_q2, x90_der_wf_q2 = np.array(
    drag_gaussian_pulse_waveforms(pi_amp_q2 / 2, pi_len, pi_sigma, drag_coef_q2, anharmonicity_q2, AC_stark_detuning_q2)
)
x90_I_wf_q2 = x90_wf_q2
x90_Q_wf_q2 = x90_der_wf_q2

minus_x90_wf_q1, minus_x90_der_wf_q1 = np.array(
    drag_gaussian_pulse_waveforms(
        -pi_amp_q1 / 2, pi_len, pi_sigma, drag_coef_q1, anharmonicity_q1, AC_stark_detuning_q1
    )
)
minus_x90_I_wf_q1 = minus_x90_wf_q1
minus_x90_Q_wf_q1 = minus_x90_der_wf_q1
minus_x90_wf_q2, minus_x90_der_wf_q2 = np.array(
    drag_gaussian_pulse_waveforms(
        -pi_amp_q2 / 2, pi_len, pi_sigma, drag_coef_q2, anharmonicity_q2, AC_stark_detuning_q2
    )
)
minus_x90_I_wf_q2 = minus_x90_wf_q2
minus_x90_Q_wf_q2 = minus_x90_der_wf_q2

y180_wf_q1, y180_der_wf_q1 = np.array(
    drag_gaussian_pulse_waveforms(pi_amp_q1, pi_len, pi_sigma, drag_coef_q1, anharmonicity_q1, AC_stark_detuning_q1)
)
y180_I_wf_q1 = (-1) * y180_der_wf_q1
y180_Q_wf_q1 = y180_wf_q1
y180_wf_q2, y180_der_wf_q2 = np.array(
    drag_gaussian_pulse_waveforms(pi_amp_q2, pi_len, pi_sigma, drag_coef_q2, anharmonicity_q2, AC_stark_detuning_q2)
)
y180_I_wf_q2 = (-1) * y180_der_wf_q2
y180_Q_wf_q2 = y180_wf_q2

y90_wf_q1, y90_der_wf_q1 = np.array(
    drag_gaussian_pulse_waveforms(pi_amp_q1 / 2, pi_len, pi_sigma, drag_coef_q1, anharmonicity_q1, AC_stark_detuning_q1)
)
y90_I_wf_q1 = (-1) * y90_der_wf_q1
y90_Q_wf_q1 = y90_wf_q1
y90_wf_q2, y90_der_wf_q2 = np.array(
    drag_gaussian_pulse_waveforms(pi_amp_q2 / 2, pi_len, pi_sigma, drag_coef_q2, anharmonicity_q2, AC_stark_detuning_q2)
)
y90_I_wf_q2 = (-1) * y90_der_wf_q2
y90_Q_wf_q2 = y90_wf_q2

minus_y90_wf_q1, minus_y90_der_wf_q1 = np.array(
    drag_gaussian_pulse_waveforms(
        -pi_amp_q1 / 2, pi_len, pi_sigma, drag_coef_q1, anharmonicity_q1, AC_stark_detuning_q1
    )
)
minus_y90_I_wf_q1 = (-1) * minus_y90_der_wf_q1
minus_y90_Q_wf_q1 = minus_y90_wf_q1
minus_y90_wf_q2, minus_y90_der_wf_q2 = np.array(
    drag_gaussian_pulse_waveforms(
        -pi_amp_q2 / 2, pi_len, pi_sigma, drag_coef_q2, anharmonicity_q2, AC_stark_detuning_q2
    )
)
minus_y90_I_wf_q2 = (-1) * minus_y90_der_wf_q2
minus_y90_Q_wf_q2 = minus_y90_wf_q2

#############################################
#                Resonators                 #
#############################################
resonator_LO = 10.0 * u.GHz
resonator_IF_q1 = int(100 * u.MHz)
resonator_IF_q2 = int(200 * u.MHz)

readout_len = 1000
readout_amp_q1 = 0.1
readout_amp_q2 = 0.1

time_of_flight = 28
depletion_time = 2 * u.us

mixer_resonator_g_q1 = 0.0
mixer_resonator_g_q2 = 0.0
mixer_resonator_phi_q1 = 0.0
mixer_resonator_phi_q2 = 0.0

opt_weights = False
if opt_weights:
    from qualang_tools.config.integration_weights_tools import convert_integration_weights

    weights_q1 = np.load("optimal_weights_q1.npz")
    opt_weights_real_q1 = convert_integration_weights(weights_q1["weights_real"])
    opt_weights_minus_imag_q1 = convert_integration_weights(weights_q1["weights_minus_imag"])
    opt_weights_imag_q1 = convert_integration_weights(weights_q1["weights_imag"])
    opt_weights_minus_real_q1 = convert_integration_weights(weights_q1["weights_minus_real"])
    weights_q2 = np.load("optimal_weights_q2.npz")
    opt_weights_real_q2 = convert_integration_weights(weights_q2["weights_real"])
    opt_weights_minus_imag_q2 = convert_integration_weights(weights_q2["weights_minus_imag"])
    opt_weights_imag_q2 = convert_integration_weights(weights_q2["weights_imag"])
    opt_weights_minus_real_q2 = convert_integration_weights(weights_q2["weights_minus_real"])
else:
    opt_weights_real_q1 = [(1.0, readout_len)]
    opt_weights_minus_imag_q1 = [(0.0, readout_len)]
    opt_weights_imag_q1 = [(0.0, readout_len)]
    opt_weights_minus_real_q1 = [(-1.0, readout_len)]
    opt_weights_real_q2 = [(1.0, readout_len)]
    opt_weights_minus_imag_q2 = [(0.0, readout_len)]
    opt_weights_imag_q2 = [(0.0, readout_len)]
    opt_weights_minus_real_q2 = [(-1.0, readout_len)]

# State discrimination
rotation_angle_q1 = (0.0 / 180) * np.pi
rotation_angle_q2 = (0.0 / 180) * np.pi
ge_threshold_q1 = 0.0
ge_threshold_q2 = 0.0

#############################################
#                  Config                   #
#############################################
config = {
    "controllers": {
        "con1": {
            "analog_outputs": {
                3: {"offset": 0.0},  # I q1_xy
                4: {"offset": 0.0},  # Q q1_xy
                5: {"offset": 0.0},  # I readout
                6: {"offset": 0.0},  # Q readout
                7: {"offset": 0.0},  # I q2_xy
                8: {"offset": 0.0},  # Q q2_xy
                9: {"offset": coupling_off_flux},  # q1_z flux (CZ gate)
            },
            "digital_outputs": {
                1: {},
            },
            "analog_inputs": {
                1: {"offset": 0.0, "gain_db": 0},
                2: {"offset": 0.0, "gain_db": 0},
            },
        },
    },
    "elements": {
        "rr1": {
            "mixInputs": {
                "I": ("con1", 5),
                "Q": ("con1", 6),
                "lo_frequency": resonator_LO,
                "mixer": "mixer_resonator",
            },
            "intermediate_frequency": resonator_IF_q1,
            "outputs": {
                "out1": ("con1", 1),
                "out2": ("con1", 2),
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
            "operations": {
                "cw": "const_pulse",
                "readout": "readout_pulse_q1",
            },
        },
        "rr2": {
            "mixInputs": {
                "I": ("con1", 5),
                "Q": ("con1", 6),
                "lo_frequency": resonator_LO,
                "mixer": "mixer_resonator",
            },
            "intermediate_frequency": resonator_IF_q2,
            "outputs": {
                "out1": ("con1", 1),
                "out2": ("con1", 2),
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
            "operations": {
                "cw": "const_pulse",
                "readout": "readout_pulse_q2",
            },
        },
        "q1_xy": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": qubit_LO_q1,
                "mixer": "mixer_qubit_q1",
            },
            "intermediate_frequency": qubit_IF_q1,
            "operations": {
                "cw": "const_pulse",
                "x180": "x180_pulse_q1",
                "x90": "x90_pulse_q1",
                "-x90": "-x90_pulse_q1",
                "y90": "y90_pulse_q1",
                "y180": "y180_pulse_q1",
                "-y90": "-y90_pulse_q1",
            },
        },
        "q2_xy": {
            "mixInputs": {
                "I": ("con1", 7),
                "Q": ("con1", 8),
                "lo_frequency": qubit_LO_q2,
                "mixer": "mixer_qubit_q2",
            },
            "intermediate_frequency": qubit_IF_q2,
            "operations": {
                "cw": "const_pulse",
                "x180": "x180_pulse_q2",
                "x90": "x90_pulse_q2",
                "-x90": "-x90_pulse_q2",
                "y90": "y90_pulse_q2",
                "y180": "y180_pulse_q2",
                "-y90": "-y90_pulse_q2",
            },
        },
        "tc12": {
            "singleInput": {
                "port": ("con1", 9),
            },
            "operations": {
                "const": "const_flux_pulse",
            },
        },
    },
    "pulses": {
        "const_pulse": {
            "operation": "control",
            "length": const_len,
            "waveforms": {"I": "const_wf", "Q": "zero_wf"},
        },
        "const_flux_pulse": {
            "operation": "control",
            "length": const_flux_len,
            "waveforms": {"single": "const_flux_wf"},
        },
        "x90_pulse_q1": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {"I": "x90_I_wf_q1", "Q": "x90_Q_wf_q1"},
        },
        "x180_pulse_q1": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {"I": "x180_I_wf_q1", "Q": "x180_Q_wf_q1"},
        },
        "-x90_pulse_q1": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {"I": "minus_x90_I_wf_q1", "Q": "minus_x90_Q_wf_q1"},
        },
        "y90_pulse_q1": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {"I": "y90_I_wf_q1", "Q": "y90_Q_wf_q1"},
        },
        "y180_pulse_q1": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {"I": "y180_I_wf_q1", "Q": "y180_Q_wf_q1"},
        },
        "-y90_pulse_q1": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {"I": "minus_y90_I_wf_q1", "Q": "minus_y90_Q_wf_q1"},
        },
        "x90_pulse_q2": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {"I": "x90_I_wf_q2", "Q": "x90_Q_wf_q2"},
        },
        "x180_pulse_q2": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {"I": "x180_I_wf_q2", "Q": "x180_Q_wf_q2"},
        },
        "-x90_pulse_q2": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {"I": "minus_x90_I_wf_q2", "Q": "minus_x90_Q_wf_q2"},
        },
        "y90_pulse_q2": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {"I": "y90_I_wf_q2", "Q": "y90_Q_wf_q2"},
        },
        "y180_pulse_q2": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {"I": "y180_I_wf_q2", "Q": "y180_Q_wf_q2"},
        },
        "-y90_pulse_q2": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {"I": "minus_y90_I_wf_q2", "Q": "minus_y90_Q_wf_q2"},
        },
        "readout_pulse_q1": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {"I": "readout_wf_q1", "Q": "zero_wf"},
            "integration_weights": {
                "cos": "cosine_weights",
                "sin": "sine_weights",
                "minus_sin": "minus_sine_weights",
                "rotated_cos": "rotated_cosine_weights_q1",
                "rotated_sin": "rotated_sine_weights_q1",
                "rotated_minus_sin": "rotated_minus_sine_weights_q1",
                "opt_cos": "opt_cosine_weights_q1",
                "opt_sin": "opt_sine_weights_q1",
                "opt_minus_sin": "opt_minus_sine_weights_q1",
            },
            "digital_marker": "ON",
        },
        "readout_pulse_q2": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {"I": "readout_wf_q2", "Q": "zero_wf"},
            "integration_weights": {
                "cos": "cosine_weights",
                "sin": "sine_weights",
                "minus_sin": "minus_sine_weights",
                "rotated_cos": "rotated_cosine_weights_q2",
                "rotated_sin": "rotated_sine_weights_q2",
                "rotated_minus_sin": "rotated_minus_sine_weights_q2",
                "opt_cos": "opt_cosine_weights_q2",
                "opt_sin": "opt_sine_weights_q2",
                "opt_minus_sin": "opt_minus_sine_weights_q2",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": const_amp},
        "const_flux_wf": {"type": "constant", "sample": const_flux_amp},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "x90_I_wf_q1": {"type": "arbitrary", "samples": x90_I_wf_q1.tolist()},
        "x90_Q_wf_q1": {"type": "arbitrary", "samples": x90_Q_wf_q1.tolist()},
        "x180_I_wf_q1": {"type": "arbitrary", "samples": x180_I_wf_q1.tolist()},
        "x180_Q_wf_q1": {"type": "arbitrary", "samples": x180_Q_wf_q1.tolist()},
        "minus_x90_I_wf_q1": {"type": "arbitrary", "samples": minus_x90_I_wf_q1.tolist()},
        "minus_x90_Q_wf_q1": {"type": "arbitrary", "samples": minus_x90_Q_wf_q1.tolist()},
        "y90_I_wf_q1": {"type": "arbitrary", "samples": y90_I_wf_q1.tolist()},
        "y90_Q_wf_q1": {"type": "arbitrary", "samples": y90_Q_wf_q1.tolist()},
        "y180_I_wf_q1": {"type": "arbitrary", "samples": y180_I_wf_q1.tolist()},
        "y180_Q_wf_q1": {"type": "arbitrary", "samples": y180_Q_wf_q1.tolist()},
        "minus_y90_I_wf_q1": {"type": "arbitrary", "samples": minus_y90_I_wf_q1.tolist()},
        "minus_y90_Q_wf_q1": {"type": "arbitrary", "samples": minus_y90_Q_wf_q1.tolist()},
        "readout_wf_q1": {"type": "constant", "sample": readout_amp_q1},
        "x90_I_wf_q2": {"type": "arbitrary", "samples": x90_I_wf_q2.tolist()},
        "x90_Q_wf_q2": {"type": "arbitrary", "samples": x90_Q_wf_q2.tolist()},
        "x180_I_wf_q2": {"type": "arbitrary", "samples": x180_I_wf_q2.tolist()},
        "x180_Q_wf_q2": {"type": "arbitrary", "samples": x180_Q_wf_q2.tolist()},
        "minus_x90_I_wf_q2": {"type": "arbitrary", "samples": minus_x90_I_wf_q2.tolist()},
        "minus_x90_Q_wf_q2": {"type": "arbitrary", "samples": minus_x90_Q_wf_q2.tolist()},
        "y90_I_wf_q2": {"type": "arbitrary", "samples": y90_I_wf_q2.tolist()},
        "y90_Q_wf_q2": {"type": "arbitrary", "samples": y90_Q_wf_q2.tolist()},
        "y180_I_wf_q2": {"type": "arbitrary", "samples": y180_I_wf_q2.tolist()},
        "y180_Q_wf_q2": {"type": "arbitrary", "samples": y180_Q_wf_q2.tolist()},
        "minus_y90_I_wf_q2": {"type": "arbitrary", "samples": minus_y90_I_wf_q2.tolist()},
        "minus_y90_Q_wf_q2": {"type": "arbitrary", "samples": minus_y90_Q_wf_q2.tolist()},
        "readout_wf_q2": {"type": "constant", "sample": readout_amp_q2},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "cosine_weights": {"cosine": [(1.0, readout_len)], "sine": [(0.0, readout_len)]},
        "sine_weights": {"cosine": [(0.0, readout_len)], "sine": [(1.0, readout_len)]},
        "minus_sine_weights": {"cosine": [(0.0, readout_len)], "sine": [(-1.0, readout_len)]},
        "rotated_cosine_weights_q1": {
            "cosine": [(np.cos(rotation_angle_q1), readout_len)],
            "sine": [(np.sin(rotation_angle_q1), readout_len)],
        },
        "rotated_sine_weights_q1": {
            "cosine": [(-np.sin(rotation_angle_q1), readout_len)],
            "sine": [(np.cos(rotation_angle_q1), readout_len)],
        },
        "rotated_minus_sine_weights_q1": {
            "cosine": [(np.sin(rotation_angle_q1), readout_len)],
            "sine": [(-np.cos(rotation_angle_q1), readout_len)],
        },
        "rotated_cosine_weights_q2": {
            "cosine": [(np.cos(rotation_angle_q2), readout_len)],
            "sine": [(np.sin(rotation_angle_q2), readout_len)],
        },
        "rotated_sine_weights_q2": {
            "cosine": [(-np.sin(rotation_angle_q2), readout_len)],
            "sine": [(np.cos(rotation_angle_q2), readout_len)],
        },
        "rotated_minus_sine_weights_q2": {
            "cosine": [(np.sin(rotation_angle_q2), readout_len)],
            "sine": [(-np.cos(rotation_angle_q2), readout_len)],
        },
        "opt_cosine_weights_q1": {"cosine": opt_weights_real_q1, "sine": opt_weights_minus_imag_q1},
        "opt_sine_weights_q1": {"cosine": opt_weights_imag_q1, "sine": opt_weights_real_q1},
        "opt_minus_sine_weights_q1": {"cosine": opt_weights_minus_imag_q1, "sine": opt_weights_minus_real_q1},
        "opt_cosine_weights_q2": {"cosine": opt_weights_real_q2, "sine": opt_weights_minus_imag_q2},
        "opt_sine_weights_q2": {"cosine": opt_weights_imag_q2, "sine": opt_weights_real_q2},
        "opt_minus_sine_weights_q2": {"cosine": opt_weights_minus_imag_q2, "sine": opt_weights_minus_real_q2},
    },
    "mixers": {
        "mixer_qubit_q1": [
            {
                "intermediate_frequency": qubit_IF_q1,
                "lo_frequency": qubit_LO_q1,
                "correction": IQ_imbalance(mixer_qubit_g_q1, mixer_qubit_phi_q1),
            },
        ],
        "mixer_qubit_q2": [
            {
                "intermediate_frequency": qubit_IF_q2,
                "lo_frequency": qubit_LO_q2,
                "correction": IQ_imbalance(mixer_qubit_g_q2, mixer_qubit_phi_q2),
            },
        ],
        "mixer_resonator": [
            {
                "intermediate_frequency": resonator_IF_q1,
                "lo_frequency": resonator_LO,
                "correction": IQ_imbalance(mixer_resonator_g_q1, mixer_resonator_phi_q1),
            },
            {
                "intermediate_frequency": resonator_IF_q2,
                "lo_frequency": resonator_LO,
                "correction": IQ_imbalance(mixer_resonator_g_q2, mixer_resonator_phi_q2),
            },
        ],
    },
}
