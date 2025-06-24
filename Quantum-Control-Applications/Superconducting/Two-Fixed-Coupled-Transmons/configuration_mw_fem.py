# %%
"""
Octave configuration working for QOP222 and qm-qua==1.1.5 and newer.
"""

from pathlib import Path

import numpy as np
import plotly.io as pio
from qualang_tools.config.waveform_tools import drag_gaussian_pulse_waveforms
from qualang_tools.units import unit

pio.renderers.default = "browser"

#######################
# AUXILIARY FUNCTIONS #
#######################
u = unit(coerce_to_integer=True)

######################
# Network parameters #
######################
qop_ip = "127.0.0.1"  # Write the QM router IP address
cluster_name = None  # Write your cluster_name if version >= QOP220
qop_port = None  # Write the QOP port if version < QOP220
octave_config = None


#############
# Save Path #
#############
# Path to save data
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
# Qubits full scale power
qubit_full_scale_power_dbm_q1 = -8
qubit_full_scale_power_dbm_q2 = -8
# Qubits bands
# The keyword "band" refers to the following frequency bands:
#   1: (50 MHz - 5.5 GHz)
#   2: (4.5 GHz - 7.5 GHz)
#   3: (6.5 GHz - 10.5 GHz)
qubit_band_q1 = 3
qubit_band_q2 = 3
# Qubits LO
qubit_LO_q1 = 8.00 * u.GHz
qubit_LO_q2 = 7.50 * u.GHz
# Qubits IF
qubit_IF_q1 = -100 * u.MHz
qubit_IF_q2 = -200 * u.MHz
# Qubits_delay
qubit_delay_q1 = 0
qubit_delay_q2 = 0

# Relaxation time
qubit1_T1 = int(30 * u.us)
qubit2_T1 = int(30 * u.us)
thermalization_time = 5 * max(qubit1_T1, qubit2_T1)

# CW pulse parameter
const_len = 1000
const_amp = 0.25

# Pi pulse parameters
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
# No DRAG when alpha=0, it's just a gaussian.

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
# No DRAG when alpha=0, it's just a gaussian.

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
# No DRAG when alpha=0, it's just a gaussian.

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
# No DRAG when alpha=0, it's just a gaussian.

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
# No DRAG when alpha=0, it's just a gaussian.

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
# No DRAG when alpha=0, it's just a gaussian.


#############################################
#              Cross Resonance              #
#############################################

# CR Drive full scale power
cr_drive_full_scale_power_dbm_c1t2 = -2
cr_drive_full_scale_power_dbm_c2t1 = -2
# CR Cancel full scale power
cr_cancel_full_scale_power_dbm_c1t2 = -8
cr_cancel_full_scale_power_dbm_c2t1 = -8

# CR Drive bands
cr_drive_band_c1t2 = qubit_band_q2
cr_drive_band_c2t1 = qubit_band_q1
# CR Cancel bands
cr_cancel_band_c1t2 = qubit_band_q2
cr_cancel_band_c2t1 = qubit_band_q1

# CR Drive LO
cr_drive_LO_c1t2 = qubit_LO_q2
cr_drive_LO_c2t1 = qubit_LO_q1
# CR Cancel LO
cr_cancel_LO_c1t2 = qubit_LO_q2
cr_cancel_LO_c2t1 = qubit_LO_q1

# CR Drive IF
cr_drive_IF_c1t2 = qubit_IF_q2
cr_drive_IF_c2t1 = qubit_IF_q1
# CR Cancel IF
cr_cancel_IF_c1t2 = qubit_IF_q2
cr_cancel_IF_c2t1 = qubit_IF_q1

# CR Drive pulse len
cr_drive_square_len_c1t2 = 120
cr_drive_square_len_c2t1 = 120
# CR Cancel pulse len
cr_cancel_square_len_c1t2 = cr_drive_square_len_c1t2
cr_cancel_square_len_c2t1 = cr_drive_square_len_c2t1

# CR Drive pulse amp
cr_drive_square_amp_c1t2 = 0.5
cr_drive_square_amp_c2t1 = 0.5
# CR Cancel pulse amp
cr_cancel_square_amp_c1t2 = 0.5
cr_cancel_square_amp_c2t1 = 0.5

# CR Drive pulse phase
cr_drive_square_phase_c1t2 = 0.0  # in units of 2pi
cr_drive_square_phase_c2t1 = 0.0  # in units of 2pi
# CR Cancel pulse phase
cr_cancel_square_phase_c1t2 = 0.0  # in units of 2pi
cr_cancel_square_phase_c2t1 = 0.0  # in units of 2pi

# CR Drive pulse phase
cr_drive_square_phase_ZI_correct_c1t2 = 0.0  # in units of 2pi
cr_drive_square_phase_ZI_correct_c2t1 = 0.0  # in units of 2pi


#############################################
#                Resonators                 #
#############################################
# Qubits full scale power
resonator_full_scale_power_dbm = -20
# Qubits bands
# The keyword "band" refers to the following frequency bands:
#   1: (50 MHz - 5.5 GHz)
#   2: (4.5 GHz - 7.5 GHz)
#   3: (6.5 GHz - 10.5 GHz)
resonator_band = 3
# Resonators LO
resonator_LO = 10.0 * u.GHz
# Resonators IF
resonator_IF_q1 = int(100 * u.MHz)
resonator_IF_q2 = int(200 * u.MHz)
# resontor_delay
resonator_delay = 0

# Readout pulse parameters
readout_len = 1000
readout_amp_q1 = 0.1
readout_amp_q2 = 0.1

# TOF and depletion time
time_of_flight = 28  # must be a multiple of 4
depletion_time = 2 * u.us

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

# state discrimination
rotation_angle_q1 = (0.0 / 180) * np.pi
rotation_angle_q2 = (0.0 / 180) * np.pi
ge_threshold_q1 = 0.0
ge_threshold_q2 = 0.0

#############################################
#                  Config                   #
#############################################
config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1000",
            "fems": {
                1: {
                    # The keyword "band" refers to the following frequency bands:
                    #   1: (50 MHz - 5.5 GHz)
                    #   2: (4.5 GHz - 7.5 GHz)
                    #   3: (6.5 GHz - 10.5 GHz)
                    # The keyword "full_scale_power_dbm" is the maximum power of
                    # normalized pulse waveforms in [-1,1]. To convert to voltage,
                    #   power_mw = 10**(full_scale_power_dbm / 10)
                    #   max_voltage_amp = np.sqrt(2 * power_mw * 50 / 1000)
                    #   amp_in_volts = waveform * max_voltage_amp
                    #   ^ equivalent to OPX+ amp
                    # Its range is -41dBm to +10dBm with 3dBm steps.
                    "type": "MW",
                    "analog_outputs": {
                        1: {
                            "sampling_rate": 1e9,
                            "full_scale_power_dbm": resonator_full_scale_power_dbm,
                            "band": resonator_band,
                            "delay": resonator_delay,
                            "upconverters": {1: {"frequency": resonator_LO}},
                        },  # RL1  0.5V => 4dbm +[+6, -45] 3db spacing
                        2: {
                            "sampling_rate": 1e9,
                            "full_scale_power_dbm": qubit_full_scale_power_dbm_q1,
                            "band": qubit_band_q1,
                            "delay": qubit_delay_q1,
                            "upconverters": {
                                1: {"frequency": qubit_LO_q1},  # cr_cancel_LO_c2t1 = qubit_LO_q1
                                2: {"frequency": cr_drive_LO_c1t2},
                            },
                        },  # q1 XY
                        3: {
                            "sampling_rate": 1e9,
                            "full_scale_power_dbm": qubit_full_scale_power_dbm_q2,
                            "band": qubit_band_q2,
                            "delay": qubit_delay_q2,
                            "upconverters": {
                                1: {"frequency": qubit_LO_q2},  # cr_cancel_LO_c1t2 = qubit_LO_q2
                                2: {"frequency": cr_drive_LO_c2t1},
                            },
                        },  # q2 XY
                    },
                    "analog_inputs": {
                        1: {
                            "sampling_rate": 1e9,
                            "band": resonator_band,
                            "gain_db": 0,
                            "downconverter_frequency": resonator_LO,
                        },  # RL1, gain_db resolution is 1
                    },
                },
            },
        },
    },
    "elements": {
        "rr1": {
            "MWInput": {
                "port": ("con1", 1, 1),
                "upconverter": 1,
            },
            "intermediate_frequency": resonator_IF_q1,  # in Hz [-350e6, +350e6]
            "MWOutput": {
                "port": ("con1", 1, 1),
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
            "operations": {
                "cw": "const_pulse",
                "readout": "readout_pulse_q1",
            },
        },
        "rr2": {
            "MWInput": {
                "port": ("con1", 1, 1),
                "upconverter": 1,
            },
            "intermediate_frequency": resonator_IF_q2,  # in Hz [-350e6, +350e6]
            "MWOutput": {
                "port": ("con1", 1, 1),
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
            "operations": {
                "cw": "const_pulse",
                "readout": "readout_pulse_q2",
            },
        },
        "q1_xy": {
            "MWInput": {
                "port": ("con1", 1, 2),
                "upconverter": 1,
            },
            "intermediate_frequency": qubit_IF_q1,  # in Hz
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
            "MWInput": {
                "port": ("con1", 1, 3),
                "upconverter": 1,
            },
            "intermediate_frequency": qubit_IF_q2,  # in Hz
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
        "cr_drive_c1t2": {
            "MWInput": {
                "port": ("con1", 1, 2),
                "upconverter": 2,
            },
            "intermediate_frequency": cr_drive_IF_c1t2,  # in Hz
            "operations": {
                "cw": "const_pulse",
                "square_positive": "square_positive_pulse_cr_drive_c1t2",
                "square_negative": "square_negative_pulse_cr_drive_c1t2",
            },
        },
        "cr_drive_c2t1": {
            "MWInput": {
                "port": ("con1", 1, 3),
                "upconverter": 2,
            },
            "intermediate_frequency": cr_drive_IF_c2t1,  # in Hz
            "operations": {
                "cw": "const_pulse",
                "square_positive": "square_positive_pulse_cr_drive_c2t1",
                "square_negative": "square_negative_pulse_cr_drive_c2t1",
            },
        },
        "cr_cancel_c1t2": {
            "MWInput": {
                "port": ("con1", 1, 3),
                "upconverter": 2,
            },
            "intermediate_frequency": cr_cancel_IF_c1t2,  # in Hz
            "operations": {
                "cw": "const_pulse",
                "square_positive": "square_positive_pulse_cr_cancel_c1t2",
                "square_negative": "square_negative_pulse_cr_cancel_c1t2",
            },
        },
        "cr_cancel_c2t1": {
            "MWInput": {
                "port": ("con1", 1, 2),
                "upconverter": 2,
            },
            "intermediate_frequency": cr_cancel_IF_c2t1,  # in Hz
            "operations": {
                "cw": "const_pulse",
                "square_positive": "square_positive_pulse_cr_cancel_c2t1",
                "square_negative": "square_negative_pulse_cr_cancel_c2t1",
            },
        },
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
        "x90_pulse_q1": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {
                "I": "x90_I_wf_q1",
                "Q": "x90_Q_wf_q1",
            },
        },
        "x180_pulse_q1": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {
                "I": "x180_I_wf_q1",
                "Q": "x180_Q_wf_q1",
            },
        },
        "-x90_pulse_q1": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {
                "I": "minus_x90_I_wf_q1",
                "Q": "minus_x90_Q_wf_q1",
            },
        },
        "y90_pulse_q1": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {
                "I": "y90_I_wf_q1",
                "Q": "y90_Q_wf_q1",
            },
        },
        "y180_pulse_q1": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {
                "I": "y180_I_wf_q1",
                "Q": "y180_Q_wf_q1",
            },
        },
        "-y90_pulse_q1": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {
                "I": "minus_y90_I_wf_q1",
                "Q": "minus_y90_Q_wf_q1",
            },
        },
        "readout_pulse_q1": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {
                "I": "readout_wf_q1",
                "Q": "zero_wf",
            },
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
        "x90_pulse_q2": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {
                "I": "x90_I_wf_q2",
                "Q": "x90_Q_wf_q2",
            },
        },
        "x180_pulse_q2": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {
                "I": "x180_I_wf_q2",
                "Q": "x180_Q_wf_q2",
            },
        },
        "-x90_pulse_q2": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {
                "I": "minus_x90_I_wf_q2",
                "Q": "minus_x90_Q_wf_q2",
            },
        },
        "y90_pulse_q2": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {
                "I": "y90_I_wf_q2",
                "Q": "y90_Q_wf_q2",
            },
        },
        "y180_pulse_q2": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {
                "I": "y180_I_wf_q2",
                "Q": "y180_Q_wf_q2",
            },
        },
        "-y90_pulse_q2": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {
                "I": "minus_y90_I_wf_q2",
                "Q": "minus_y90_Q_wf_q2",
            },
        },
        "readout_pulse_q2": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {
                "I": "readout_wf_q2",
                "Q": "zero_wf",
            },
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
        "square_positive_pulse_cr_drive_c1t2": {
            "operation": "control",
            "length": cr_drive_square_len_c1t2,
            "waveforms": {"I": "square_positive_wf_cr_drive_c1t2", "Q": "zero_wf"},
        },
        "square_positive_pulse_cr_drive_c2t1": {
            "operation": "control",
            "length": cr_drive_square_len_c2t1,
            "waveforms": {"I": "square_positive_wf_cr_drive_c2t1", "Q": "zero_wf"},
        },
        "square_negative_pulse_cr_drive_c1t2": {
            "operation": "control",
            "length": cr_drive_square_len_c1t2,
            "waveforms": {"I": "square_negative_wf_cr_drive_c1t2", "Q": "zero_wf"},
        },
        "square_negative_pulse_cr_drive_c2t1": {
            "operation": "control",
            "length": cr_drive_square_len_c2t1,
            "waveforms": {"I": "square_negative_wf_cr_drive_c2t1", "Q": "zero_wf"},
        },
        "square_positive_pulse_cr_cancel_c1t2": {
            "operation": "control",
            "length": cr_cancel_square_len_c1t2,
            "waveforms": {"I": "square_positive_wf_cr_cancel_c1t2", "Q": "zero_wf"},
        },
        "square_positive_pulse_cr_cancel_c2t1": {
            "operation": "control",
            "length": cr_cancel_square_len_c2t1,
            "waveforms": {"I": "square_positive_wf_cr_cancel_c2t1", "Q": "zero_wf"},
        },
        "square_negative_pulse_cr_cancel_c1t2": {
            "operation": "control",
            "length": cr_cancel_square_len_c1t2,
            "waveforms": {"I": "square_negative_wf_cr_cancel_c1t2", "Q": "zero_wf"},
        },
        "square_negative_pulse_cr_cancel_c2t1": {
            "operation": "control",
            "length": cr_cancel_square_len_c2t1,
            "waveforms": {"I": "square_negative_wf_cr_cancel_c2t1", "Q": "zero_wf"},
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": const_amp},
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
        "square_positive_wf_cr_drive_c1t2": {"type": "constant", "sample": cr_drive_square_amp_c1t2},
        "square_negative_wf_cr_drive_c1t2": {"type": "constant", "sample": -cr_drive_square_amp_c1t2},
        "square_positive_wf_cr_cancel_c1t2": {"type": "constant", "sample": cr_cancel_square_amp_c1t2},
        "square_negative_wf_cr_cancel_c1t2": {"type": "constant", "sample": -cr_cancel_square_amp_c1t2},
        "square_positive_wf_cr_drive_c2t1": {"type": "constant", "sample": cr_drive_square_amp_c2t1},
        "square_negative_wf_cr_drive_c2t1": {"type": "constant", "sample": -cr_drive_square_amp_c2t1},
        "square_positive_wf_cr_cancel_c2t1": {"type": "constant", "sample": cr_cancel_square_amp_c2t1},
        "square_negative_wf_cr_cancel_c2t1": {"type": "constant", "sample": -cr_cancel_square_amp_c2t1},
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
        "opt_cosine_weights_q1": {
            "cosine": opt_weights_real_q1,
            "sine": opt_weights_minus_imag_q1,
        },
        "opt_sine_weights_q1": {
            "cosine": opt_weights_imag_q1,
            "sine": opt_weights_real_q1,
        },
        "opt_minus_sine_weights_q1": {
            "cosine": opt_weights_minus_imag_q1,
            "sine": opt_weights_minus_real_q1,
        },
        "opt_cosine_weights_q2": {
            "cosine": opt_weights_real_q2,
            "sine": opt_weights_minus_imag_q2,
        },
        "opt_sine_weights_q2": {
            "cosine": opt_weights_imag_q2,
            "sine": opt_weights_real_q2,
        },
        "opt_minus_sine_weights_q2": {
            "cosine": opt_weights_minus_imag_q2,
            "sine": opt_weights_minus_real_q2,
        },
    },
}

# %%
