from quam import QuAM
import numpy as np
from qualang_tools.config.waveform_tools import drag_gaussian_pulse_waveforms
from qualang_tools.units import unit
from qualang_tools.config.waveform_tools import flattop_gaussian_waveform

#########
# PATHS #
#########

from pathlib import Path

save_dir = (Path().absolute() / "TEST" / "BETAsite" / "QM" / "OPXPlus" / "data")


#######################
# AUXILIARY FUNCTIONS #
#######################

# IQ imbalance matrix
def IQ_imbalance(g, phi):
    """
    Creates the correction matrix for the mixer imbalance caused by the gain and phase imbalances, more information can
    be seen here:
    https://docs.qualang.io/libs/examples/mixer-calibration/#non-ideal-mixer
    :param g: relative gain imbalance between the I & Q ports. (unit-less), set to 0 for no gain imbalance.
    :param phi: relative phase imbalance between the I & Q ports (radians), set to 0 for no phase imbalance.
    """
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g ** 2) * (2 * c ** 2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


#############
# VARIABLES #
#############
u = unit()

# Qubits
qubit_LO = 3.95 * u.GHz  # Used only for mixer correction and frequency rescaling for plots or computation

qubit_IF_q1 = 50 * u.MHz
qubit_IF_q2 = 100 * u.MHz
mixer_qubit_g_q1 = 0.00
mixer_qubit_g_q2 = 0.00
mixer_qubit_phi_q1 = 0.0
mixer_qubit_phi_q2 = 0.0

qubit_T1 = int(3 * u.us)

const_len = 100
const_amp = 270 * u.mV

pi_len = 800
pi_sigma = 230
pi_amp_q1 = 0.22
pi_amp_q2 = 0.22
drag_coef_q1 = 0
drag_coef_q2 = 0
anharmonicity_q1 = -200 * u.MHz
anharmonicity_q2 = -180 * u.MHz
AC_stark_detuning_q1 = 0 * u.MHz
AC_stark_detuning_q2 = 0 * u.MHz

x180_wf_q1, x180_der_wf_q1 = np.array(
    drag_gaussian_pulse_waveforms(pi_amp_q1, pi_len, pi_sigma, drag_coef_q1, anharmonicity_q1, AC_stark_detuning_q1))
x180_I_wf_q1 = x180_wf_q1
x180_Q_wf_q1 = x180_der_wf_q1
x180_wf_q2, x180_der_wf_q2 = np.array(
    drag_gaussian_pulse_waveforms(pi_amp_q2, pi_len, pi_sigma, drag_coef_q2, anharmonicity_q2, AC_stark_detuning_q2))
x180_I_wf_q2 = x180_wf_q2
x180_Q_wf_q2 = x180_der_wf_q2
# No DRAG when alpha=0, it's just a gaussian.

x90_wf_q1, x90_der_wf_q1 = np.array(
    drag_gaussian_pulse_waveforms(pi_amp_q1 / 2, pi_len, pi_sigma, drag_coef_q1, anharmonicity_q1,
                                  AC_stark_detuning_q1))
x90_I_wf_q1 = x90_wf_q1
x90_Q_wf_q1 = x90_der_wf_q1
x90_wf_q2, x90_der_wf_q2 = np.array(
    drag_gaussian_pulse_waveforms(pi_amp_q2 / 2, pi_len, pi_sigma, drag_coef_q2, anharmonicity_q2,
                                  AC_stark_detuning_q2))
x90_I_wf_q2 = x90_wf_q2
x90_Q_wf_q2 = x90_der_wf_q2
# No DRAG when alpha=0, it's just a gaussian.

minus_x90_wf_q1, minus_x90_der_wf_q1 = np.array(
    drag_gaussian_pulse_waveforms(-pi_amp_q1 / 2, pi_len, pi_sigma, drag_coef_q1, anharmonicity_q1,
                                  AC_stark_detuning_q1))
minus_x90_I_wf_q1 = minus_x90_wf_q1
minus_x90_Q_wf_q1 = minus_x90_der_wf_q1
minus_x90_wf_q2, minus_x90_der_wf_q2 = np.array(
    drag_gaussian_pulse_waveforms(-pi_amp_q2 / 2, pi_len, pi_sigma, drag_coef_q2, anharmonicity_q2,
                                  AC_stark_detuning_q2))
minus_x90_I_wf_q2 = minus_x90_wf_q2
minus_x90_Q_wf_q2 = minus_x90_der_wf_q2
# No DRAG when alpha=0, it's just a gaussian.

y180_wf_q1, y180_der_wf_q1 = np.array(
    drag_gaussian_pulse_waveforms(pi_amp_q1, pi_len, pi_sigma, drag_coef_q1, anharmonicity_q1, AC_stark_detuning_q1))
y180_I_wf_q1 = (-1) * y180_der_wf_q1
y180_Q_wf_q1 = y180_wf_q1
y180_wf_q2, y180_der_wf_q2 = np.array(
    drag_gaussian_pulse_waveforms(pi_amp_q2, pi_len, pi_sigma, drag_coef_q2, anharmonicity_q2, AC_stark_detuning_q2))
y180_I_wf_q2 = (-1) * y180_der_wf_q2
y180_Q_wf_q2 = y180_wf_q2
# No DRAG when alpha=0, it's just a gaussian.

y90_wf_q1, y90_der_wf_q1 = np.array(
    drag_gaussian_pulse_waveforms(pi_amp_q1 / 2, pi_len, pi_sigma, drag_coef_q1, anharmonicity_q1,
                                  AC_stark_detuning_q1))
y90_I_wf_q1 = (-1) * y90_der_wf_q1
y90_Q_wf_q1 = y90_wf_q1
y90_wf_q2, y90_der_wf_q2 = np.array(
    drag_gaussian_pulse_waveforms(pi_amp_q2 / 2, pi_len, pi_sigma, drag_coef_q2, anharmonicity_q2,
                                  AC_stark_detuning_q2))
y90_I_wf_q2 = (-1) * y90_der_wf_q2
y90_Q_wf_q2 = y90_wf_q2
# No DRAG when alpha=0, it's just a gaussian.

minus_y90_wf_q1, minus_y90_der_wf_q1 = np.array(
    drag_gaussian_pulse_waveforms(-pi_amp_q1 / 2, pi_len, pi_sigma, drag_coef_q1, anharmonicity_q1,
                                  AC_stark_detuning_q1))
minus_y90_I_wf_q1 = (-1) * minus_y90_der_wf_q1
minus_y90_Q_wf_q1 = minus_y90_wf_q1
minus_y90_wf_q2, minus_y90_der_wf_q2 = np.array(
    drag_gaussian_pulse_waveforms(-pi_amp_q2 / 2, pi_len, pi_sigma, drag_coef_q2, anharmonicity_q2,
                                  AC_stark_detuning_q2))
minus_y90_I_wf_q2 = (-1) * minus_y90_der_wf_q2
minus_y90_Q_wf_q2 = minus_y90_wf_q2
# No DRAG when alpha=0, it's just a gaussian.

# Resonators
resonator_LO = 6.35 * u.GHz  # Used only for mixer correction and frequency rescaling for plots or computation
LO_MHz = resonator_LO / u.MHz
resonator_IF_q1 = int(75 * u.MHz)
resonator_IF_q2 = int(133 * u.MHz)
mixer_resonator_g_q1 = 0.0
mixer_resonator_g_q2 = 0.0
mixer_resonator_phi_q1 = -0.00
mixer_resonator_phi_q2 = -0.00

readout_len = 4000
readout_amp_q1 = 0.07
readout_amp_q2 = 0.07

time_of_flight = 260  # should be a multiple of 4

# Flux line
const_flux_len = 200
const_flux_amp = 0.45

# state discrimination
rotation_angle_q1 = (0.0 / 180) * np.pi
rotation_angle_q2 = (0.0 / 180) * np.pi
ge_threshold_q1 = 0.0
ge_threshold_q2 = 0.0

def build_config(quam: QuAM):
    config = {
        "version": 1,
        "controllers": {
            "con1": {
                "analog_outputs": {
                    1: {"offset": 0.0},  # I qubit1 XY
                    2: {"offset": 0.0},  # Q qubit1 XY
                    3: {"offset": 0.0},  # I qubit2 XY
                    4: {"offset": 0.0},  # Q qubit2 XY
                    5: {"offset": quam.resonators[0].mixer_correction.offset_I},  # I readout line
                    6: {"offset": quam.resonators[0].mixer_correction.offset_Q},  # Q readout line
                    7: {"offset": 0.0},  # qubit1 Z
                    8: {"offset": 0.0},  # qubit2 Z
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
            **{f"rr{i}": {
                "mixInputs": {
                    "I": ("con1", quam.resonators[i].wiring.I),
                    "Q": ("con1", quam.resonators[i].wiring.Q),
                    "lo_frequency": quam.local_oscillators.readout[0].freq * 1e9,
                    "mixer": f"mixer_resonator{i}",
                },
                "intermediate_frequency": (quam.resonators[i].f_res - quam.local_oscillators.readout[0].freq) * 1e9,
                "operations": {
                    "cw": "const_pulse",
                    "readout": "readout_pulse_q1",
                },
                "outputs": {
                    "out1": ("con1", 1),
                    "out2": ("con1", 2),
                },
                "time_of_flight": quam.resonators[i].time_of_flight,
                "smearing": 0,
            } for i in range(len(quam.resonators))
            },
            "q1_xy": {
                "mixInputs": {
                    "I": ("con1", 3),
                    "Q": ("con1", 4),
                    "lo_frequency": qubit_LO,
                    "mixer": "mixer_qubit_q1",
                },
                "intermediate_frequency": qubit_IF_q1,  # frequency at offset ch7 (max freq)
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
                    "I": ("con1", 5),
                    "Q": ("con1", 6),
                    "lo_frequency": qubit_LO,
                    "mixer": "mixer_qubit_q2",
                },
                "intermediate_frequency": qubit_IF_q2,  # frequency at offset ch8 (max freq)
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
            "q1_z": {
                "singleInput": {
                    "port": ("con1", 7),
                },
                "operations": {
                    "const": "const_flux_pulse",
                },
            },
            "q2_z": {
                "singleInput": {
                    "port": ("con1", 8),
                },
                "operations": {
                    "const": "const_flux_pulse",
                },
            },
        },
        "pulses": {
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
            "x90_pulse_q1": {
                "operation": "control",
                "length": pi_len,
                "waveforms": {
                    "I": "x90_wf_q1",
                    "Q": "x90_der_wf_q1",
                },
            },
            "x180_pulse_q1": {
                "operation": "control",
                "length": pi_len,
                "waveforms": {
                    "I": "x180_wf_q1",
                    "Q": "x180_der_wf_q1",
                },
            },
            "-x90_pulse_q1": {
                "operation": "control",
                "length": pi_len,
                "waveforms": {
                    "I": "minus_x90_wf_q1",
                    "Q": "minus_x90_der_wf_q1",
                },
            },
            "y90_pulse_q1": {
                "operation": "control",
                "length": pi_len,
                "waveforms": {
                    "I": "y90_der_wf_q1",
                    "Q": "y90_wf_q1",
                },
            },
            "y180_pulse_q1": {
                "operation": "control",
                "length": pi_len,
                "waveforms": {
                    "I": "y180_der_wf_q1",
                    "Q": "y180_wf_q1",
                },
            },
            "-y90_pulse_q1": {
                "operation": "control",
                "length": pi_len,
                "waveforms": {
                    "I": "minus_y90_der_wf_q1",
                    "Q": "minus_y90_wf_q1",
                },
            },
            **{
                f"readout_pulse_q{i}": {
                    "operation": "measurement",
                    "length": quam.resonators[i].readout_pulse_length,
                    "waveforms": {
                        "I": f"readout{i}_wf",
                        "Q": "zero_wf",
                    },
                    "integration_weights": {
                        "cos": f"cosine_weights{i}",
                        "sin": f"sine_weights{i}",
                        "minus_sin": f"minus_sine_weights{i}",
                        "rotated_cos": f"rotated_cosine_weights{i}",
                        "rotated_sin": f"rotated_sine_weights{i}",
                        "rotated_minus_sin": f"rotated_minus_sine_weights{i}",
                    },
                    "digital_marker": "ON",
                } for i in range(len(quam.resonators))
            },
            "x90_pulse_q2": {
                "operation": "control",
                "length": pi_len,
                "waveforms": {
                    "I": "x90_wf_q2",
                    "Q": "x90_der_wf_q2",
                },
            },
            "x180_pulse_q2": {
                "operation": "control",
                "length": pi_len,
                "waveforms": {
                    "I": "x180_wf_q2",
                    "Q": "x180_der_wf_q2",
                },
            },
            "-x90_pulse_q2": {
                "operation": "control",
                "length": pi_len,
                "waveforms": {
                    "I": "minus_x90_wf_q2",
                    "Q": "minus_x90_der_wf_q2",
                },
            },
            "y90_pulse_q2": {
                "operation": "control",
                "length": pi_len,
                "waveforms": {
                    "I": "y90_der_wf_q2",
                    "Q": "y90_wf_q2",
                },
            },
            "y180_pulse_q2": {
                "operation": "control",
                "length": pi_len,
                "waveforms": {
                    "I": "y180_der_wf_q2",
                    "Q": "y180_wf_q2",
                },
            },
            "-y90_pulse_q2": {
                "operation": "control",
                "length": pi_len,
                "waveforms": {
                    "I": "minus_y90_der_wf_q2",
                    "Q": "minus_y90_wf_q2",
                },
            },
        },
        "waveforms": {
            "const_wf": {"type": "constant", "sample": const_amp},
            "const_flux_wf": {"type": "constant", "sample": const_flux_amp},
            "zero_wf": {"type": "constant", "sample": 0.0},
            "x90_wf_q1": {"type": "arbitrary", "samples": x90_wf_q1.tolist()},
            "x90_der_wf_q1": {"type": "arbitrary", "samples": x90_der_wf_q1.tolist()},
            "x180_wf_q1": {"type": "arbitrary", "samples": x180_wf_q1.tolist()},
            "x180_der_wf_q1": {"type": "arbitrary", "samples": x180_der_wf_q1.tolist()},
            "minus_x90_wf_q1": {"type": "arbitrary", "samples": minus_x90_wf_q1.tolist()},
            "minus_x90_der_wf_q1": {"type": "arbitrary", "samples": minus_x90_der_wf_q1.tolist()},
            "y90_wf_q1": {"type": "arbitrary", "samples": y90_wf_q1.tolist()},
            "y90_der_wf_q1": {"type": "arbitrary", "samples": y90_der_wf_q1.tolist()},
            "y180_wf_q1": {"type": "arbitrary", "samples": y180_wf_q1.tolist()},
            "y180_der_wf_q1": {"type": "arbitrary", "samples": y180_der_wf_q1.tolist()},
            "minus_y90_wf_q1": {"type": "arbitrary", "samples": minus_y90_wf_q1.tolist()},
            "minus_y90_der_wf_q1": {"type": "arbitrary", "samples": minus_y90_der_wf_q1.tolist()},
            **{
                f"readout{i}_wf": {"type": "constant", "sample": quam.resonators[i].readout_pulse_amp} for i in range(len(quam.resonators))
            },
            "x90_wf_q2": {"type": "arbitrary", "samples": x90_wf_q2.tolist()},
            "x90_der_wf_q2": {"type": "arbitrary", "samples": x90_der_wf_q2.tolist()},
            "x180_wf_q2": {"type": "arbitrary", "samples": x180_wf_q2.tolist()},
            "x180_der_wf_q2": {"type": "arbitrary", "samples": x180_der_wf_q2.tolist()},
            "minus_x90_wf_q2": {"type": "arbitrary", "samples": minus_x90_wf_q2.tolist()},
            "minus_x90_der_wf_q2": {"type": "arbitrary", "samples": minus_x90_der_wf_q2.tolist()},
            "y90_wf_q2": {"type": "arbitrary", "samples": y90_wf_q2.tolist()},
            "y90_der_wf_q2": {"type": "arbitrary", "samples": y90_der_wf_q2.tolist()},
            "y180_wf_q2": {"type": "arbitrary", "samples": y180_wf_q2.tolist()},
            "y180_der_wf_q2": {"type": "arbitrary", "samples": y180_der_wf_q2.tolist()},
            "minus_y90_wf_q2": {"type": "arbitrary", "samples": minus_y90_wf_q2.tolist()},
            "minus_y90_der_wf_q2": {"type": "arbitrary", "samples": minus_y90_der_wf_q2.tolist()},
        },
        "digital_waveforms": {
            "ON": {"samples": [(1, 0)]},
        },
        "integration_weights": {
            **{
                f"rotated_cosine_weights{i}": {
                    "cosine": [(1.0, quam.resonators[i].readout_pulse_length)],
                    "sine": [(0.0, quam.resonators[i].readout_pulse_length)],
                } for i in range(len(quam.resonators))
            },
            **{
                f"rotated_sine_weights{i}": {
                    "cosine": [(0.0, quam.resonators[i].readout_pulse_length)],
                    "sine": [(1.0, quam.resonators[i].readout_pulse_length)],
                } for i in range(len(quam.resonators))
            },
            **{
                f"rotated_minus_sine_weights{i}": {
                    "cosine": [(0.0, quam.resonators[i].readout_pulse_length)],
                    "sine": [(-1.0, quam.resonators[i].readout_pulse_length)],
                } for i in range(len(quam.resonators))
            },
            **{
                f"rotated_cosine_weights{i}": {
                    "cosine": [(np.cos(quam.resonators[i].rotation_angle), quam.resonators[i].readout_pulse_length)],
                    "sine": [(-np.sin(quam.resonators[i].rotation_angle), quam.resonators[i].readout_pulse_length)],
                } for i in range(len(quam.resonators))
            },
            **{
                f"rotated_sine_weights{i}": {
                    "cosine": [(np.sin(quam.resonators[i].rotation_angle), quam.resonators[i].readout_pulse_length)],
                    "sine": [(np.cos(quam.resonators[i].rotation_angle), quam.resonators[i].readout_pulse_length)],
                } for i in range(len(quam.resonators))
            },
            **{
                f"rotated_minus_sine_weights{i}": {
                    "cosine": [(-np.sin(quam.resonators[i].rotation_angle), quam.resonators[i].readout_pulse_length)],
                    "sine": [(-np.cos(quam.resonators[i].rotation_angle), quam.resonators[i].readout_pulse_length)],
                } for i in range(len(quam.resonators))
            },
        },
        "mixers": {
            "mixer_qubit_q1": [
                {
                    "intermediate_frequency": qubit_IF_q1,
                    "lo_frequency": qubit_LO,
                    "correction": IQ_imbalance(mixer_qubit_g_q1, mixer_qubit_phi_q1),
                }
            ],
            "mixer_qubit_q2": [
                {
                    "intermediate_frequency": qubit_IF_q2,
                    "lo_frequency": qubit_LO,
                    "correction": IQ_imbalance(mixer_qubit_g_q2, mixer_qubit_phi_q2),
                }
            ],
            **{
                f"mixer_resonator{i}": [
                    {
                        "intermediate_frequency": (quam.resonators[i].f_res - quam.local_oscillators.readout[
                            0].freq) * 1e9,
                        "lo_frequency": quam.local_oscillators.readout[0].freq,
                        "correction": IQ_imbalance(quam.resonators[i].mixer_correction.gain,
                                                   quam.resonators[i].mixer_correction.phase),
                    },
                ] for i in range(len(quam.resonators))
            },

        },
    }
    return config
