import numpy as np
from scipy.signal.windows import gaussian
from qualang_tools.config.waveform_tools import drag_gaussian_pulse_waveforms
from qualang_tools.units import unit
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter, fetching_tool
from quam import QuAM

#######################
# AUXILIARY FUNCTIONS #
#######################


# IQ imbalance matrix
def IQ_imbalance(g, phi):
    """
    Creates the correction matrix for the mixer imbalance caused by the gain and phase imbalances, more information can
    be seen here:
    https://docs.qualang.io/libs/examples/mixer-calibration/#non-ideal-mixer

    :param g: relative gain imbalance between the I & Q ports (unit-less). Set to 0 for no gain imbalance.
    :param phi: relative phase imbalance between the I & Q ports (radians). Set to 0 for no phase imbalance.
    """
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


#############
# VARIABLES #
#############
u = unit()

# Qubits
mixer_qubit_g = 0.0
mixer_qubit_phi = 0.0

const_len = 100
const_amp = 0.1
square_pi_len = 100
square_pi_amp = 0.1

gauss_len = 16
gauss_sigma = gauss_len / 5
gauss_amp = 0.4
gauss_wf = gauss_amp * gaussian(gauss_len, gauss_sigma)

# Resonator
mixer_resonator_g = 0.0
mixer_resonator_phi = 0.0


def build_config(quam: QuAM):
    qubit = quam.qubits[0]
    resonator = quam.resonators[0]
    storage = quam.storage[0]

    drag_coef = qubit.pulse_params.drag_coef
    anharmonicity = qubit.anharmonicity * u.MHz
    AC_stark_detuning = 0 * u.MHz

    saturation_len = qubit.pulse_params.saturation_len * 1e9
    saturation_amp = qubit.pulse_params.saturation_amp

    storage_sat_len = storage.saturation_params.length * 1e9
    storage_sat_amp = storage.saturation_params.amplitude

    displace_len = storage.displacement_params.length
    displace_sigma = displace_len / 5
    displace_amp = storage.displacement_params.amplitude
    displace_wf = displace_amp * gaussian(displace_len, displace_sigma)

    x180_len = qubit.pulse_params.length * 1e9
    x180_sigma = qubit.pulse_params.length * 1e9 / 5
    x180_amp = qubit.pulse_params.amplitude.x180
    x180_wf, x180_der_wf = np.array(
        drag_gaussian_pulse_waveforms(x180_amp, x180_len, x180_sigma, drag_coef, anharmonicity, AC_stark_detuning)
    )
    x180_I_wf = x180_wf
    x180_Q_wf = x180_der_wf
    # No DRAG when alpha=0, it's just a gaussian.

    x90_len = qubit.pulse_params.length * 1e9
    x90_sigma = qubit.pulse_params.length * 1e9 / 5
    x90_amp = qubit.pulse_params.amplitude.x90
    x90_wf, x90_der_wf = np.array(
        drag_gaussian_pulse_waveforms(x90_amp, x90_len, x90_sigma, drag_coef, anharmonicity, AC_stark_detuning)
    )
    x90_I_wf = x90_wf
    x90_Q_wf = x90_der_wf
    # No DRAG when alpha=0, it's just a gaussian.

    short_x90_len = qubit.pulse_params.short_pi_len *1e9
    short_x90_sigma = qubit.pulse_params.short_pi_len *1e9/5
    short_x90_amp = qubit.pulse_params.short_pi_amp / 2
    short_x90_wf, short_x90_der_wf = np.array(
        drag_gaussian_pulse_waveforms(short_x90_amp, short_x90_len, short_x90_sigma, drag_coef, anharmonicity, AC_stark_detuning)
    )
    short_x90_I_wf = short_x90_wf
    short_x90_Q_wf = short_x90_der_wf


    minus_x90_len = qubit.pulse_params.length * 1e9
    minus_x90_sigma = qubit.pulse_params.length * 1e9 / 5
    minus_x90_amp = qubit.pulse_params.amplitude.minus_x90
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

    short_minus_x90_len = qubit.pulse_params.short_pi_len * 1e9
    short_minus_x90_sigma = qubit.pulse_params.short_pi_len * 1e9 / 5
    short_minus_x90_amp = - qubit.pulse_params.short_pi_amp / 2
    short_minus_x90_wf, short_minus_x90_der_wf = np.array(
        drag_gaussian_pulse_waveforms(
            short_minus_x90_amp,
            short_minus_x90_len,
            short_minus_x90_sigma,
            drag_coef,
            anharmonicity,
            AC_stark_detuning,
        )
    )
    short_minus_x90_I_wf = short_minus_x90_wf
    short_minus_x90_Q_wf = short_minus_x90_der_wf

    y180_len = qubit.pulse_params.length * 1e9
    y180_sigma = qubit.pulse_params.length * 1e9 / 5
    y180_amp = qubit.pulse_params.amplitude.y180
    y180_wf, y180_der_wf = np.array(
        drag_gaussian_pulse_waveforms(y180_amp, y180_len, y180_sigma, drag_coef, anharmonicity, AC_stark_detuning)
    )
    y180_I_wf = (-1) * y180_der_wf
    y180_Q_wf = y180_wf
    # No DRAG when alpha=0, it's just a gaussian.

    y90_len = qubit.pulse_params.length * 1e9
    y90_sigma = qubit.pulse_params.length * 1e9 / 5
    y90_amp = qubit.pulse_params.amplitude.y90
    y90_wf, y90_der_wf = np.array(
        drag_gaussian_pulse_waveforms(y90_amp, y90_len, y90_sigma, drag_coef, anharmonicity, AC_stark_detuning)
    )
    y90_I_wf = (-1) * y90_der_wf
    y90_Q_wf = y90_wf
    # No DRAG when alpha=0, it's just a gaussian.

    minus_y90_len = qubit.pulse_params.length * 1e9
    minus_y90_sigma = qubit.pulse_params.length * 1e9 / 5
    minus_y90_amp = qubit.pulse_params.amplitude.minus_y90
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

    readout_len = resonator.readout_params.readout_length * 1e9
    readout_amp = resonator.readout_params.readout_amplitude
    # IQ Plane
    rotation_angle = (resonator.readout_params.rotation_angle / 180) * np.pi

    # CLEAR pulse
    CLEAR_wf = [2*readout_amp] * int(resonator.time_constant*1e9) + [readout_amp] * int(readout_len - 2*resonator.time_constant*1e9) + [-2*readout_amp] * int(resonator.time_constant*1e9)

    config = {
        "version": 1,
        "controllers": {
            "con1": {
                "analog_outputs": {
                    1: {"offset": 0.0},  # I resonator
                    2: {"offset": 0.0},  # Q resonator
                    3: {"offset": 0.0},  # I qubit
                    4: {"offset": 0.0},  # Q qubit
                    7: {"offset": 0.0},  # I storage
                    8: {"offset": 0.0},  # Q storage
                },
                "digital_outputs": {},
                "analog_inputs": {
                    1: {
                        "offset": resonator.wiring_output.I.offset,
                        "gain_db": resonator.wiring_output.gain,
                    },  # I from down-conversion
                    2: {
                        "offset": resonator.wiring_output.Q.offset,
                        "gain_db": resonator.wiring_output.gain,
                    },  # Q from down-conversion
                },
            },
        },
        "elements": {
            "qubit": {
                "mixInputs": {
                    "I": (qubit.wiring.I.con, qubit.wiring.I.ao),
                    "Q": (qubit.wiring.Q.con, qubit.wiring.Q.ao),
                    "lo_frequency": qubit.lo,
                    "mixer": "mixer_qubit",
                },
                "intermediate_frequency": qubit.f_01 - qubit.lo,
                "operations": {
                    "cw": "const_pulse",
                    "saturation": "saturation_pulse",
                    "gauss": "gaussian_pulse",
                    "pi": "x180_pulse",
                    "pi_half": "x90_pulse",
                    "x90": "x90_pulse",
                    "short_x90": "short_x90_pulse",
                    "x180": "x180_pulse",
                    "-x90": "-x90_pulse",
                    "short_-x90": "short_-x90_pulse",
                    "y90": "y90_pulse",
                    "y180": "y180_pulse",
                    "-y90": "-y90_pulse",
                },
            },
            "storage": {
                "mixInputs": {
                    "I": (storage.wiring.I.con, storage.wiring.I.ao),
                    "Q": (storage.wiring.Q.con, storage.wiring.Q.ao),
                    "lo_frequency": storage.lo,
                    "mixer": "mixer_storage",
                },
                "intermediate_frequency": storage.frequency - storage.lo,
                "operations": {
                    "cw": "const_pulse",
                    "displace": "displace_pulse",  # Q = zero_wf
                    "saturation": "saturation_storage_pulse",
                },
            },
            "resonator": {
                "mixInputs": {
                    "I": (resonator.wiring_input.I.con, resonator.wiring_input.I.ao),
                    "Q": (resonator.wiring_input.Q.con, resonator.wiring_input.Q.ao),
                    "lo_frequency": resonator.lo,
                    "mixer": "mixer_resonator",
                },
                "intermediate_frequency": resonator.frequency - resonator.lo,
                "operations": {
                    "cw": "const_pulse",
                    "readout": "readout_pulse",
                    "CLEAR": "CLEAR_pulse",
                },
                "outputs": {
                    "out1": (resonator.wiring_output.I.con, resonator.wiring_output.I.ai),
                    "out2": (resonator.wiring_output.Q.con, resonator.wiring_output.Q.ai),
                },
                "time_of_flight": resonator.tof,
                "smearing": 0,
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
            "square_pi_pulse": {
                "operation": "control",
                "length": square_pi_len,
                "waveforms": {
                    "I": "square_pi_wf",
                    "Q": "zero_wf",
                },
            },
            "saturation_pulse": {
                "operation": "control",
                "length": saturation_len,
                "waveforms": {"I": "saturation_drive_wf", "Q": "zero_wf"},
            },
            "saturation_storage_pulse": {
                "operation": "control",
                "length": storage_sat_len,
                "waveforms": {"I": "storage_sat_drive_wf", "Q": "zero_wf"},
            },
            "gaussian_pulse": {
                "operation": "control",
                "length": gauss_len,
                "waveforms": {
                    "I": "gauss_wf",
                    "Q": "zero_wf",
                },
            },
            "displace_pulse": {
                "operation": "control",
                "length": displace_len,
                "waveforms": {
                    "I": "displace_wf",
                    "Q": "zero_wf",
                },
            },
            "x90_pulse": {
                "operation": "control",
                "length": x90_len,
                "waveforms": {
                    "I": "x90_I_wf",
                    "Q": "x90_Q_wf",
                },
            },
            "short_x90_pulse": {
                "operation": "control",
                "length": short_x90_len,
                "waveforms": {
                    "I": "short_x90_I_wf",
                    "Q": "short_x90_Q_wf",
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
            "-x90_pulse": {
                "operation": "control",
                "length": minus_x90_len,
                "waveforms": {
                    "I": "minus_x90_I_wf",
                    "Q": "minus_x90_Q_wf",
                },
            },
            "short_-x90_pulse": {
                "operation": "control",
                "length": short_minus_x90_len,
                "waveforms": {
                    "I": "short_minus_x90_I_wf",
                    "Q": "short_minus_x90_Q_wf",
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
                },
                "digital_marker": "ON",
            },
            "CLEAR_pulse": {
                "operation": "measurement",
                "length": readout_len,
                "waveforms": {
                    "I": "CLEAR_wf",
                    "Q": "zero_wf",
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

        },
        "waveforms": {
            "const_wf": {"type": "constant", "sample": const_amp},
            "saturation_drive_wf": {"type": "constant", "sample": saturation_amp},
            "storage_sat_drive_wf": {"type": "constant", "sample": storage_sat_amp},
            "square_pi_wf": {"type": "constant", "sample": square_pi_amp},
            "displace_wf": {"type": "arbitrary", "samples": displace_wf.tolist()},
            "zero_wf": {"type": "constant", "sample": 0.0},
            "gauss_wf": {"type": "arbitrary", "samples": gauss_wf.tolist()},
            "x90_I_wf": {"type": "arbitrary", "samples": x90_I_wf.tolist()},
            "x90_Q_wf": {"type": "arbitrary", "samples": x90_Q_wf.tolist()},
            "short_x90_I_wf": {"type": "arbitrary", "samples": short_x90_I_wf.tolist()},
            "short_x90_Q_wf": {"type": "arbitrary", "samples": short_x90_Q_wf.tolist()},
            "x180_I_wf": {"type": "arbitrary", "samples": x180_I_wf.tolist()},
            "x180_Q_wf": {"type": "arbitrary", "samples": x180_Q_wf.tolist()},
            "minus_x90_I_wf": {"type": "arbitrary", "samples": minus_x90_I_wf.tolist()},
            "minus_x90_Q_wf": {"type": "arbitrary", "samples": minus_x90_Q_wf.tolist()},
            "short_minus_x90_I_wf": {"type": "arbitrary", "samples": short_minus_x90_I_wf.tolist()},
            "short_minus_x90_Q_wf": {"type": "arbitrary", "samples": short_minus_x90_Q_wf.tolist()},
            "y90_Q_wf": {"type": "arbitrary", "samples": y90_Q_wf.tolist()},
            "y90_I_wf": {"type": "arbitrary", "samples": y90_I_wf.tolist()},
            "y180_Q_wf": {"type": "arbitrary", "samples": y180_Q_wf.tolist()},
            "y180_I_wf": {"type": "arbitrary", "samples": y180_I_wf.tolist()},
            "minus_y90_Q_wf": {"type": "arbitrary", "samples": minus_y90_Q_wf.tolist()},
            "minus_y90_I_wf": {"type": "arbitrary", "samples": minus_y90_I_wf.tolist()},
            "readout_wf": {"type": "constant", "sample": readout_amp},
            "CLEAR_wf": {"type": "arbitrary", "samples": CLEAR_wf},
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
        },
        "mixers": {
            "mixer_qubit": [
                {
                    "intermediate_frequency": qubit.f_01 - qubit.lo,
                    "lo_frequency": qubit.lo,
                    "correction": [1.0, 0.0, 0.0, 1.0],
                }
            ],
            "mixer_resonator": [
                {
                    "intermediate_frequency": resonator.frequency - resonator.lo,
                    "lo_frequency": resonator.lo,
                    "correction": [1.0, 0.0, 0.0, 1.0],
                }
            ],
            "mixer_storage": [
                {
                    "intermediate_frequency": storage.frequency - storage.lo,
                    "lo_frequency": storage.lo,
                    "correction": [1.0, 0.0, 0.0, 1.0],
                }
            ],
        },
    }

    return config
