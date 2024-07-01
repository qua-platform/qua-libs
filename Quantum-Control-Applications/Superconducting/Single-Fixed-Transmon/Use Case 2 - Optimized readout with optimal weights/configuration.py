import numpy as np
from qm.qua import declare, fixed, measure, dual_demod, assign
from scipy.signal.windows import gaussian
from qualang_tools.config.waveform_tools import drag_gaussian_pulse_waveforms


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


# Readout macro
def readout_macro(threshold=None, state=None, I=None, Q=None):
    """
    A macro for performing the readout, with the ability to perform state discrimination.
    If `threshold` is given, the information in the `I` quadrature will be compared against the threshold and `state`
    would be `True` if `I > threshold`.
    Note that it is assumed that the results are rotated such that all the information is in the `I` quadrature.

    :param threshold: Optional. The threshold to compare `I` against.
    :param state: A QUA variable for the state information, only used when a threshold is given.
        Should be of type `bool`. If not given, a new variable will be created
    :param I: A QUA variable for the information in the `I` quadrature. Should be of type `Fixed`. If not given, a new
        variable will be created
    :param Q: A QUA variable for the information in the `Q` quadrature. Should be of type `Fixed`. If not given, a new
        variable will be created
    :return: Three QUA variables populated with the results of the readout: (`state`, `I`, `Q`)
    """
    if I is None:
        I = declare(fixed)
    if Q is None:
        Q = declare(fixed)
    if threshold is not None and state is None:
        state = declare(bool)
    measure(
        "readout",
        "resonator",
        None,
        dual_demod.full("rotated_cos", "rotated_sin", I),
        dual_demod.full("rotated_minus_sin", "rotated_cos", Q),
    )
    if threshold is not None:
        assign(state, I > threshold)
    return state, I, Q


#############
# VARIABLES #
#############

qop_ip = "127.0.0.1"

# Qubits
qubit_IF = -95.82483696051314e6 + 0.15e6
qubit_ef_IF = -259.1453349274868e6
# anaharmonicity 12-01 = -163.320497966974e6
qubit_LO = 7e9
qubit_ef_LO = 7e9
mixer_qubit_g = 0.0
mixer_qubit_phi = 0.0

qubit_T1 = int(20e3)

saturation_len = 100000
saturation_amp = 0.005
const_len = 1000
const_amp = 0.1
square_pi_len = 100
square_pi_amp = 0.1

# drag_coef = -0.005
# det = -53e6
drag_coef = 0
det = 0

gauss_len = 16
gauss_sigma = gauss_len / 5
gauss_amp = 0.35
gauss_wf = gauss_amp * gaussian(gauss_len, gauss_sigma)

x180_len = 32
x180_sigma = x180_len / 5
x180_amp = 0.046 * 1.105 * 1.035  # *2 due to half time
x180_wf, x180_der_wf = np.array(
    drag_gaussian_pulse_waveforms(x180_amp, x180_len, x180_sigma, alpha=drag_coef, delta=-0.163e9, detuning=det)
)
x180_der_wf = (-1) * x180_der_wf
# No DRAG when alpha=0, it's just a gaussian.

x90_len = x180_len
x90_sigma = x90_len / 5
x90_amp = 0.027 * 0.97 * 1.004  # x180_amp / 2
x90_wf, x90_der_wf = np.array(
    drag_gaussian_pulse_waveforms(x90_amp, x90_len, x90_sigma, alpha=drag_coef, delta=-0.163e9, detuning=det)
)
x90_der_wf = (-1) * x90_der_wf
# No DRAG when alpha=0, it's just a gaussian.

minus_x90_len = x180_len
minus_x90_sigma = minus_x90_len / 5
minus_x90_amp = -x90_amp
minus_x90_wf, minus_x90_der_wf = np.array(
    drag_gaussian_pulse_waveforms(
        minus_x90_amp, minus_x90_len, minus_x90_sigma, alpha=drag_coef, delta=-0.163e9, detuning=det
    )
)
minus_x90_der_wf = (-1) * minus_x90_der_wf
# No DRAG when alpha=0, it's just a gaussian.

y180_len = x180_len
y180_sigma = y180_len / 5
y180_amp = x180_amp
y180_wf, y180_der_wf = np.array(
    drag_gaussian_pulse_waveforms(y180_amp, y180_len, y180_sigma, alpha=drag_coef, delta=-0.163e9, detuning=det)
)
y180_der_wf = (1) * y180_der_wf  # for the correct sign in config
# No DRAG when alpha=0, it's just a gaussian.

y90_len = x180_len
y90_sigma = y90_len / 5
y90_amp = 0.027 * 0.97 * 1.004  # *2 due to 16 ns pulse
y90_wf, y90_der_wf = np.array(
    drag_gaussian_pulse_waveforms(y90_amp, y90_len, y90_sigma, alpha=drag_coef, delta=-0.163e9, detuning=det)
)
y90_der_wf = (1) * y90_der_wf
# No DRAG when alpha=0, it's just a gaussian.

minus_y90_len = y180_len
minus_y90_sigma = minus_y90_len / 5
minus_y90_amp = -y90_amp
minus_y90_wf, minus_y90_der_wf = np.array(
    drag_gaussian_pulse_waveforms(
        minus_y90_amp, minus_y90_len, minus_y90_sigma, alpha=drag_coef, delta=-0.163e9, detuning=det
    )
)
minus_y90_der_wf = (1) * minus_y90_der_wf
# No DRAG when alpha=0, it's just a gaussian.

# Resonator
resonator_IF = -145.45e6
resonator_LO = 5.5e9
mixer_resonator_g = 0.0
mixer_resonator_phi = 0.0

smearing = 0
time_of_flight = 180 + 20 * 4

short_readout_len = 500
short_readout_amp = 0.4
readout_len = 380  # 760
readout_amp = 0.06 * 1.5 * 1.1
long_readout_len = 50000
long_readout_amp = 0.004

# IQ Plane
rotation_angle = (-153.9 / 180) * np.pi
ge_threshold = -3.170e-04


config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": 0.0},  # q0 I
                2: {"offset": 0.0},  # q0 Q
                3: {"offset": 0.0},  #
                4: {"offset": 0.0},  #
                5: {"offset": 0.0},  # resonator
                6: {"offset": 0.0},  # resonator
                7: {"offset": -0.27524204128326885},  # qo flux
                8: {"offset": 0.0},  #
                9: {"offset": 0.0},  #
                10: {"offset": 0.0},  #
            },
            "digital_outputs": {},
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
                "pi": "x180_pulse",
                "pi_half": "x90_pulse",
                "x90": "x90_pulse",
                "x180": "x180_pulse",
                "-x90": "-x90_pulse",
                "y90": "y90_pulse",
                "y180": "y180_pulse",
                "-y90": "-y90_pulse",
            },
        },
        "qubit_ef": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": qubit_ef_LO,
                "mixer": "mixer_qubit_ef",
            },
            "intermediate_frequency": qubit_ef_IF,
            "operations": {
                "cw": "const_pulse",
                "saturation": "saturation_pulse",
            },
        },
        "resonator": {
            "mixInputs": {
                "I": ("con1", 5),
                "Q": ("con1", 6),
                "lo_frequency": resonator_LO,
                "mixer": "mixer_resonator",
            },
            "intermediate_frequency": resonator_IF,
            "operations": {
                "cw": "const_pulse",
                "short_readout": "short_readout_pulse",
                "readout": "readout_pulse",
                "long_readout": "long_readout_pulse",
            },
            "outputs": {
                "out1": ("con1", 1),
                "out2": ("con1", 2),
            },
            "time_of_flight": time_of_flight,
            "smearing": smearing,
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
        "gaussian_pulse": {
            "operation": "control",
            "length": gauss_len,
            "waveforms": {
                "I": "gauss_wf",
                "Q": "zero_wf",
            },
        },
        "x90_pulse": {
            "operation": "control",
            "length": x90_len,
            "waveforms": {
                "I": "x90_wf",
                "Q": "x90_der_wf",
            },
        },
        "x180_pulse": {
            "operation": "control",
            "length": x180_len,
            "waveforms": {
                "I": "x180_wf",
                "Q": "x180_der_wf",
            },
        },
        "-x90_pulse": {
            "operation": "control",
            "length": minus_x90_len,
            "waveforms": {
                "I": "minus_x90_wf",
                "Q": "minus_x90_der_wf",
            },
        },
        "y90_pulse": {
            "operation": "control",
            "length": y90_len,
            "waveforms": {
                "I": "y90_der_wf",
                "Q": "y90_wf",
            },
        },
        "y180_pulse": {
            "operation": "control",
            "length": y180_len,
            "waveforms": {
                "I": "y180_der_wf",
                "Q": "y180_wf",
            },
        },
        "-y90_pulse": {
            "operation": "control",
            "length": minus_y90_len,
            "waveforms": {
                "I": "minus_y90_der_wf",
                "Q": "minus_y90_wf",
            },
        },
        "short_readout_pulse": {
            "operation": "measurement",
            "length": short_readout_len,
            "waveforms": {
                "I": "short_readout_wf",
                "Q": "zero_wf",
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
        "long_readout_pulse": {
            "operation": "measurement",
            "length": long_readout_len,
            "waveforms": {
                "I": "long_readout_wf",
                "Q": "zero_wf",
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
        "saturation_drive_wf": {"type": "constant", "sample": saturation_amp},
        "square_pi_wf": {"type": "constant", "sample": square_pi_amp},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "gauss_wf": {"type": "arbitrary", "samples": gauss_wf.tolist()},
        "x90_wf": {"type": "arbitrary", "samples": x90_wf.tolist()},
        "x90_der_wf": {"type": "arbitrary", "samples": x90_der_wf.tolist()},
        "x180_wf": {"type": "arbitrary", "samples": x180_wf.tolist()},
        "x180_der_wf": {"type": "arbitrary", "samples": x180_der_wf.tolist()},
        "minus_x90_wf": {"type": "arbitrary", "samples": minus_x90_wf.tolist()},
        "minus_x90_der_wf": {"type": "arbitrary", "samples": minus_x90_der_wf.tolist()},
        "y90_wf": {"type": "arbitrary", "samples": y90_wf.tolist()},
        "y90_der_wf": {"type": "arbitrary", "samples": y90_der_wf.tolist()},
        "y180_wf": {"type": "arbitrary", "samples": y180_wf.tolist()},
        "y180_der_wf": {"type": "arbitrary", "samples": y180_der_wf.tolist()},
        "minus_y90_wf": {"type": "arbitrary", "samples": minus_x90_wf.tolist()},
        "minus_y90_der_wf": {"type": "arbitrary", "samples": minus_y90_der_wf.tolist()},
        "short_readout_wf": {"type": "constant", "sample": short_readout_amp},
        "readout_wf": {"type": "constant", "sample": readout_amp},
        "long_readout_wf": {"type": "constant", "sample": long_readout_amp},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "short_cosine_weights": {
            "cosine": [(1.0, short_readout_len)],
            "sine": [(0.0, short_readout_len)],
        },
        "short_sine_weights": {
            "cosine": [(0.0, short_readout_len)],
            "sine": [(1.0, short_readout_len)],
        },
        "short_minus_sine_weights": {
            "cosine": [(0.0, short_readout_len)],
            "sine": [(-1.0, short_readout_len)],
        },
        "short_rotated_cosine_weights": {
            "cosine": [(np.cos(rotation_angle), short_readout_len)],
            "sine": [(-np.sin(rotation_angle), short_readout_len)],
        },
        "short_rotated_sine_weights": {
            "cosine": [(np.sin(rotation_angle), short_readout_len)],
            "sine": [(np.cos(rotation_angle), short_readout_len)],
        },
        "short_rotated_minus_sine_weights": {
            "cosine": [(-np.sin(rotation_angle), short_readout_len)],
            "sine": [(-np.cos(rotation_angle), short_readout_len)],
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
        "mixer_qubit_ef": [
            {
                "intermediate_frequency": qubit_ef_IF,
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
