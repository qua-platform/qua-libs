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

    :param g: relative gain imbalance between the I & Q ports. (unit-less), set to 0 for no gain imbalance.
    :param phi: relative phase imbalance between the I & Q ports (radians), set to 0 for no phase imbalance.
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
        dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I),
        dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q),
    )
    if threshold is not None:
        assign(state, I > threshold)
    return state, I, Q


#############
# VARIABLES #
#############

qop_ip = "127.0.0.1"
qop_port = 80

# Qubits
qubit_IF = 50e6
qubit_LO = 7e9
mixer_qubit_g = 0.0
mixer_qubit_phi = 0.0

qubit_T1 = int(10e3)

saturation_len = 1000
saturation_amp = 0.1
const_len = 100
const_amp = 0.1
square_pi_len = 100
square_pi_amp = 0.1

gauss_len = 20
gauss_sigma = gauss_len / 5
gauss_amp = 0.35
gauss_wf = gauss_amp * gaussian(gauss_len, gauss_sigma)

x180_len = 40
x180_sigma = x180_len / 5
x180_amp = 0.35
x180_wf, x180_der_wf = np.array(drag_gaussian_pulse_waveforms(x180_amp, x180_len, x180_sigma, alpha=0, delta=1))
# No DRAG when alpha=0, it's just a gaussian.

x90_len = x180_len
x90_sigma = x90_len / 5
x90_amp = x180_amp / 2
x90_wf, x90_der_wf = np.array(drag_gaussian_pulse_waveforms(x90_amp, x90_len, x90_sigma, alpha=0, delta=1))
# No DRAG when alpha=0, it's just a gaussian.

x270_len = x180_len
x270_sigma = x270_len / 5
x270_amp = -x90_amp
x270_wf, x270_der_wf = np.array(drag_gaussian_pulse_waveforms(x270_amp, x270_len, x270_sigma, alpha=0, delta=1))
# No DRAG when alpha=0, it's just a gaussian.

y180_len = x180_len
y180_sigma = y180_len / 5
y180_amp = x180_amp
y180_wf, y180_der_wf = np.array(drag_gaussian_pulse_waveforms(y180_amp, y180_len, y180_sigma, alpha=0, delta=1))
# No DRAG when alpha=0, it's just a gaussian.

y90_len = x180_len
y90_sigma = y90_len / 5
y90_amp = y180_amp / 2
y90_wf, y90_der_wf = np.array(drag_gaussian_pulse_waveforms(y90_amp, y90_len, y90_sigma, alpha=0, delta=1))
# No DRAG when alpha=0, it's just a gaussian.

y270_len = x180_len
y270_sigma = y270_len / 5
y270_amp = -y90_amp
y270_wf, y270_der_wf = np.array(drag_gaussian_pulse_waveforms(y270_amp, y270_len, y270_sigma, alpha=0, delta=1))
# No DRAG when alpha=0, it's just a gaussian.

# Resonator
resonator_IF = 60e6
resonator_LO = 5.5e9
mixer_resonator_g = 0.0
mixer_resonator_phi = 0.0

time_of_flight = 180

short_readout_len = 500
short_readout_amp = 0.4
readout_len = 5000
readout_amp = 0.2
long_readout_len = 50000
long_readout_amp = 0.1

# IQ Plane
rotation_angle = (0.0 / 180) * np.pi
ge_threshold = 0.0


config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": 0.0},  # I qubit
                2: {"offset": 0.0},  # Q qubit
                3: {"offset": 0.0},  # I resonator
                4: {"offset": 0.0},  # Q resonator
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
                "x270": "x270_pulse",
                "y90": "y90_pulse",
                "y180": "y180_pulse",
                "y270": "y270_pulse",
            },
        },
        "resonator": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
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
        "x270_pulse": {
            "operation": "control",
            "length": x270_len,
            "waveforms": {
                "I": "x270_wf",
                "Q": "x270_der_wf",
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
        "y270_pulse": {
            "operation": "control",
            "length": y270_len,
            "waveforms": {
                "I": "y270_der_wf",
                "Q": "y270_wf",
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
        "x270_wf": {"type": "arbitrary", "samples": x270_wf.tolist()},
        "x270_der_wf": {"type": "arbitrary", "samples": x270_der_wf.tolist()},
        "y90_wf": {"type": "arbitrary", "samples": y90_wf.tolist()},
        "y90_der_wf": {"type": "arbitrary", "samples": y90_der_wf.tolist()},
        "y180_wf": {"type": "arbitrary", "samples": y180_wf.tolist()},
        "y180_der_wf": {"type": "arbitrary", "samples": y180_der_wf.tolist()},
        "y270_wf": {"type": "arbitrary", "samples": y270_wf.tolist()},
        "y270_der_wf": {"type": "arbitrary", "samples": y270_der_wf.tolist()},
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
        "mixer_resonator": [
            {
                "intermediate_frequency": resonator_IF,
                "lo_frequency": resonator_LO,
                "correction": IQ_imbalance(mixer_resonator_g, mixer_resonator_phi),
            }
        ],
    },
}
