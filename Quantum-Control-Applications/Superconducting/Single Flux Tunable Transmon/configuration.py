import numpy as np
from scipy.signal.windows import gaussian
from qualang_tools.config.waveform_tools import drag_gaussian_pulse_waveforms
from qualang_tools.units import unit
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter, fetching_tool


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


#############
# VARIABLES #
#############
u = unit()

qop_ip = "127.0.0.1"

# Qubits
qubit_LO = 7.4 * u.GHz  # Used only for mixer correction and frequency rescaling for plots or computation
qubit_IF = 110 * u.MHz
mixer_qubit_g = 0.0
mixer_qubit_phi = 0.0

qubit_T1 = int(10 * u.us)

const_len = 100
const_amp = 50 * u.mV

pi_len = 100
pi_amp = 0.05

drag_coef = 0
anharmonicity = -200 * u.MHz
AC_stark_detuning = 0 * u.MHz

gauss_len = 200
gauss_sigma = gauss_len / 5
gauss_amp = 0.25
gauss_wf = gauss_amp * gaussian(gauss_len, gauss_sigma)

x180_len = 40
x180_sigma = x180_len / 5
x180_amp = 0.35
x180_wf, x180_der_wf = np.array(
    drag_gaussian_pulse_waveforms(x180_amp, x180_len, x180_sigma, drag_coef, anharmonicity, AC_stark_detuning)
)
x180_I_wf = x180_wf
x180_Q_wf = x180_der_wf
# No DRAG when alpha=0, it's just a gaussian.

x90_len = x180_len
x90_sigma = x90_len / 5
x90_amp = x180_amp / 2
x90_wf, x90_der_wf = np.array(
    drag_gaussian_pulse_waveforms(x90_amp, x90_len, x90_sigma, drag_coef, anharmonicity, AC_stark_detuning)
)
x90_I_wf = x90_wf
x90_Q_wf = x90_der_wf
# No DRAG when alpha=0, it's just a gaussian.

minus_x90_len = x180_len
minus_x90_sigma = minus_x90_len / 5
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

# Resonator
resonator_LO = 4.8 * u.GHz  # Used only for mixer correction and frequency rescaling for plots or computation
resonator_IF = 60 * u.MHz
mixer_resonator_g = 0.0
mixer_resonator_phi = 0.0

readout_len = 20
readout_amp = 0.25

time_of_flight = 300

# Flux line
const_flux_len = 200
const_flux_amp = 0.45

# IQ Plane Angle
rotation_angle = (0 / 180) * np.pi
# Threshold for single shot g-e discrimination
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
                5: {"offset": 0.0},  # flux line
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
                "gauss": "gaussian_pulse",
                "pi": "pi_pulse",
                "pi_half": "pi_half_pulse",
                "x180": "x180_pulse",
                "x90": "x90_pulse",
                "-x90": "-x90_pulse",
                "y90": "y90_pulse",
                "y180": "y180_pulse",
                "-y90": "-y90_pulse",
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
                "readout": "readout_pulse",
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
                "port": ("con1", 5),
            },
            "operations": {
                "const": "const_flux_pulse",
            },
        },
        "flux_line_sticky": {
            "singleInput": {
                "port": ("con1", 5),
            },
            "hold_offset": {"duration": 1},  # in clock cycles (4ns)
            "operations": {
                "const": "const_flux_pulse",
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
            "length": pi_len,
            "waveforms": {
                "I": "pi_wf",
                "Q": "zero_wf",
            },
        },
        "pi_half_pulse": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {
                "I": "pi_half_wf",
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
        "gaussian_pulse": {
            "operation": "control",
            "length": gauss_len,
            "waveforms": {
                "I": "gauss_wf",
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
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": const_amp},
        "pi_wf": {"type": "constant", "sample": pi_amp},
        "pi_half_wf": {"type": "constant", "sample": pi_amp / 2},
        "const_flux_wf": {"type": "constant", "sample": const_flux_amp},
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
        "minus_y90_wf": {"type": "arbitrary", "samples": minus_y90_wf.tolist()},
        "minus_y90_der_wf": {"type": "arbitrary", "samples": minus_y90_der_wf.tolist()},
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
