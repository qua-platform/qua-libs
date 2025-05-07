import numpy as np
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter, fetching_tool

# from qualang_tools.config.waveform_tools import drag_gaussian_pulse_waveforms, drag_cosine_pulse_waveforms


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


def drag_cosine_pulse_waveforms(amplitude, length, alpha, anharmonicity, detuning=0, **kwargs):
    """
    Creates Cosine based DRAG waveforms that compensate for the leakage and for the AC stark shift.

    These DRAG waveforms has been implemented following the next Refs.:
    Chen et al. PRL, 116, 020501 (2016)
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.020501
    and Chen's thesis
    https://web.physics.ucsb.edu/~martinisgroup/theses/Chen2018.pdf

    :param float amplitude: The amplitude in volts.
    :param int length: The pulse length in ns.
    :param float alpha: The DRAG coefficient.
    :param float anharmonicity: f_21 - f_10 - The differences in energy between the 2-1 and the 1-0 energy levels, in Hz.
    :param float detuning: The frequency shift to correct for AC stark shift, in Hz.
    :return: Returns a tuple of two lists. The first list is the I waveform (real part) and the second is the
        Q waveform (imaginary part)
    """
    delta = kwargs.get("delta", None)
    if delta is not None:
        anharmonicity = delta
        print("'delta' has been replaced by 'anharmonicity' and will be deprecated in the future.")
    if alpha != 0 and anharmonicity == 0:
        raise Exception("Cannot create a DRAG pulse with `anharmonicity=0`")
    t = np.arange(length, dtype=int)  # An array of size pulse length in ns
    end_point = length - 1
    cos_wave = 0.5 * amplitude * (1 - np.cos(t * 2 * np.pi / end_point))  # The cosine function
    sin_wave = (
        0.5 * amplitude * (2 * np.pi / end_point * 1e9) * np.sin(t * 2 * np.pi / end_point)
    )  # The derivative of cosine function
    z = cos_wave + 1j * 0
    if alpha != 0:
        # The complex DRAG envelope:
        z += 1j * sin_wave * (alpha / (anharmonicity - 2 * np.pi * detuning))
        # The complex detuned DRAG envelope:
        z *= np.exp(1j * 2 * np.pi * detuning * t * 1e-9)
    I_wf = z.real.tolist()  # The `I` component is the real part of the waveform
    Q_wf = z.imag.tolist()  # The `Q` component is the imaginary part of the waveform
    return I_wf, Q_wf


#############
# VARIABLES #
#############

qop_ip = "127.0.0.1"

###########################################
# Qubits parameters
###########################################
qubit_T1 = int(20e3)
# Qubit 0
qubit_IF = -93.27227789896837e6 - 0.01343020580706522e6
qubit_LO = 7e9
mixer_qubit_g = -0.006
mixer_qubit_phi = -0.013
drag_coef = 0.11976190787994763
det = 0
# Qubit 1 (same LO and mixer parameters as qubit 0)
qubit1_IF = -265.2888535547536e6 + 0.8339050229069631e6 - 0.025038658196256947e6
drag_coef1 = 0
det1 = 0

############################ qubit 0 waveforms
x180_len = 16
x180_sigma = x180_len / 5
x180_amp = 0.098  # *2 due to half time
x180_wf, x180_der_wf = np.array(
    drag_cosine_pulse_waveforms(x180_amp, x180_len, alpha=drag_coef, anharmonicity=(-2 * np.pi * 0.163e9), detuning=det)
)
# No DRAG when alpha=0, it's just a gaussian.
x90_len = x180_len
x90_sigma = x90_len / 5
x90_amp = 0.098 / 2
x90_wf, x90_der_wf = np.array(
    drag_cosine_pulse_waveforms(x90_amp, x90_len, alpha=drag_coef, anharmonicity=(-2 * np.pi * 0.163e9), detuning=det)
)
# No DRAG when alpha=0, it's just a gaussian.
y90_len = x180_len
y90_sigma = y90_len / 5
y90_amp = 0.098 / 2
y90_wf, y90_der_wf = np.array(
    drag_cosine_pulse_waveforms(y90_amp, y90_len, alpha=drag_coef, anharmonicity=(-2 * np.pi * 0.163e9), detuning=det)
)
y90_der_wf = (-1) * y90_der_wf
# No DRAG when alpha=0, it's just a gaussian.

######################################### qubit 1 waveforms
x180_len1 = 100
x180_amp1 = 0.225  # *2 due to half time
x180_wf1, x180_der_wf1 = np.array(
    drag_cosine_pulse_waveforms(
        x180_amp1, x180_len1, alpha=drag_coef1, anharmonicity=(-2 * np.pi * 0.163e9), detuning=det1
    )
)
# No DRAG when alpha=0, it's just a gaussian.
x90_len1 = x180_len1
x90_amp1 = 0.225 / 2
x90_wf1, x90_der_wf1 = np.array(
    drag_cosine_pulse_waveforms(
        x90_amp1, x90_len1, alpha=drag_coef1, anharmonicity=(-2 * np.pi * 0.163e9), detuning=det1
    )
)
# No DRAG when alpha=0, it's just a gaussian.
y90_len1 = x180_len1
y90_amp1 = 0.225 / 2
y90_wf1, y90_der_wf1 = np.array(
    drag_cosine_pulse_waveforms(
        y90_amp1, y90_len1, alpha=drag_coef1, anharmonicity=(-2 * np.pi * 0.163e9), detuning=det1
    )
)
y90_der_wf1 = (-1) * y90_der_wf1
# No DRAG when alpha=0, it's just a gaussian.

###########################################
# Measurement parameters
###########################################
# Resonator 0
resonator_IF = -145e6
resonator_LO = 5.5e9
mixer_resonator_g = 0.0
mixer_resonator_phi = 0.0
# Resonator 1 (same LO and mixer parameters as resonator 0)
resonator1_IF = 76.1e6
# Measurement operations
readout_len = 760
readout_amp = 0.06 * 0.7
readout_len1 = 660
readout_amp1 = 0.04902 * 0.9
smearing = 0
time_of_flight = 180 + 20 * 4

###########################################
# IQ plane parameters
###########################################
# IQ Plane 0
# 2 filters
# rotation_angle = ((-100.5-153.7-152.9-205.4) / 180) * np.pi
# ge_threshold = -1.890e-05
# 1 filter
rotation_angle = ((-100.5) / 180) * np.pi
ge_threshold = -3.582e-05
# no filter
# rotation_angle = ((-100.5 - 136.5) / 180) * np.pi
# ge_threshold = -2.911e-05

# IQ Plane 1
# 2 filters
# rotation_angle1 = ((-100.5 - 107.8 - 62.6) / 180) * np.pi
# ge_threshold1 = 1.323e-04
# 1 filter
rotation_angle1 = ((-61.5 - 100.2 - 1.4) / 180) * np.pi
ge_threshold1 = 1.290e-04

###########################################
# Flux line parameters
###########################################
# Flux line 0
offset7 = -0.27524204128326885 + 0.005
flux_line_IF = 0e6
const_flux_len = 260
const_flux_amp = 0.15
# Filter taps
fir0 = [1.05838468, -0.99737684]
iir0 = [0.93899217]
# fir0 = []
# iir0 = []

# Flux line 1 (same IF as flux line 0)
offset8 = -0.23492508406984103
const_flux_len1 = 260
const_flux_amp1 = 0.15
# Filter taps
fir1 = [1.0684318, -1.01981586]
iir1 = [0.95138406]
# fir1 = []
# iir1 = []

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": -0.0102 * 1},  # q0 I
                2: {"offset": 0.0316 * 1},  # q0 Q
                3: {"offset": 0.0},  # q1 I
                4: {"offset": 0.0},  # q1 Q
                5: {"offset": 0.0},  # resonators I
                6: {"offset": 0.0},  # resonators Q
                7: {"offset": offset7, "filter": {"feedforward": fir0, "feedback": iir0}},  # qo flux line
                8: {"offset": offset8, "filter": {"feedforward": fir1, "feedback": iir1}},  # q1 flux line
            },
            "digital_outputs": {},
            "analog_inputs": {
                1: {"offset": 0.0, "gain_db": 0},  # I from down-conversion
                2: {"offset": 0.0, "gain_db": 0},  # Q from down-conversion
            },
        },
    },
    "elements": {
        "qubit0": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": qubit_LO,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": qubit_IF,
            "operations": {
                "x90": "x90_pulse",
                "x180": "x180_pulse",
                "y90": "y90_pulse",
            },
        },
        "qubit1": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": qubit_LO,
                "mixer": "mixer_qubit1",
            },
            "intermediate_frequency": qubit1_IF,
            "operations": {
                "x90": "x90_pulse1",
                "x180": "x180_pulse1",
                "y90": "y90_pulse1",
            },
        },
        "resonator0": {
            "mixInputs": {
                "I": ("con1", 5),
                "Q": ("con1", 6),
                "lo_frequency": resonator_LO,
                "mixer": "mixer_resonator",
            },
            "intermediate_frequency": resonator_IF,
            "operations": {
                "readout": "readout_pulse",
            },
            "outputs": {
                "out1": ("con1", 1),
                "out2": ("con1", 2),
            },
            "time_of_flight": time_of_flight,
            "smearing": smearing,
        },
        "resonator1": {
            "mixInputs": {
                "I": ("con1", 5),
                "Q": ("con1", 6),
                "lo_frequency": resonator_LO,
                "mixer": "mixer_resonator1",
            },
            "intermediate_frequency": resonator1_IF,
            "operations": {
                "readout": "readout_pulse1",
            },
            "outputs": {
                "out1": ("con1", 1),
                "out2": ("con1", 2),
            },
            "time_of_flight": time_of_flight,
            "smearing": smearing,
        },
        "flux_line0": {
            "singleInput": {
                "port": ("con1", 7),
            },
            "intermediate_frequency": flux_line_IF,
            "operations": {
                "cw": "const_flux_pulse",
            },
        },
        "flux_line1": {
            "singleInput": {
                "port": ("con1", 8),
            },
            "intermediate_frequency": flux_line_IF,
            "operations": {
                "cw": "const_flux_pulse1",
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
        "const_flux_pulse1": {
            "operation": "control",
            "length": const_flux_len1,
            "waveforms": {
                "single": "const_flux_wf1",
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
        "y90_pulse": {
            "operation": "control",
            "length": y90_len,
            "waveforms": {
                "I": "y90_der_wf",
                "Q": "y90_wf",
            },
        },
        "x90_pulse1": {
            "operation": "control",
            "length": x90_len1,
            "waveforms": {
                "I": "x90_wf1",
                "Q": "x90_der_wf1",
            },
        },
        "x180_pulse1": {
            "operation": "control",
            "length": x180_len1,
            "waveforms": {
                "I": "x180_wf1",
                "Q": "x180_der_wf1",
            },
        },
        "y90_pulse1": {
            "operation": "control",
            "length": y90_len1,
            "waveforms": {
                "I": "y90_der_wf1",
                "Q": "y90_wf1",
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
                "rotated_cos": "rotated_cosine_weights",
                "rotated_sin": "rotated_sine_weights",
                "rotated_minus_sin": "rotated_minus_sine_weights",
            },
            "digital_marker": "ON",
        },
        "readout_pulse1": {
            "operation": "measurement",
            "length": readout_len1,
            "waveforms": {
                "I": "readout_wf1",
                "Q": "zero_wf",
            },
            "integration_weights": {
                "rotated_cos": "rotated_cosine_weights1",
                "rotated_sin": "rotated_sine_weights1",
                "rotated_minus_sin": "rotated_minus_sine_weights1",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "zero_wf": {"type": "constant", "sample": 0.0},
        "x90_wf": {"type": "arbitrary", "samples": x90_wf.tolist()},
        "x90_der_wf": {"type": "arbitrary", "samples": x90_der_wf.tolist()},
        "x180_wf": {"type": "arbitrary", "samples": x180_wf.tolist()},
        "x180_der_wf": {"type": "arbitrary", "samples": x180_der_wf.tolist()},
        "y90_wf": {"type": "arbitrary", "samples": y90_wf.tolist()},
        "y90_der_wf": {"type": "arbitrary", "samples": y90_der_wf.tolist()},
        "x90_wf1": {"type": "arbitrary", "samples": x90_wf1.tolist()},
        "x90_der_wf1": {"type": "arbitrary", "samples": x90_der_wf1.tolist()},
        "x180_wf1": {"type": "arbitrary", "samples": x180_wf1.tolist()},
        "x180_der_wf1": {"type": "arbitrary", "samples": x180_der_wf1.tolist()},
        "y90_wf1": {"type": "arbitrary", "samples": y90_wf1.tolist()},
        "y90_der_wf1": {"type": "arbitrary", "samples": y90_der_wf1.tolist()},
        "readout_wf": {"type": "constant", "sample": readout_amp},
        "readout_wf1": {"type": "constant", "sample": readout_amp1},
        "const_flux_wf": {"type": "constant", "sample": const_flux_amp},
        "const_flux_wf1": {"type": "constant", "sample": const_flux_amp1},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
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
        "rotated_cosine_weights1": {
            "cosine": [(np.cos(rotation_angle1), readout_len1)],
            "sine": [(-np.sin(rotation_angle1), readout_len1)],
        },
        "rotated_sine_weights1": {
            "cosine": [(np.sin(rotation_angle1), readout_len1)],
            "sine": [(np.cos(rotation_angle1), readout_len1)],
        },
        "rotated_minus_sine_weights1": {
            "cosine": [(-np.sin(rotation_angle1), readout_len1)],
            "sine": [(-np.cos(rotation_angle1), readout_len1)],
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
        "mixer_qubit1": [
            {
                "intermediate_frequency": qubit1_IF,
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
        "mixer_resonator1": [
            {
                "intermediate_frequency": resonator1_IF,
                "lo_frequency": resonator_LO,
                "correction": IQ_imbalance(mixer_resonator_g, mixer_resonator_phi),
            }
        ],
    },
}
