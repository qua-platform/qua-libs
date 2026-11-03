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


def flat_top_gaussian(amp, pulse_length, flat_length):
    gauss_length = pulse_length - flat_length
    gauss = (amp * gaussian(gauss_length, gauss_length/5)).tolist()
    return gauss[:gauss_length//2] + [amp] * flat_length + gauss[gauss_length//2:]


def gaussian_rise_const(amp, pulse_length, flat_length):
    gauss_length = int((pulse_length - flat_length)*2)
    gauss = (amp * gaussian(gauss_length, gauss_length/5)).tolist()
    return gauss[:gauss_length//2] + [amp] * flat_length


def gaussian_fall_const(amp, pulse_length, flat_length):
    gauss_length = int((pulse_length - flat_length)*2)
    gauss = (amp * gaussian(gauss_length, gauss_length/5)).tolist()
    return [amp] * flat_length + gauss[gauss_length//2:]


def gaussian_rise(amp, pulse_length, sigma):
    gauss_length = int((pulse_length)*2)
    gauss = (amp * gaussian(gauss_length, sigma*2)).tolist()
    return gauss[:gauss_length//2]


def gaussian_fall(amp, pulse_length, sigma):
    gauss_length = int((pulse_length)*2)
    gauss = (amp * gaussian(gauss_length, sigma*2)).tolist()
    return gauss[gauss_length//2:]

def ring_up_tanh(length):
    ts = np.linspace(-2, 2, length)
    return (1 + np.tanh(ts*2))/2

def ring_up_cos(length):
    return 0.5*(1 - np.cos(np.linspace(0, np.pi, length)))

def ring_up_wave(length, reverse=False, type='cos'):
    if type == 'cos':
        wave = ring_up_cos(length)
    elif type == 'tanh':
        wave = ring_up_tanh(length)
    else:
        raise ValueError("Type must be 'cos' or 'tanh', not %s" %type)
    if reverse:
        wave = wave[::-1]
    return wave

def smoothed_constant_wave(length, sigma, type='cos'):
    return np.concatenate([ring_up_wave(sigma, type=type), np.ones(length - 2*sigma), ring_up_wave(sigma, reverse=True, type=type)])

#############
# VARIABLES #
#############

qop_ip = "192.168.88.10"
qop_port = "80"

# Qubit
qubit_IF = -40e6
qubit_LO = 7e9
mixer_qubit_g = -0.055
mixer_qubit_phi = -0.1

qubit_T1 = int(20e3)

cat_pump = 0.05

rise_len = 500
fall_len = 500
rise_amp = 1e-3


rise_wf = rise_amp*ring_up_tanh(rise_len)
fall_wf = rise_amp*ring_up_wave(rise_len, reverse=True, type='tanh')

saturation_len = 10000
saturation_amp = 0.04  #  0.4V it gives like -8 dBm
const_len = 1000
const_amp = cat_pump
square_pi_len = 100
square_pi_amp = 0.1

gauss_len = 16
gauss_sigma = gauss_len / 5
gauss_amp = 1e-3
gauss_wf = gauss_amp * gaussian(gauss_len, gauss_sigma)

pi_len = 20
pi_sigma = pi_len / 5
pi_amp = 0.35
pi_wf = pi_amp * gaussian(pi_len, pi_sigma)

pi_half_len = 20
pi_half_sigma = pi_half_len / 5
pi_half_amp = pi_amp / 2
pi_half_wf = pi_half_amp * gaussian(pi_half_len, pi_half_sigma)

x180_len = 40
x180_sigma = x180_len / 5
x180_amp = 0.35
x180_drag_wf, x180_drag_der_wf = np.array(
    drag_gaussian_pulse_waveforms(x180_amp, x180_len, x180_sigma, alpha=0, delta=1)
)
minus_x180_drag_wf, minus_x180_drag_der_wf = np.array(
    drag_gaussian_pulse_waveforms((-1)*x180_amp, x180_len, x180_sigma, alpha=0, delta=1)
)
# No DRAG when alpha=0, it's just a gaussian.

x90_len = x180_len
x90_sigma = x90_len / 5
x90_amp = x180_amp / 2
x90_drag_wf, x90_drag_der_wf = np.array(drag_gaussian_pulse_waveforms(x90_amp, x90_len, x90_sigma, alpha=0, delta=1))
minus_x90_drag_wf, minus_x90_drag_der_wf = np.array(drag_gaussian_pulse_waveforms((-1)*x90_amp, x90_len, x90_sigma, alpha=0, delta=1))
# No DRAG when alpha=0, it's just a gaussian.

# Resonator
resonator_IF = -50e6
resonator_LO = 5.5e9
mixer_resonator_g = 0.005
mixer_resonator_phi = -0.085

time_of_flight = 180 + 112

passive_len = 4000
short_readout_len = 500
short_readout_amp = 0.4
readout_len = 4000
readout_amp = 0.28  # 0.28 V is -59 dBm after splitter
long_readout_len = 50000
long_readout_amp = 0.004

# JPC pump
on_len = 20

# IQ Plane
rotation_angle = (-70.0 / 180) * np.pi
ge_threshold = 0.0

# Squeeze drive
squeeze_IF = -80e6
# squeeze_IF = 2*qubit_IF
squeeze_LO = 4e9
mixer_squeeze_g = -0.02
mixer_squeeze_phi = -0.16

ftc_rise_len = 1000
ftc_fall_len = 1000
ftc_amp = cat_pump
# ftc_rise_wf = gaussian_rise(ftc_amp, ftc_rise_len, ftc_rise_len/5)
ftc_rise_wf = ftc_amp*ring_up_tanh(ftc_rise_len)
# ftc_fall_wf = gaussian_fall(ftc_amp, ftc_fall_len, ftc_fall_len/5)
ftc_fall_wf = ftc_amp*ring_up_wave(ftc_fall_len, reverse=True, type='tanh')

ftc_len = 100
ftc_flat = 80
# ftc_amp = 0.2
# ftc_wf = flat_top_gaussian(ftc_amp, ftc_len, ftc_flat)
ftc_wf = ftc_amp*smoothed_constant_wave(ftc_len, int((ftc_len-ftc_flat)/2), type = 'tanh')

ftc_len_trunc = 100
ftc_flat_trunc = 80
ftc_amp_trunc = 0.1
ftc_trunc_wf = gaussian_rise_const(ftc_amp_trunc, ftc_len_trunc, ftc_flat_trunc)

ftc_len_trunc_fall = 100
ftc_flat_trunc_fall = 80
ftc_amp_trunc_fall = 0.1
ftc_trunc_fall_wf = gaussian_fall_const(ftc_amp_trunc_fall, ftc_len_trunc_fall, ftc_flat_trunc_fall)


# CQR drive
cqr_IF = resonator_IF - (squeeze_IF/2)
# cqr_IF = -15e6
cqr_LO = 6e9
mixer_cqr_g = 0.0935
mixer_cqr_phi = 0.0312

cqr_rise_len = 20
cqr_fall_len = 20
cqr_amp = 0.15
cqr_rise_wf = gaussian_rise(cqr_amp, cqr_rise_len, cqr_rise_len/5)
cqr_fall_wf = gaussian_fall(cqr_amp, cqr_fall_len, cqr_fall_len/5)

cqr_len = 4000
cqr_flat = 3000
# cqr_amp = 0.2
# cqr_wf = flat_top_gaussian(cqr_amp, cqr_len, cqr_flat)
cqr_wf = cqr_amp*smoothed_constant_wave(cqr_len, int((cqr_len-cqr_flat)/2), type = 'tanh')


config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": -0.0139},  # Q Fock qubit
                2: {"offset": -0.0051},  # I Fock qubit
                3: {"offset": 0.026},  # Q squeeze drive
                4: {"offset": 0.006},  # I squeeze drive
                5: {"offset":  -0.0059},  # Q cqr drive
                6: {"offset": -0.0081},  # I cqr drive
                7: {"offset": 0.048},  # Q resonator
                8: {"offset": 0.026},  # I resonator
            },
            "digital_outputs": {
                1: {},  # Fock qubit switch
                2: {},  # squeeze drive switch
                3: {},  # cqr drive switch
                4: {},  # resonator switch
                5: {},  # JPC pump switch
            },
            "analog_inputs": {
                1: {"offset": -0.0042024609375, "gain_db": 10},  # I from down-conversion
                2: {"offset": 0.0, "gain_db": 0},  # Q from down-conversion
            },
        },
        "con2": {
            "analog_outputs": {
                1: {"offset": 0.0},  # I Fock qubit
                2: {"offset": 0.0},  # Q Fock qubit
                3: {"offset": 0.0},  # I squeeze drive
                4: {"offset": 0.0},  # Q squeeze drive
                5: {"offset": 0.0},  # I cqr drive
                6: {"offset": 0.0},  # Q cqr drive
                7: {"offset": 0.0},  # I resonator
                8: {"offset": 0.0},  # Q resonator
            },
            "digital_outputs": {
                1: {},  # Fock qubit switch
                2: {},  # squeeze drive switch
                3: {},  # cqr drive switch
                4: {},  # resonator switch
                5: {},  # JPC pump switch
            },
            "analog_inputs": {
                1: {"offset": 0.0, "gain_db": 0},  # I from down-conversion
                2: {"offset": 0.0, "gain_db": 0},  # Q from down-conversion
            },
        },
    },
    "elements": {
        "Fock_qubit": {
            "mixInputs": {
                "I": ("con1", 2),
                "Q": ("con1", 1),
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
                "x180": "x180_pulse",
                "x90": "x90_pulse",
                "x270": "x270_pulse",
                "y180": "y180_pulse",
                "y90": "y90_pulse",
                "y270": "y270_pulse",
                "rise":"rise_pulse",
                "fall":"fall_pulse",
            },
        },
        "squeeze_drive": {
            "mixInputs": {
                "I": ("con1", 4),
                "Q": ("con1", 3),
                "lo_frequency": squeeze_LO,
                "mixer": "mixer_squeeze",
            },
            "intermediate_frequency": squeeze_IF,
            "operations": {
                "cw": "const_pulse",
                "-cw": "minus_const_pulse",
                "saturation": "saturation_pulse",
                "gauss": "gaussian_pulse",
                "pi": "pi_pulse",
                "pi_half": "pi_half_pulse",
                "x180": "x180_pulse",
                "x90": "x90_pulse",
                "ftc_rise": "ftc_rise_pulse",
                "ftc_fall": "ftc_fall_pulse",
                "ftc": "ftc_pulse",
                "ftc_trunc": "ftc_trunc_pulse",
                "ftc_trunc_fall": "ftc_trunc_fall_pulse",
            },
        },

        "squeeze_rise": {
            "mixInputs": {
                "I": ("con1", 4),
                "Q": ("con1", 3),
                "lo_frequency": squeeze_LO,
                "mixer": "mixer_squeeze",
            },
            "intermediate_frequency": squeeze_IF,
            "operations": {
                "cw": "const_pulse",
                "saturation": "saturation_pulse",
                "gauss": "gaussian_pulse",
                "pi": "pi_pulse",
                "pi_half": "pi_half_pulse",
                "x180": "x180_pulse",
                "x90": "x90_pulse",
                "ftc_rise": "ftc_rise_pulse",
                "-ftc_rise": "minus_ftc_rise_pulse",
                "ftc_fall": "ftc_fall_pulse",
            },
        },

        "squeeze_fall": {
            "mixInputs": {
                "I": ("con1", 4),
                "Q": ("con1", 3),
                "lo_frequency": squeeze_LO,
                "mixer": "mixer_squeeze",
            },
            "intermediate_frequency": squeeze_IF,
            "operations": {
                "cw": "const_pulse",
                "saturation": "saturation_pulse",
                "gauss": "gaussian_pulse",
                "pi": "pi_pulse",
                "pi_half": "pi_half_pulse",
                "x180": "x180_pulse",
                "x90": "x90_pulse",
                "ftc_rise": "ftc_rise_pulse",
                "ftc_fall": "ftc_fall_pulse",
                "-ftc_fall": "minus_ftc_fall_pulse",
            },
        },

        "cqr_drive": {
            "mixInputs": {
                "I": ("con1", 6),
                "Q": ("con1", 5),
                "lo_frequency": cqr_LO,
                "mixer": "mixer_cqr",
            },
            "intermediate_frequency": cqr_IF,
            "operations": {
                "cw": "const_pulse",
                "saturation": "saturation_pulse",
                "gauss": "gaussian_pulse",
                "pi": "x180_pulse",
                "pi_half": "x90_pulse",
                "x180": "x180_pulse",
                "x90": "x90_pulse",
                "cqr_rise": "cqr_rise_pulse",
                "cqr_fall": "cqr_fall_pulse",
                "cqr": "cqr_pulse"
            },
        },

        "cqr_rise": {
            "mixInputs": {
                "I": ("con1", 6),
                "Q": ("con1", 5),
                "lo_frequency": cqr_LO,
                "mixer": "mixer_cqr",
            },
            "intermediate_frequency": cqr_IF,
            "operations": {
                "cw": "const_pulse",
                "saturation": "saturation_pulse",
                "gauss": "gaussian_pulse",
                "pi": "x180_pulse",
                "pi_half": "x90_pulse",
                "x180": "x180_pulse",
                "x90": "x90_pulse",
                "cqr_rise": "cqr_rise_pulse",
                "cqr_fall": "cqr_fall_pulse",
            },
        },

        "cqr_fall": {
            "mixInputs": {
                "I": ("con1", 6),
                "Q": ("con1", 5),
                "lo_frequency": cqr_LO,
                "mixer": "mixer_cqr",
            },
            "intermediate_frequency": cqr_IF,
            "operations": {
                "cw": "const_pulse",
                "saturation": "saturation_pulse",
                "gauss": "gaussian_pulse",
                "pi": "x180_pulse",
                "pi_half": "x90_pulse",
                "x180": "x180_pulse",
                "x90": "x90_pulse",
                "cqr_rise": "cqr_rise_pulse",
                "cqr_fall": "cqr_fall_pulse",
            },
        },

        "resonator": {
            "mixInputs": {
                "I": ("con1", 8),
                "Q": ("con1", 7),
                "lo_frequency": resonator_LO,
                "mixer": "mixer_resonator",
            },
            "intermediate_frequency": resonator_IF,
            "operations": {
                "cw": "const_pulse",
                "short_readout": "short_readout_pulse",
                "readout": "readout_pulse",
                "passive_readout": "passive_readout_pulse",
                "long_readout": "long_readout_pulse",
            },
            "outputs": {
                "out1": ("con1", 1),
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
        },
        "Fock_switch": {
            'digitalInputs': {
                'switch': {
                    'port': ('con1', 1),
                    'delay': 136,  # needs between play and digital
                    'buffer': 0,
                },
            },
            'operations': {
                'saturation_on': 'saturation_on_pulse_Fock',
                'gauss_on': 'gauss_on_pulse_Fock',
                'x_on': 'x_on_pulse_Fock',
            },
        },
        "squeeze_switch": {
            'digitalInputs': {
                'switch': {
                    'port': ('con1', 2),
                    'delay': 136,
                    'buffer': 0,
                },
            },
            'operations': {
                'cw_on': 'cw_on_pulse_squeeze',
                'on': 'on_pulse_squeeze',
            },
        },
        "cqr_switch": {
            'digitalInputs': {
                'switch': {
                    'port': ('con1', 3),
                    'delay': 136,
                    'buffer': 0,
                },
            },
            'operations': {
                'on': 'on_pulse_cqr',
            },
        },
        "resonator_switch": {
            'digitalInputs': {
                'switch': {
                    'port': ('con1', 4),
                    'delay': 136,
                    'buffer': 0,
                },
            },
            'operations': {
                'on': 'on_pulse_resonator',
                'short_on': 'short_on_pulse_resonator',
                'long_on': 'long_on_pulse_resonator',
            },
        },
        "SPC_pump": {
            'digitalInputs': {
                'switch': {
                    'port': ('con1', 5),
                    'delay': 136,
                    'buffer': 0,
                },
            },
            'operations': {
                'on': 'on_pulse_JPC',
            },
        },
    },
    "pulses": {
        "rise_pulse": {
            "operation": 'control',
            'length': rise_len,
            'waveforms': {
                'I': "rise_wf",
                'Q': 'zero_wf',
            },
        },
        
        "fall_pulse": {
            "operation": 'control',
            'length': fall_len,
            'waveforms': {
                'I': "fall_wf",
                'Q': 'zero_wf',
            },
        },
        
        
        "ftc_pulse": {
            "operation": 'control',
            'length': ftc_len,
            'waveforms': {
                'I': "ftc_wf",
                'Q': 'zero_wf',
            },
        },
        "ftc_trunc_pulse": {
            "operation": 'control',
            'length': ftc_len_trunc,
            'waveforms': {
                'I': "ftc_trunc_wf",
                'Q': 'zero_wf',
            },
        },
        "ftc_trunc_fall_pulse": {
            "operation": 'control',
            'length': ftc_len_trunc_fall,
            'waveforms': {
                'I': "ftc_trunc_fall_wf",
                'Q': 'zero_wf',
            },
        },
        "cqr_pulse": {
            "operation": 'control',
            'length': cqr_len,
            'waveforms': {
                'I': "cqr_wf",
                'Q': 'zero_wf',
            },
        },
        "ftc_rise_pulse": {
            'operation': 'control',
            'length': ftc_rise_len,
            "waveforms": {
                "I": "ftc_rise_wf",
                "Q": "zero_wf",
            },
        },
        "minus_ftc_rise_pulse": {
            'operation': 'control',
            'length': ftc_rise_len,
            "waveforms": {
                "I": "minus_ftc_rise_wf",
                "Q": "zero_wf",
            },
        },
        "ftc_fall_pulse": {
            'operation': 'control',
            'length': ftc_fall_len,
            "waveforms": {
                "I": "ftc_fall_wf",
                "Q": "zero_wf",
            },
        },
        "minus_ftc_fall_pulse": {
            'operation': 'control',
            'length': ftc_fall_len,
            "waveforms": {
                "I": "minus_ftc_fall_wf",
                "Q": "zero_wf",
            },
        },
        "cqr_rise_pulse": {
            'operation': 'control',
            'length': cqr_rise_len,
            "waveforms": {
                "I": "cqr_rise_wf",
                "Q": "zero_wf",
            },
        },
        "cqr_fall_pulse": {
            'operation': 'control',
            'length': cqr_fall_len,
            "waveforms": {
                "I": "cqr_fall_wf",
                "Q": "zero_wf",
            },
        },
        "saturation_on_pulse_Fock": {
            'operation': 'control',
            'length': saturation_len,
            'digital_marker': 'ON',
        },
        "gauss_on_pulse_Fock": {
            'operation': 'control',
            'length': gauss_len,
            'digital_marker': 'ON',
        },
        "x_on_pulse_Fock": {
            'operation': 'control',
            'length': x180_len,
            'digital_marker': 'ON',
        },
        "cw_on_pulse_squeeze": {
            'operation': 'control',
            'length': const_len,
            'digital_marker': 'ON',
        },
        "on_pulse_squeeze": {
            'operation': 'control',
            'length': on_len,
            'digital_marker': 'ON',
        },
        "on_pulse_cqr": {
            'operation': 'control',
            'length': on_len,
            'digital_marker': 'ON',
        },
        "on_pulse_resonator": {
            'operation': 'control',
            'length': readout_len,
            'digital_marker': 'ON',
        },
        "short_on_pulse_resonator": {
            'operation': 'control',
            'length': short_readout_len,
            'digital_marker': 'ON',
        },
        "long_on_pulse_resonator": {
            'operation': 'control',
            'length': long_readout_len,
            'digital_marker': 'ON',
        },
        "on_pulse_JPC": {
            'operation': 'control',
            'length': on_len,
            'digital_marker': 'ON',
        },
        "const_pulse": {
            "operation": "control",
            "length": const_len,
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
        },
        "minus_const_pulse": {
            "operation": "control",
            "length": const_len,
            "waveforms": {
                "I": "minus_const_wf",
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
            "length": pi_half_len,
            "waveforms": {
                "I": "pi_half_wf",
                "Q": "zero_wf",
            },
        },
        "x180_pulse": {
            "operation": "control",
            "length": x180_len,
            "waveforms": {
                "I": "x180_drag_wf",
                "Q": "x180_drag_der_wf",
            },
        },
        "x90_pulse": {
            "operation": "control",
            "length": x90_len,
            "waveforms": {
                "I": "x90_drag_wf",
                "Q": "x90_drag_der_wf",
            },
        },
        "x270_pulse": {
            "operation": "control",
            "length": x90_len,
            "waveforms": {
                "I": "minus_x90_drag_wf",
                "Q": "minus_x90_drag_der_wf",
            },
        },
        "y180_pulse": {
            "operation": "control",
            "length": x180_len,
            "waveforms": {
                "I": "minus_x180_drag_der_wf",
                "Q": "x180_drag_wf",
            },
        },
        "y90_pulse": {
            "operation": "control",
            "length": x90_len,
            "waveforms": {
                "I": "minus_x90_drag_der_wf",
                "Q": "x90_drag_wf",
            },
        },
        "y270_pulse": {
            "operation": "control",
            "length": x90_len,
            "waveforms": {
                "I": "x90_drag_der_wf",
                "Q": "minus_x90_drag_wf",
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
        "passive_readout_pulse": {
            "operation": "measurement",
            "length": passive_len,
            "waveforms": {
                "I": "zero_wf",
                "Q": "zero_wf",
            },
            "integration_weights": {
                "cos": "p_cosine_weights",
                "sin": "p_sine_weights",
                "minus_sin": "p_minus_sine_weights",
                "rotated_cos": "p_rotated_cosine_weights",
                "rotated_sin": "p_rotated_sine_weights",
                "rotated_minus_sin": "p_rotated_minus_sine_weights",
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
        "minus_const_wf": {"type": "constant", "sample": (-1)*const_amp},
        "saturation_drive_wf": {"type": "constant", "sample": saturation_amp},
        "square_pi_wf": {"type": "constant", "sample": square_pi_amp},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "gauss_wf": {"type": "arbitrary", "samples": gauss_wf.tolist()},
        "pi_wf": {"type": "arbitrary", "samples": pi_wf.tolist()},
        "pi_half_wf": {"type": "arbitrary", "samples": pi_half_wf.tolist()},
        "x180_drag_wf": {"type": "arbitrary", "samples": x180_drag_wf.tolist()},
        "x180_drag_der_wf": {"type": "arbitrary", "samples": x180_drag_der_wf.tolist()},
        "x90_drag_wf": {"type": "arbitrary", "samples": x90_drag_wf.tolist()},
        "x90_drag_der_wf": {"type": "arbitrary", "samples": x90_drag_der_wf.tolist()},
        "minus_x180_drag_wf": {"type": "arbitrary", "samples": minus_x180_drag_wf.tolist()},
        "minus_x180_drag_der_wf": {"type": "arbitrary", "samples": minus_x180_drag_der_wf.tolist()},
        "minus_x90_drag_wf": {"type": "arbitrary", "samples": minus_x90_drag_wf.tolist()},
        "minus_x90_drag_der_wf": {"type": "arbitrary", "samples": minus_x90_drag_der_wf.tolist()},
        "short_readout_wf": {"type": "constant", "sample": short_readout_amp},
        "readout_wf": {"type": "constant", "sample": readout_amp},
        "long_readout_wf": {"type": "constant", "sample": long_readout_amp},
        "ftc_rise_wf": {'type': 'arbitrary', "samples": ftc_rise_wf},
        "minus_ftc_rise_wf": {'type': 'arbitrary', "samples": (-1)*ftc_rise_wf},
        "ftc_fall_wf": {'type': 'arbitrary', "samples": ftc_fall_wf},
        "minus_ftc_fall_wf": {'type': 'arbitrary', "samples": (-1)*ftc_fall_wf},
        "cqr_rise_wf": {'type': 'arbitrary', "samples": cqr_rise_wf},
        "cqr_fall_wf": {'type': 'arbitrary', "samples": cqr_fall_wf},
        'cqr_wf': {'type': 'arbitrary', 'samples': cqr_wf},
        'ftc_wf': {'type': 'arbitrary', 'samples': ftc_wf},
        'ftc_trunc_wf': {'type': 'arbitrary', 'samples': ftc_trunc_wf},
        'ftc_trunc_fall_wf': {'type': 'arbitrary', 'samples': ftc_trunc_fall_wf},
        "rise_wf": {'type': 'arbitrary', "samples": rise_wf},
        "fall_wf": {'type': 'arbitrary', "samples": fall_wf},
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
        "p_cosine_weights": {
            "cosine": [(1.0, passive_len)],
            "sine": [(0.0, passive_len)],
        },
        "p_sine_weights": {
            "cosine": [(0.0, passive_len)],
            "sine": [(1.0, passive_len)],
        },
        "p_minus_sine_weights": {
            "cosine": [(0.0, passive_len)],
            "sine": [(-1.0, passive_len)],
        },
        "p_rotated_cosine_weights": {
            "cosine": [(np.cos(rotation_angle), passive_len)],
            "sine": [(-np.sin(rotation_angle), passive_len)],
        },
        "p_rotated_sine_weights": {
            "cosine": [(np.sin(rotation_angle), passive_len)],
            "sine": [(np.cos(rotation_angle), passive_len)],
        },
        "p_rotated_minus_sine_weights": {
            "cosine": [(-np.sin(rotation_angle), passive_len)],
            "sine": [(-np.cos(rotation_angle), passive_len)],
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
        "mixer_squeeze": [
            {
                "intermediate_frequency": squeeze_IF,
                "lo_frequency": squeeze_LO,
                "correction": IQ_imbalance(mixer_squeeze_g, mixer_squeeze_phi),
            }
        ],
        "mixer_cqr": [
            {
                "intermediate_frequency": cqr_IF,
                "lo_frequency": cqr_LO,
                "correction": IQ_imbalance(mixer_cqr_g, mixer_cqr_phi),
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
