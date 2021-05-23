from qm.qua import *
from bakery import *
pulse_len = 100
readout_len = 400
qubit_IF = 50e6
rr_IF = 50e6
qubit_LO = 6.345e9
rr_LO = 4.755e9


def gauss(amplitude, mu, sigma, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    return [float(x) for x in gauss_wave]


def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g ** 2) * (2 * c ** 2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


gauss_pulse = gauss(0.2, 0, 12, pulse_len)

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},  # qe-I
                2: {"offset": +0.0},  # qe-Q
                3: {"offset": +0.0},  # rr-I
                4: {"offset": +0.0},  # rr-Q
            },
            "digital_outputs": {
                1: {},
            },
            "analog_inputs": {
                1: {"offset": +0.0},
            },
        }
    },
    "elements": {
        "qe1": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
            },
            "outputs": {"output1": ("con1", 1)},
            "intermediate_frequency": 0,
            "operations": {
                "I" : "IPulse",
                "X/2": "X/2Pulse",
                "X": "XPulse",
                "-X/2": "-X/2Pulse",
                "Y/2": "Y/2Pulse",
                "Y": "YPulse",
                "-Y/2": "-Y/2Pulse",
            },
            "time_of_flight": 180,
            "smearing": 0,
        },
        "rr": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": rr_LO,
                "mixer": "mixer_RR",
            },
            "intermediate_frequency": rr_IF,
            "operations": {
                "readout": "readout_pulse",
            },
            "outputs": {"out1": ("con1", 1)},
            "time_of_flight": 28,
            "smearing": 0,
        },
    },
    "pulses": {
        "constPulse": {
            "operation": "control",
            "length": pulse_len,
            "waveforms": {"I": "gauss_wf", "Q": "gauss_wf"},
        },
        "IPulse":{
            "operation": "control",
            "length": pulse_len,
            "waveforms": {"I": "zero_wf", "Q": "zero_wf"},
        },
        "XPulse": {
            "operation": "control",
            "length": pulse_len,
            "waveforms": {"I": "pi_wf", "Q": "zero_wf"},
        },
        "X/2Pulse": {
            "operation": "control",
            "length": pulse_len,
            "waveforms": {"I": "pi/2_wf", "Q": "zero_wf"},
        },
        "-X/2Pulse": {
            "operation": "control",
            "length": pulse_len,
            "waveforms": {"I": "-pi/2_wf", "Q": "zero_wf"},
        },
        "YPulse": {
            "operation": "control",
            "length": pulse_len,
            "waveforms": {"I": "zero_wf", "Q": "pi_wf"},
        },
        "Y/2Pulse": {
            "operation": "control",
            "length": pulse_len,
            "waveforms": {"I": "zero_wf", "Q": "pi/2_wf"},
        },
        "-Y/2Pulse": {
            "operation": "control",
            "length": pulse_len,
            "waveforms": {"I": "zero_wf", "Q": "-pi/2_wf"},
        },
        "readout_pulse": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {"I": "readout_wf", "Q": "zero_wf"},
            "integration_weights": {
                "integW1": "integW1",
                "integW2": "integW2",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
        "gauss_wf": {"type": "arbitrary", "samples": gauss_pulse},
        "pi_wf": {"type": "arbitrary", "samples": gauss(0.2, 0, 12, pulse_len)},
        "-pi/2_wf": {"type": "arbitrary", "samples": gauss(-0.1, 0, 12, pulse_len)},
        "pi/2_wf": {"type": "arbitrary", "samples": gauss(0.1, 0, 12, pulse_len)},
        "zero_wf": {"type": "constant", "sample": 0},
        "readout_wf": {"type": "constant", "sample": 0.3},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "integW1": {
            "cosine": [1.0] * int(readout_len / 4),
            "sine": [0.0] * int(readout_len / 4),
        },
        "integW2": {
            "cosine": [0.0] * int(readout_len / 4),
            "sine": [1.0] * int(readout_len / 4),
        },
    },
    "mixers": {
        "mixer_qubit": [
            {
                "intermediate_frequency": qubit_IF,
                "lo_frequency": qubit_LO,
                "correction": IQ_imbalance(0.0, 0.0),
            }
        ],
        "mixer_RR": [
            {
                "intermediate_frequency": rr_IF,
                "lo_frequency": rr_LO,
                "correction": IQ_imbalance(0.0, 0.0),
            }
        ],
    },
}

# The list of 1 Qubit cliffords, X are pi rotations, X/2 are pi/2 rotations around the X axis (Y accordingly)
cliffords = [
    ["I"],
    ["X"],
    ["Y"],
    ["Y", "X"],
    ["X/2", "Y/2"],
    ["X/2", "-Y/2"],
    ["-X/2", "Y/2"],
    ["-X/2", "-Y/2"],
    ["Y/2", "X/2"],
    ["Y/2", "-X/2"],
    ["-Y/2", "X/2"],
    ["-Y/2", "-X/2"],
    ["X/2"],
    ["-X/2"],
    ["Y/2"],
    ["-Y/2"],
    ["-X/2", "Y/2", "X/2"],
    ["-X/2", "-Y/2", "X/2"],
    ["X", "Y/2"],
    ["X", "-Y/2"],
    ["Y", "X/2"],
    ["Y", "-X/2"],
    ["X/2", "Y/2", "X/2"],
    ["-X/2", "Y/2", "-X/2"],
]
operations = {
        "z": ["I"],
        "-x": ["-Y/2"],
        "y": ["X/2"],
        "-y": ["-X/2"],
        "x": ["Y/2"],
        "-z": ["X"],
    }

transformations = {
        "x": {
            "I": "x",
            "X/2": "x",
            "X": "x",
            "-X/2": "x",
            "Y/2": "z",
            "Y": "-x",
            "-Y/2": "-z",
        },
        "-x": {
            "I": "-x",
            "X/2": "-x",
            "X": "-x",
            "-X/2": "-x",
            "Y/2": "-z",
            "Y": "x",
            "-Y/2": "z",
        },
        "y": {
            "I": "y",
            "X/2": "z",
            "X": "-y",
            "-X/2": "-z",
            "Y/2": "y",
            "Y": "y",
            "-Y/2": "y",
        },
        "-y": {
            "I": "-y",
            "X/2": "-z",
            "X": "y",
            "-X/2": "z",
            "Y/2": "-y",
            "Y": "-y",
            "-Y/2": "-y",
        },
        "z": {
            "I": "z",
            "X/2": "-y",
            "X": "-z",
            "-X/2": "y",
            "Y/2": "-x",
            "Y": "-z",
            "-Y/2": "x",
        },
        "-z": {
            "I": "-z",
            "X/2": "y",
            "X": "z",
            "-X/2": "-y",
            "Y/2": "x",
            "Y": "z",
            "-Y/2": "-x",
        },
    }


def recovery_clifford(state: str):
    """
    Returns the required clifford to return to the ground state based on the position on the bloch sphere
    :param state: The current position on the Bloch sphere
    :return: A string representing the recovery clifford
    """
    # operations = {'x': ['I'], '-x': ['Y'], 'y': ['X/2', '-Y/2'], '-y': ['-X/2', '-Y/2'], 'z': ['-Y/2'], '-z': ['Y/2']}

    return operations[state]


def transform_state(input_state: str, transformation: str):
    """
    A function to track the next position on the Bloch sphere based on the current position and the applied clifford
    :param input_state: Position on the bloch sphere (one of the six poles)
    :param transformation: A clifford operation
    :return: The next state on the bloch sphere
    """

    return transformations[input_state][transformation]


def play_clifford(clifford: list, state: str, b: Baking):
    """

    :param clifford: a list of cliffords
    :param state: a string representing the current state on the bloch sphere
    :param b: Baking object where the sequence should be generated
    :return: the final state on the bloch sphere
    """
    for op in clifford:
        state = transform_state(state, op)
        if op != "I":
            b.play(op, "qe1")
    return state


def randomize_and_play_circuit(n_gates: int, b: Baking, init_state: str = "z"):
    """

    :param n_gates: the depth of the circuit
    :param init_state: starting position on the bloch sphere
    :param b: Baking object
    :return:
    """
    state = init_state
    for ind in range(n_gates):
        state = play_clifford(cliffords[np.random.randint(0, len(cliffords))], state, b)
    return state


def randomize_interleaved_circuit(interleave_op: list, d: int, init_state: str = "z"):
    """
    :param interleave_op: The operation to interleave represented as a list of cliffords
    :param d: the depth of the circuit
    :param init_state: the initial state on the bloch sphere
    :return: the final state on the bloch spehre
    """
    state = init_state
    for ind in range(d):
        state = play_clifford(cliffords[np.random.randint(0, len(cliffords))], state)
        state = play_clifford(interleave_op, state)
    return state


def generate_cliffords(b: Baking, qe: str, pulse_length: int):

    short_pi = gauss(0.2, 0, 1, pulse_length)
    short_pi_2 = gauss(0.1, 0., 1, pulse_length)
    short_minus_pi_2 = gauss(-0.1, 0., 1, pulse_length)
    short_0 = [0.] * pulse_length

    b.add_Op("X", qe, [short_pi, short_0])
    b.add_Op("Y", qe, [short_0, short_pi])
    b.add_Op("X/2", qe, [short_pi_2, short_0])
    b.add_Op("Y/2", qe, [short_0, short_pi_2])
    b.add_Op("-X/2", qe, [short_minus_pi_2, short_0])
    b.add_Op("-Y/2", qe, [short_0, short_minus_pi_2])


def measure_state(state, I):
    """
    A measurement function depending on the type of qubit.
    This example implementation is typical of a SC qubit measurement (via a dispersive readout)
    :param state: a QUA var where the state will be saved
    :param I: a QUA var containing the demod result
    :return: none
    """
    th = 0
    measure("readout", "rr", None, integration.full("integW1", I))
    assign(state, I > th)


def active_reset(state):

    with if_(state):
        play("X", "qe1")
