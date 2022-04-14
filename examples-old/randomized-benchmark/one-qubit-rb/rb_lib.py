import numpy as np
from qm.qua import *

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


def recovery_clifford(state: str):
    """
    Returns the required clifford to return to the ground state based on the position on the bloch sphere
    :param state: The current position on the Bloch sphere
    :return: A string representing the recovery clifford
    """
    # operations = {'x': ['I'], '-x': ['Y'], 'y': ['X/2', '-Y/2'], '-y': ['-X/2', '-Y/2'], 'z': ['-Y/2'], '-z': ['Y/2']}
    operations = {
        "z": ["I"],
        "-x": ["-Y/2"],
        "y": ["X/2"],
        "-y": ["-X/2"],
        "x": ["Y/2"],
        "-z": ["X"],
    }
    return operations[state]


def transform_state(input_state: str, transformation: str):
    """
    A function to track the next position on the Bloch sphere based on the current position and the applied clifford
    :param input_state: Position on the bloch sphere (one of the six poles)
    :param transformation: A clifford operation
    :return: The next state on the bloch sphere
    """
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

    return transformations[input_state][transformation]


def play_clifford(clifford: list, state: str):
    """

    :param clifford: a list of cliffords
    :param state: a string representing the current state on the bloch sphere
    :return: the final state on the bloch sphere
    """
    for op in clifford:
        state = transform_state(state, op)
        if op != "I":
            play(op, "qe1")
    return state


def randomize_and_play_circuit(n_gates: int, init_state: str = "z"):
    """

    :param n_gates: the depth of the circuit
    :param init_state: starting position on the bloch sphere
    :return:
    """
    state = init_state
    for ind in range(n_gates):
        state = play_clifford(cliffords[np.random.randint(0, len(cliffords))], state)
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
