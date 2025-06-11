"""
This script contains useful QUA macros for the two-qubit cross-entropy benchmarking use case.

Author: Arthur Strauss - Quantum Machines
Last updated: 2024-04-30
"""

import pandas as pd
from matplotlib import pyplot as plt, colors
from qm.qua import *
from qualang_tools.addons.variables import assign_variables_to_element
import numpy as np
from scipy.optimize import optimize
from scipy.stats import stats


def assign_amplitude_matrix(gate, amp_matrix, gate_dict: dict):
    """
    QUA Macro for assigning the amplitude matrix arguments for a given gate index.
    :param gate: Gate index
    :param amp_matrix: Amplitude matrix arguments
    :param gate_dict: Dictionary of gates
    """
    with switch_(gate):
        for i in range(len(gate_dict)):
            with case_(i):
                for j in range(4):
                    assign(amp_matrix[j], gate_dict[i]["amp_matrix"][j])


def qua_declaration(n_qubits: int, readout_elements: list):
    """
    Macro to declare the necessary QUA variables

    :param n_qubits: Number of qubits used in this experiment
    :param readout_elements: List of readout elements
    :return:
    """
    I, Q = [[declare(fixed) for _ in range(n_qubits)] for _ in range(2)]
    I_st, Q_st = [[declare_stream() for _ in range(n_qubits)] for _ in range(2)]
    # Workaround to manually assign the results variables to the readout elements
    for i in range(n_qubits):
        assign_variables_to_element(readout_elements[i], I[i], Q[i])
    return I, I_st, Q, Q_st


def play_T_gate_set(gate, amp_matrix, qubit_el):
    """
    QUA Macro for the T gate set
    The two first cases will focus on the X90 and Y90 gates, while the last case will focus on the T gate.
    :param gate: Gate index
    :param amp_matrix: Amplitude matrix
    :param qubit_el: Qubit element
    """
    with switch_(gate, unsafe=True):
        for j in range(2):  # X90, Y90
            with case_(j):
                play("sx" * amp(*amp_matrix), qubit_el)
        with case_(2):  # T gate
            frame_rotation_2pi(0.125, qubit_el)


def play_SW_gate_set(gate, amp_matrix, qubit_el):
    """
    QUA Macro for the SW gate set
    The two first cases will focus on the X90 and Y90 gates, while the last case will focus on the SW gate.
    :param gate: Gate index
    :param amp_matrix: Amplitude matrix
    :param qubit_el: Qubit element
    """
    with switch_(gate, unsafe=True):
        for j, gate in enumerate(["sx", "sy"]):  # X90, Y90, SW
            with case_(j):
                play(gate, qubit_el)
        with case_(2):
            play("sx" * amp(*amp_matrix), qubit_el)


def play_random_sq_gate(gate, amp_matrix, qubit_el, gate_dict: dict):
    """
    QUA Macro for playing a random single-qubit gate
    :param gate: Gate index
    :param amp_matrix: Amplitude matrix
    :param qubit_el: Qubit element
    :param gate_dict: Dictionary of gates
    """

    if gate_dict[2]["gate"].name == "t":  # T gate involves frame rotation
        play_T_gate_set(gate, amp_matrix, qubit_el)
    elif gate_dict[2]["gate"].label == "sw":
        play_SW_gate_set(gate, amp_matrix, qubit_el)
    else:
        play("sx" * amp(*amp_matrix), qubit_el)


def cz_gate(control, target, CZ_operations: dict):
    """
    QUA Macro for the CZ gate
    :param control: Control qubit index
    :param target: Target qubit index
    :param CZ_operations: Dictionary of CZ operations (control, target) -> (cz_el, cz_op)
    """
    cz_el, cz_op = CZ_operations[(control, target)]
    play(cz_op, cz_el)


def binary(n, length):
    """
    Convert an integer to a binary string of a given length
    :param n: Integer to convert
    :param length: Length of the output string
    :return: Binary string corresponding to integer n
    """
    return bin(n)[2:].zfill(length)


def cross_entropy(p, q, epsilon=1e-15):
    """
    Calculate cross entropy between two probability distributions.

    Parameters:
    - p: numpy array, the true probability distribution
    - q: numpy array, the predicted probability distribution
    - epsilon: small value to avoid taking the logarithm of zero

    Returns:
    - Cross entropy between p and q
    """
    q = np.maximum(q, epsilon)  # Avoid taking the logarithm of zero
    x_entropy = -np.sum(p * np.log(q))
    return x_entropy


def create_subplot(data, subplot_number, title, depths, seqs):
    print(title)
    print("data: %s" % data)
    print(subplot_number)
    plt.subplot(subplot_number)
    # plt.pcolor(depths, range(seqs), np.abs(data), vmin=0., vmax=1.)
    plt.pcolor(depths, range(seqs), np.abs(data))
    ax = plt.gca()
    ax.set_title(title)
    if subplot_number > 244:
        ax.set_xlabel("Circuit depth")
    ax.set_ylabel("Sequences")
    ax.set_xticks(depths)
    ax.set_yticks(np.arange(1, seqs + 1))
    plt.colorbar()


def per_cycle_depth(df):
    fid_lsq = df["numerator"].sum() / df["denominator"].sum()

    cycle_depth = df.name
    xx = np.linspace(0, df["x"].max())
    (l,) = plt.plot(xx, fid_lsq * xx, color=colors[cycle_depth])
    plt.scatter(df["x"], df["y"], color=colors[cycle_depth])

    global _lines
    _lines += [l]  # for legend
    return pd.Series({"fidelity": fid_lsq})


# Define Cirq functions for fitting (redefined here for avoiding additional dependencies)
# Those functions are slightly adapted to deal with possible singularities and outliers in the data
def exponential_decay(cycle_depths: np.ndarray, a: float, layer_fid: float) -> np.ndarray:
    """An exponential decay for fitting.

    This computes `a * layer_fid**cycle_depths`

    Args:
        cycle_depths: The various depths at which fidelity was estimated. This is the independent
            variable in the exponential function.
        a: A scale parameter in the exponential function.
        layer_fid: The base of the exponent in the exponential function.
    """
    return a * layer_fid**cycle_depths


def fit_exponential_decay(cycle_depths: np.ndarray, fidelities: np.ndarray) -> tuple[float, float, float, float]:
    """Fit an exponential model fidelity = a * layer_fid**x using nonlinear least squares.

    This uses `exponential_decay` as the function to fit with parameters `a` and `layer_fid`.
    This function is taken from the following Cirq code: https://github.com/quantumlib/Cirq/blob/main/cirq-core/cirq/experiments/xeb_fitting.py

    Args:
        cycle_depths: The various depths at which fidelity was estimated. Each element is `x`
            in the fit expression.
        fidelities: The estimated fidelities for each cycle depth. Each element is `fidelity`
            in the fit expression.

    Returns:
        a: The first fit parameter that scales the exponential function, perhaps accounting for
            state prep and measurement (SPAM) error.
        layer_fid: The second fit parameters which serves as the base of the exponential.
        a_std: The standard deviation of the `a` parameter estimate.
        layer_fid_std: The standard deviation of the `layer_fid` parameter estimate.
    """
    cycle_depths = np.asarray(cycle_depths)
    fidelities = np.asarray(fidelities)
    mask = (fidelities > 0) & (fidelities < 1)
    print()
    masked_cycle_depths = cycle_depths[mask]
    masked_fidelities = fidelities[mask]

    log_fidelities = np.log(masked_fidelities)

    slope, intercept, _, _, _ = stats.linregress(masked_cycle_depths, log_fidelities)
    layer_fid_0 = np.clip(np.exp(slope), 0, 1)
    a_0 = np.clip(np.exp(intercept), 0, 1)

    try:
        (a, layer_fid), pcov = optimize.curve_fit(
            exponential_decay,
            masked_cycle_depths,
            masked_fidelities,
            p0=(a_0, layer_fid_0),
            bounds=((0, 0), (1, 1)),
            nan_policy="omit",
        )
    except ValueError:  # pragma: no cover
        return 0, 0, np.inf, np.inf

    a_std, layer_fid_std = np.sqrt(np.diag(pcov))
    return a, layer_fid, a_std, layer_fid_std
