# helper functions pertinent to single- and two-qubit tomography

import numpy as np
import functools

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm, colormaps


titlefont = {"color": "black", "weight": "normal", "size": 10}
axisfont = {"color": "black", "weight": "normal", "size": 8}
ticksfont = {"color": "black", "weight": "normal", "size": 8}


from qm.qua import *


def rotated_multiplexed_state_discrimination(I, I_st, Q, Q_st, states, states_st, resonators, thresholds):
    """
    Perform multiplexed state discrimination on two qubits

    :param I: List of QUA variables to which the I quadrature measurements
    are written
    :param I_st: Optional list of QUA streams to which I values are streamed
    :param Q: List of QUA variables to which the Q quadrature measurements
    are written
    :param states: List of QUA variables to which qubit state are written
    :param states_st: Optional list of QUA streams to which qubit states
    are streamed
    :param resonators: List of integers denoting readout resonators
    appearing in the config file as "ff{res}" elements
    :param thresholds: List of readout thresholds from a calibrated
    IQ blob two-state discriminator. I values above each threshold
    signify that the qubit was found to be in the excited state

    :return:
    """

    if type(resonators) is not list:
        resonators = [resonators]

    if type(thresholds) is not list:
        thresholds = [thresholds]

    for ind, res in enumerate(resonators):
        measure(
            "readout",
            f"rr{res}",
            None,
            dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I[ind]),
            dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q[ind]),
        )

        assign(states[ind], I[ind] > thresholds[ind])

        if states_st is not None:
            save(states[ind], states_st[ind])
        if I_st is not None:
            save(I[ind], I_st[ind])
        if Q_st is not None:
            save(Q[ind], Q_st[ind])


@functools.lru_cache()
def func_F1(n: int):
    # one of the six gates to create one of the six cardinal Bloch sphere
    # states
    if n == 0:  # identity
        return np.array([[1, 0], [0, 1]])
    elif n == 1:  # X180/Y180
        return np.array([[0, 1], [1, 0]])
    elif n == 2:  # Y90
        return (1 / np.sqrt(2)) * np.array([[1, -1], [1, 1]])
    elif n == 3:  # Y-90
        return (1 / np.sqrt(2)) * np.array([[1, 1], [-1, 1]])
    elif n == 4:  # X-90
        return (1 / np.sqrt(2)) * np.array([[1, 1j], [1j, 1]])
    elif n == 5:  # X90
        return (1 / np.sqrt(2)) * np.array([[1, -1j], [-1j, 1]])
    else:  # Error
        raise ValueError("Input integer should be between 0 and 5, inclusive")


@functools.lru_cache()
def func_E1(n: int):
    # four Pauli operators
    if n == 0:  # identity
        return np.array([[1, 0], [0, 1]])
    elif n == 1:  # X
        return np.array([[0, 1], [1, 0]])
    elif n == 2:  # Y
        return np.array([[0, -1j], [1j, 0]])
    elif n == 3:  # Z
        return np.array([[1, 0], [0, -1]])
    else:  # Error
        raise ValueError("Input integer should be between 0 and 3, inclusive")


@functools.lru_cache()
def func_c1(i: int, j: int):
    # Constants c1[i,j] such that
    # func_E1[i] = sum_j c1[i,j]func_F1[j]|0><0|func_F1[j]^{\dagger}
    if (i, j) == (0, 0):
        return 1
    elif (i, j) == (0, 1):
        return 1
    elif (i, j) == (1, 0):
        return -(1 + 1j)
    elif (i, j) == (1, 1):
        return -(1 + 1j)
    elif (i, j) == (1, 2):
        return 2
    elif (i, j) == (1, 4):
        return 1j
    elif (i, j) == (1, 5):
        return 1j
    elif (i, j) == (2, 4):
        return 1
    elif (i, j) == (2, 5):
        return -1
    elif (i, j) == (3, 0):
        return 1
    elif (i, j) == (3, 1):
        return -1
    else:
        return 0


def B_Bloch1(i, j, m, n):

    # start from qubit ground state, create one of six Bloch sphere states
    # using func_F1, add Pauli operator basis states func_E1, and measure one
    # of the six Bloch sphere state projectors,
    # func_F1^{\dagger} |0><0| func_F1. These constants appear when expressing
    # the Pauli states in terms of combinations of Bloch sphere states,
    # assuming you start from the qubit ground state

    starting_state = np.array([[1, 0], [0, 0]])
    bloch_state = func_F1(i) @ starting_state @ (func_F1(i).conj().T)
    add_paulis = func_E1(m) @ bloch_state @ (func_E1(n).conj().T)
    measure_bloch_state = (func_F1(j).conj().T) @ starting_state @ func_F1(j) @ add_paulis

    return measure_bloch_state.trace()


def P_Pauli1(l, k, m, n):

    result = 0

    for i in range(0, 6):
        for j in range(0, 6):
            result += func_c1(l, i) * np.conj(func_c1(k, j)) * B_Bloch1(i, j, m, n)

    return result


def map_from_bloch_state_to_pauli_basis1(l, k, arr):

    if arr.shape != (6, 6):
        raise ValueError("Input array must be 6x6")

    if (k not in [0, 1, 2, 3]) or (l not in [0, 1, 2, 3]):
        raise ValueError("Input indices must be between 0 and 3, inclusive")

    result = 0

    for i in range(0, 6):
        for j in range(0, 6):
            result += func_c1(l, i) * np.conj(func_c1(k, j)) * arr[i][j]

    return result


def plot_process_tomography1(chi_vector, save_file: str = None):

    plt.rcParams["text.usetex"] = True

    cmap = colormaps.get_cmap("viridis")

    # set up figure
    fig = plt.figure(figsize=(9, 3), dpi=250, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    # coordinates
    r = np.arange(4)
    _x, _y = np.meshgrid(r, r)
    x, y = _x.ravel(), _y.ravel()

    values0 = np.abs(chi_vector)
    values1 = np.angle(chi_vector)

    top = values0
    bottom = np.zeros_like(top)

    width = depth = 0.7

    norm = Normalize(vmin=-np.pi, vmax=np.pi)
    colors = cmap(norm(values1))

    xy_ticks_labels = [r"$I$", r"$X$", r"$Y$", r"$Z$"]

    ax.bar3d(x, y, bottom, width, depth, top, shade=True, color=colors)
    ax.set_xticks(r + 0.5, labels=xy_ticks_labels)
    ax.set_yticks(r + 0.5, labels=xy_ticks_labels)
    ax.set_zticks([0, (max(top) / 2).round(2), max(top).round(2)])
    ax.set_xlabel("Prepared", fontdict=axisfont)
    ax.set_ylabel("Measured", fontdict=axisfont)
    ax.set_title(r"$\chi$ matrix", fontdict=titlefont)
    ax.view_init(20, -60, 0)

    sc = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_ticks(
        ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        labels=[r"$-\pi$", r"$-\pi/2$", r"0", r"$\pi/2$", r"$\pi$"],
    )

    if save_file:
        plt.savefig(save_file, bbox_inches="tight")

    plt.show()


@functools.lru_cache()
def func_F2(n: int):
    # one of the six gates to create one of the six cardinal Bloch sphere
    # states
    if n == 0:  # identity
        return np.array([[1, 0], [0, 1]])
    elif n == 1:  # X180/Y180
        return np.array([[0, 1], [1, 0]])
    elif n == 2:  # Y90
        return (1 / np.sqrt(2)) * np.array([[1, -1], [1, 1]])
    elif n == 3:  # Y-90
        return (1 / np.sqrt(2)) * np.array([[1, 1], [-1, 1]])
    elif n == 4:  # X-90
        return (1 / np.sqrt(2)) * np.array([[1, 1j], [1j, 1]])
    elif n == 5:  # X90
        return (1 / np.sqrt(2)) * np.array([[1, -1j], [-1j, 1]])
    else:  # Error
        raise ValueError("Input integer should be between 0 and 5, inclusive")


@functools.lru_cache()
def func_E2(n: int):
    # the 16 two-qubit Pauli operators

    i = np.array([[1, 0], [0, 1]])
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.array([[1, 0], [0, -1]])

    if n == 0:  # I x I
        return np.kron(i, i)
    elif n == 1:  # I x X
        return np.kron(i, x)
    elif n == 2:  # I x Y
        return np.kron(i, y)
    elif n == 3:  # I x Z
        return np.kron(i, z)
    elif n == 4:  # X x I
        return np.kron(x, i)
    elif n == 5:  # X x X
        return np.kron(x, x)
    elif n == 6:  # X x Y
        return np.kron(x, y)
    elif n == 7:  # X x Z
        return np.kron(x, z)
    elif n == 8:  # Y x I
        return np.kron(y, i)
    elif n == 9:  # Y x X
        return np.kron(y, x)
    elif n == 10:  # Y x Y
        return np.kron(y, y)
    elif n == 11:  # Y x Z
        return np.kron(y, z)
    elif n == 12:  # Z x I
        return np.kron(z, i)
    elif n == 13:  # Z x X
        return np.kron(z, x)
    elif n == 14:  # Z x Y
        return np.kron(z, y)
    elif n == 15:  # Z x Z
        return np.kron(z, z)
    else:
        raise ValueError("Input integer should be between 0 and 15, inclusive")


@functools.lru_cache()
def func_c2(i: int, j: int, k: int):
    # Constants c2[i, j, k] such that
    # func_E2[i] = sum_{j,k} func_c2[i,j,k] (func_F2[j] x func_F2[k]) |0>|0><0|<0| (func_F2[j]^{\dagger} x func_F2[k]^{\dagger})
    if (i, j, k) == (0, 0, 0):
        return 1
    elif (i, j, k) == (0, 0, 1):
        return 1
    elif (i, j, k) == (0, 1, 0):
        return 1
    elif (i, j, k) == (0, 1, 1):
        return 1
    elif (i, j, k) == (1, 0, 0):
        return -(1 + 1j)
    elif (i, j, k) == (1, 0, 1):
        return -(1 + 1j)
    elif (i, j, k) == (1, 0, 2):
        return 2
    elif (i, j, k) == (1, 0, 4):
        return 1j
    elif (i, j, k) == (1, 0, 5):
        return 1j
    elif (i, j, k) == (1, 1, 0):
        return -(1 + 1j)
    elif (i, j, k) == (1, 1, 1):
        return -(1 + 1j)
    elif (i, j, k) == (1, 1, 2):
        return 2
    elif (i, j, k) == (1, 1, 4):
        return 1j
    elif (i, j, k) == (1, 1, 5):
        return 1j
    elif (i, j, k) == (2, 0, 4):
        return 1
    elif (i, j, k) == (2, 0, 5):
        return -1
    elif (i, j, k) == (2, 1, 4):
        return 1
    elif (i, j, k) == (2, 1, 5):
        return -1
    elif (i, j, k) == (3, 0, 0):
        return 1
    elif (i, j, k) == (3, 0, 1):
        return -1
    elif (i, j, k) == (3, 1, 0):
        return 1
    elif (i, j, k) == (3, 1, 1):
        return -1
    elif (i, j, k) == (4, 0, 0):
        return -(1 + 1j)
    elif (i, j, k) == (4, 0, 1):
        return -(1 + 1j)
    elif (i, j, k) == (4, 1, 0):
        return -(1 + 1j)
    elif (i, j, k) == (4, 1, 1):
        return -(1 + 1j)
    elif (i, j, k) == (4, 2, 0):
        return 2
    elif (i, j, k) == (4, 2, 1):
        return 2
    elif (i, j, k) == (4, 4, 0):
        return 1j
    elif (i, j, k) == (4, 4, 1):
        return 1j
    elif (i, j, k) == (4, 5, 0):
        return 1j
    elif (i, j, k) == (4, 5, 1):
        return 1j
    elif (i, j, k) == (5, 0, 0):
        return (1 + 1j) ** 2
    elif (i, j, k) == (5, 0, 1):
        return (1 + 1j) ** 2
    elif (i, j, k) == (5, 1, 0):
        return (1 + 1j) ** 2
    elif (i, j, k) == (5, 1, 1):
        return (1 + 1j) ** 2
    elif (i, j, k) == (5, 0, 2):
        return -2 * (1 + 1j)
    elif (i, j, k) == (5, 1, 2):
        return -2 * (1 + 1j)
    elif (i, j, k) == (5, 2, 0):
        return -2 * (1 + 1j)
    elif (i, j, k) == (5, 2, 1):
        return -2 * (1 + 1j)
    elif (i, j, k) == (5, 0, 4):
        return -1j * (1 + 1j)
    elif (i, j, k) == (5, 0, 5):
        return -1j * (1 + 1j)
    elif (i, j, k) == (5, 1, 4):
        return -1j * (1 + 1j)
    elif (i, j, k) == (5, 1, 5):
        return -1j * (1 + 1j)
    elif (i, j, k) == (5, 4, 0):
        return -1j * (1 + 1j)
    elif (i, j, k) == (5, 4, 1):
        return -1j * (1 + 1j)
    elif (i, j, k) == (5, 5, 0):
        return -1j * (1 + 1j)
    elif (i, j, k) == (5, 5, 1):
        return -1j * (1 + 1j)
    elif (i, j, k) == (5, 2, 2):
        return 4
    elif (i, j, k) == (5, 2, 4):
        return 2 * 1j
    elif (i, j, k) == (5, 2, 5):
        return 2 * 1j
    elif (i, j, k) == (5, 4, 2):
        return 2 * 1j
    elif (i, j, k) == (5, 5, 2):
        return 2 * 1j
    elif (i, j, k) == (5, 4, 4):
        return -1
    elif (i, j, k) == (5, 4, 5):
        return -1
    elif (i, j, k) == (5, 5, 4):
        return -1
    elif (i, j, k) == (5, 5, 5):
        return -1
    elif (i, j, k) == (6, 0, 4):
        return -(1 + 1j)
    elif (i, j, k) == (6, 1, 4):
        return -(1 + 1j)
    elif (i, j, k) == (6, 0, 5):
        return 1 + 1j
    elif (i, j, k) == (6, 1, 5):
        return 1 + 1j
    elif (i, j, k) == (6, 2, 4):
        return 2
    elif (i, j, k) == (6, 2, 5):
        return -2
    elif (i, j, k) == (6, 4, 4):
        return 1j
    elif (i, j, k) == (6, 5, 4):
        return 1j
    elif (i, j, k) == (6, 4, 5):
        return -1j
    elif (i, j, k) == (6, 5, 5):
        return -1j
    elif (i, j, k) == (7, 0, 0):
        return -(1 + 1j)
    elif (i, j, k) == (7, 1, 0):
        return -(1 + 1j)
    elif (i, j, k) == (7, 0, 1):
        return 1 + 1j
    elif (i, j, k) == (7, 1, 1):
        return 1 + 1j
    elif (i, j, k) == (7, 2, 0):
        return 2
    elif (i, j, k) == (7, 2, 1):
        return -2
    elif (i, j, k) == (7, 4, 0):
        return 1j
    elif (i, j, k) == (7, 5, 0):
        return 1j
    elif (i, j, k) == (7, 4, 1):
        return -1j
    elif (i, j, k) == (7, 5, 1):
        return -1j
    elif (i, j, k) == (8, 4, 0):
        return 1
    elif (i, j, k) == (8, 4, 1):
        return 1
    elif (i, j, k) == (8, 5, 0):
        return -1
    elif (i, j, k) == (8, 5, 1):
        return -1
    elif (i, j, k) == (9, 4, 0):
        return -(1 + 1j)
    elif (i, j, k) == (9, 4, 1):
        return -(1 + 1j)
    elif (i, j, k) == (9, 5, 0):
        return 1 + 1j
    elif (i, j, k) == (9, 5, 1):
        return 1 + 1j
    elif (i, j, k) == (9, 4, 4):
        return 1j
    elif (i, j, k) == (9, 4, 5):
        return 1j
    elif (i, j, k) == (9, 5, 4):
        return -1j
    elif (i, j, k) == (9, 5, 5):
        return -1j
    elif (i, j, k) == (9, 4, 2):
        return 2
    elif (i, j, k) == (9, 5, 2):
        return -2
    elif (i, j, k) == (10, 4, 4):
        return 1
    elif (i, j, k) == (10, 5, 5):
        return 1
    elif (i, j, k) == (10, 4, 5):
        return -1
    elif (i, j, k) == (10, 5, 4):
        return -1
    elif (i, j, k) == (11, 4, 0):
        return 1
    elif (i, j, k) == (11, 5, 1):
        return 1
    elif (i, j, k) == (11, 4, 1):
        return -1
    elif (i, j, k) == (11, 5, 0):
        return -1
    elif (i, j, k) == (12, 0, 0):
        return 1
    elif (i, j, k) == (12, 0, 1):
        return 1
    elif (i, j, k) == (12, 1, 0):
        return -1
    elif (i, j, k) == (12, 1, 1):
        return -1
    elif (i, j, k) == (13, 0, 0):
        return -(1 + 1j)
    elif (i, j, k) == (13, 0, 1):
        return -(1 + 1j)
    elif (i, j, k) == (13, 1, 0):
        return 1 + 1j
    elif (i, j, k) == (13, 1, 1):
        return 1 + 1j
    elif (i, j, k) == (13, 0, 2):
        return 2
    elif (i, j, k) == (13, 1, 2):
        return -2
    elif (i, j, k) == (13, 0, 4):
        return 1j
    elif (i, j, k) == (13, 0, 5):
        return 1j
    elif (i, j, k) == (13, 1, 4):
        return -1j
    elif (i, j, k) == (13, 1, 5):
        return -1j
    elif (i, j, k) == (14, 0, 4):
        return 1
    elif (i, j, k) == (14, 1, 5):
        return 1
    elif (i, j, k) == (14, 0, 5):
        return -1
    elif (i, j, k) == (14, 1, 4):
        return -1
    elif (i, j, k) == (15, 0, 0):
        return 1
    elif (i, j, k) == (15, 1, 1):
        return 1
    elif (i, j, k) == (15, 0, 1):
        return -1
    elif (i, j, k) == (15, 1, 0):
        return -1
    else:
        return 0


def B_Bloch2(i, j, k, l, m, n):

    # start from qubits ground states, create one of six Bloch sphere states
    # using func_F2 for each of the two qubits, add one of the 16
    # Pauli operator basis func_E2, and measure one of the six Bloch sphere
    # state projectors for each qubit, i.e.
    # func_F2[.]^{\dagger}func_F2[.]^{\dagger} |0>|0><0|<0| func_F2[.] func_F2[.].
    # These B_Bloch2 constants appear when expressing the Pauli states
    # in terms of combinations of Bloch sphere states, assuming you
    # start from the qubits' ground states

    starting_state = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    bloch_state = np.kron(func_F2(i), func_F2(j)) @ starting_state @ np.kron(func_F2(i).conj().T, func_F2(j).conj().T)
    add_paulis = func_E2(m) @ bloch_state @ (func_E2(n).conj().T)
    measure_bloch_state = (
        np.kron(func_F2(k).conj().T, func_F2(l).conj().T)
        @ starting_state
        @ np.kron(func_F2(k), func_F2(l))
        @ add_paulis
    )

    return measure_bloch_state.trace()


def P_Pauli2(s, t, m, n):

    result = 0

    for i in range(0, 6):
        for j in range(0, 6):
            for k in range(0, 6):
                for l in range(0, 6):
                    result += func_c2(s, i, j) * np.conj(func_c2(t, k, l)) * B_Bloch2(i, j, k, l, m, n)

    return result


def map_from_bloch_state_to_pauli_basis2(q, n, arr):

    if arr.shape != (6, 6, 6, 6):
        raise ValueError("Input array must be 6x6x6x6")

    if (q not in range(0, 16)) or (n not in range(0, 16)):
        raise ValueError("Input indices must be between 0 and 15, inclusive")

    result = 0

    for i in range(0, 6):  # first qubit prepare
        for j in range(0, 6):  # second qubit prepare
            for k in range(0, 6):  # first qubit measure
                for l in range(0, 6):  # second qubit measure
                    result += func_c2(q, i, j) * np.conj(func_c2(n, k, l)) * arr[i][j][k][l]

    return result


def plot_process_tomography2(chi_vector, save_file: str = None):

    plt.rcParams["text.usetex"] = True

    ticksfont = {"color": "black", "weight": "normal", "size": 4}

    cmap = colormaps.get_cmap("viridis")

    # set up figure
    fig = plt.figure(figsize=(9, 3), dpi=250, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    # coordinates
    r = np.arange(16)
    _x, _y = np.meshgrid(r, r)
    x, y = _x.ravel(), _y.ravel()

    values0 = np.abs(chi_vector)
    values1 = np.angle(chi_vector)

    top = values0
    bottom = np.zeros_like(top)

    width = depth = 0.7

    norm = Normalize(vmin=-np.pi, vmax=np.pi)
    colors = cmap(norm(values1))

    xy_ticks_labels = [
        r"$II$",
        r"$IX$",
        r"$IY$",
        r"$IZ$",
        r"$XI$",
        r"$XX$",
        r"$XY$",
        r"$XZ$",
        r"$YI$",
        r"$YX$",
        r"$YY$",
        r"$YZ$",
        r"$ZI$",
        r"$ZX$",
        r"$ZY$",
        r"$ZZ$",
    ]

    ax.bar3d(x, y, bottom, width, depth, top, shade=True, color=colors)
    ax.set_xticks(r + 0.5, labels=xy_ticks_labels, fontdict=ticksfont)
    ax.set_yticks(r + 0.5, labels=xy_ticks_labels, fontdict=ticksfont)
    ax.set_zticks([0, (max(top) / 2).round(2), max(top).round(2)])
    ax.set_xlabel("Prepared", fontdict=axisfont)
    ax.set_ylabel("Measured", fontdict=axisfont)
    ax.set_title(r"$\chi$ matrix", fontdict=titlefont)
    ax.view_init(20, -60, 0)

    sc = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_ticks(
        ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        labels=[r"$-\pi$", r"$-\pi/2$", r"0", r"$\pi/2$", r"$\pi$"],
    )

    if save_file:
        plt.savefig(save_file, bbox_inches="tight")

    plt.show()
