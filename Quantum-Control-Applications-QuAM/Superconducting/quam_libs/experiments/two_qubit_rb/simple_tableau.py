from typing import Tuple, Union

import numpy as np


def _beta(v, u):
    lut = [[0, 0, 0, 0], [0, 0, 3, 1], [0, 1, 0, 3], [0, 3, 1, 0]]
    lut_map = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}
    n = len(u) // 2
    beta = 0
    for i in range(n):
        beta += lut[lut_map[tuple(v[2 * i : 2 * i + 2])]][lut_map[tuple(u[2 * i : 2 * i + 2])]]
    return beta


def _compose_alpha(g1, alpha1, g2, alpha2):
    n = len(alpha1) // 2
    alpha21 = []
    two_alpha21 = np.zeros(2 * n, dtype=np.uint8)
    for i in range(2 * n):
        b_i = _calc_b_i(g1, g2, i)
        two_alpha21[i] += (2 * alpha1[i] + 2 * np.dot(g1[:, i], alpha2) + b_i) % 4
        assert two_alpha21[i] % 2 == 0
        alpha21.append(two_alpha21[i] // 2)
    return np.array(alpha21).astype(np.uint8)


def _calc_b_i(g1, g2, i):
    # add i for every Y in the column of g1
    b_i = np.dot(g1[::2, i], g1[1::2, i]) % 4
    n = g1.shape[0] // 2

    # reduce using beta
    current = np.zeros(2 * n, dtype=np.uint8)
    for j in range(2 * n):
        b_i = (b_i + _beta(current, g1[j, i] * g2[:, j])) % 4
        current = (current + g1[j, i] * g2[:, j]) % 2
    return b_i


def _calc_inverse_alpha(g1, alpha1):
    n = len(alpha1) // 2
    lam = _lambda(n)
    inv_g1 = lam @ g1.T @ lam % 2
    b = np.array([_calc_b_i(g1, inv_g1, i) for i in range(2 * n)])
    two_alpha2 = -(inv_g1.T @ (2 * alpha1 + b)) % 4
    assert np.all(two_alpha2 % 2 == 0)
    return two_alpha2 // 2


class SimpleTableau:
    """
    A class for representing and acting of Clifford Tableaus using only elementary logical and
    arithmetic operations. Implements the `CliffordTableau` API.
    """

    def __init__(self, g, alpha):
        g = np.array(g, dtype=np.uint8)
        alpha = np.array(alpha, dtype=np.uint8)
        if not len(alpha) % 2 == 0:
            raise ValueError(f"alpha needs to have an even length but length is {len(alpha)}")
        g_shape = g.shape
        if not (len(g_shape) == 2 and g_shape[0] == g_shape[1] and g_shape[0] % 2 == 0):
            raise ValueError(f"g has shape {g_shape}, which is not an even square matrix")
        if not len(alpha) % 2 == 0:
            raise ValueError(f"alpha has len {len(alpha)}, which is not even")
        self._n = len(alpha) // 2
        if not _is_symplectic(g, self._n):
            raise ValueError("g is not a symplectic matrix")
        self._np_repr = np.vstack((g, alpha)).astype(np.uint8)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def g(self):
        return self._np_repr[:-1, :]

    @property
    def alpha(self):
        return self._np_repr[-1]

    @property
    def n(self):
        return self._n

    def __str__(self):
        # todo: work with more than single digit qubit number
        st = " " * 2 + "|" + " ".join(f"x{i} z{i}" for i in range(self._n)) + "\n"
        st += "-" * 2 + "+" + "-" * 6 * self._n + "\n"
        for i in range(2 * self._n):
            st += f"z{i // 2}|" if i % 2 else f"x{i // 2}|"
            st += "  ".join(str(entry) for entry in self._np_repr[i]) + "\n"
        st += "s |" + "  ".join(str(entry) for entry in self._np_repr[i + 1]) + "\n"
        return st

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return np.array_equal(self._np_repr, other._np_repr)

    def __hash__(self):
        return hash(self._np_repr.tobytes())

    def then(self, other: "SimpleTableau") -> "SimpleTableau":
        if self.n != other.n:
            raise ValueError(f"number of qubits of self={self.n} and of other={other.n} is incompatible")
        g12 = other.g @ self.g % 2
        alpha12 = _compose_alpha(self.g, self.alpha, other.g, other.alpha)
        return SimpleTableau(g12, alpha12)

    def inverse(self) -> "SimpleTableau":
        lam = _lambda(self.n)
        return SimpleTableau((lam @ self.g.T @ lam) % 2, _calc_inverse_alpha(self.g, self.alpha))

    def is_identity(self):
        return np.array_equal(self.g, np.eye(2 * self.n)) and np.array_equal(self.alpha, np.zeros(2 * self.n))


_single_qubit_gate_conversions = {
    "I": (np.identity(2), np.zeros(2)),
    "H": (np.array([[0, 1], [1, 0]]), np.zeros(2)),
    "X": (np.identity(2), np.array([0, 1])),
    "Z": (np.identity(2), np.array([1, 0])),
    "Y": (np.identity(2), np.array([1, 1])),
    "S": (np.array([[1, 0], [1, 1]]), np.zeros(2)),
    "SX": (np.array([[1, 1], [0, 1]]), np.array([0, 1])),
    "SY": (np.array([[0, 1], [1, 0]]), np.array([1, 0])),
    "-SY": (np.array([[0, 1], [1, 0]]), np.array([0, 1])),
    "-SX": (np.array([[1, 1], [0, 1]]), np.array([0, 0])),
}

_two_qubit_gate_conversions = {
    "CNOT": (np.array([[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1]]).T, np.zeros(4)),
    "ISWAP": (
        np.array([[0, 1, 1, 1], [1, 1, 0, 1], [0, 0, 0, 1], [0, 0, 1, 0]]).T,
        np.zeros(4),
    ),  # not 100% sure this is correct
    "SWAP": (np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]).T, np.zeros(4)),
    "CZ": (np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]]).T, np.zeros(4)),
}

known_names = set(_single_qubit_gate_conversions.keys()).union(set(_two_qubit_gate_conversions.keys()))


def generate_from_name(name: str, target: Union[int, Tuple[int, int]], n=None) -> SimpleTableau:
    """
    Generate a `SimpleTableau` object from a name.
    Args:
        name: gate name, must be in `known_names`
        target: For a single qubit gate, the qubit on which the gate operates, from 0 to n-1. For two qubit gates,
                a tuple of the form (control, target) on which the gate operates
        n: number of total qubits on which gate operates. If omitted, will be determined by the highest target index.

    Returns: the `SimpleTableau` object

    """
    if n is None:
        n = np.max(target) + 1
    g = np.identity(2 * n)
    alpha = np.zeros(2 * n)
    if name in _single_qubit_gate_conversions:
        if not isinstance(target, int) or target >= n:
            raise ValueError(f"invalid target {target} for single qubit gate {name} on {n} qubits")
        _embed_single_qubit_gate(alpha, g, name, target)
        return SimpleTableau(g, alpha)

    elif name in _two_qubit_gate_conversions:
        if not isinstance(target, tuple) or max(target) >= n:
            raise ValueError(f"invalid target {target} for two qubit gate {name} on {n} qubits")
        _embed_two_qubit_gate(alpha, g, name, target)
        return SimpleTableau(g, alpha)
    else:
        raise ValueError(f"unknown gate {name}")


def _embed_two_qubit_gate(alpha, g, name, target):
    g2q, alpha2q = _two_qubit_gate_conversions[name]
    g[2 * target[0] : 2 * target[0] + 2, 2 * target[0] : 2 * target[0] + 2] = g2q[:2, :2]
    g[2 * target[0] : 2 * target[0] + 2, 2 * target[1] : 2 * target[1] + 2] = g2q[:2, 2:4]
    g[2 * target[1] : 2 * target[1] + 2, 2 * target[0] : 2 * target[0] + 2] = g2q[2:4, :2]
    g[2 * target[1] : 2 * target[1] + 2, 2 * target[1] : 2 * target[1] + 2] = g2q[2:4, 2:4]
    alpha[2 * target[0] : 2 * target[0] + 2] = alpha2q[:2]
    alpha[2 * target[1] : 2 * target[1] + 2] = alpha2q[2:4]


def _embed_single_qubit_gate(alpha, g, name, target):
    g[2 * target : 2 * target + 2, 2 * target : 2 * target + 2] = _single_qubit_gate_conversions[name][0]
    alpha[2 * target : 2 * target + 2] = _single_qubit_gate_conversions[name][1]


def _lambda(n):
    return np.diag([1] + [0, 1] * (n - 1), 1) + np.diag([1] + [0, 1] * (n - 1), -1)


def _is_symplectic(mat, n):
    lhs = mat @ _lambda(n) @ mat.T % 2
    return np.all(lhs == _lambda(n))


_int_bit_map = {0: [0, 0], 1: [1, 0], 2: [1, 1], 3: [0, 1]}


def stim_to_simple(tableau) -> SimpleTableau:
    n = len(tableau)
    g = np.zeros((2 * n, 2 * n), dtype=np.uint8)
    alpha = np.zeros(2 * n, dtype=np.uint8)
    for j in range(0, 2 * n, 2):
        pauli_string_x = tableau.x_output(j // 2)
        pauli_string_z = tableau.z_output(j // 2)
        alpha[j] = pauli_string_x.sign.real < 0
        alpha[j + 1] = pauli_string_z.sign.real < 0
        for i in range(0, 2 * n, 2):
            g[i : i + 2, j] = _int_bit_map[pauli_string_x[i // 2]]
            g[i : i + 2, j + 1] = _int_bit_map[pauli_string_z[i // 2]]
    return SimpleTableau(g, alpha)
