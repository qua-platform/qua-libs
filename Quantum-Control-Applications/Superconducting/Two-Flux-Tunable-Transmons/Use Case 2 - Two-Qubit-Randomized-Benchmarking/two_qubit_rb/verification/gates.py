import numpy as np
from dataclasses import dataclass

import cirq


@dataclass
class Gate:
    def matrix(self) -> np.ndarray:
        raise NotImplementedError()


@dataclass
class PhasedXZ(Gate):
    q: int  # control qubit index
    x: float  # amplitude scaling float
    z: float
    a: float

    def __str__(self):
        return f"PXZ({self.q}, amp={self.x}, z={self.z}, a={self.a})"

    def matrix(self):
        phased_xz = cirq.PhasedXZGate(axis_phase_exponent=self.a, x_exponent=self.x, z_exponent=self.z)

        phased_xz = cirq.unitary(phased_xz)
        I = cirq.unitary(cirq.I)

        if self.q == 1:
            return np.kron(phased_xz, I)
        elif self.q == 2:
            return np.kron(I, phased_xz)
        else:
            raise NotImplementedError()


@dataclass
class CZ(Gate):
    def __str__(self):
        return f"CZ"

    def matrix(self):
        return cirq.unitary(cirq.CZ)


@dataclass
class CNOT(Gate):
    q: int  # control qubit index

    def __str__(self):
        return f"CNOT({self.q}, {2 if self.q == 1 else 1})"

    def matrix(self):
        if self.q == 1:
            cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        elif self.q == 2:
            cnot = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
        else:
            raise NotImplementedError()

        return cnot
