from functools import lru_cache

import numpy as np
from dataclasses import dataclass

import cirq


I = cirq.unitary(cirq.I)

@dataclass(frozen=True)
class Gate:
    def matrix(self) -> np.ndarray:
        raise NotImplementedError()

    def gate_str(self) -> str:
        raise NotImplementedError()


@dataclass(frozen=True)
class PhasedXZ(Gate):
    q: int  # control qubit index
    x: float  # amplitude scaling float
    z: float
    a: float

    def __str__(self):
        return f"PXZ({self.q}, amp={self.x}, z={self.z}, a={self.a})"

    def gate_str(self) -> str:
        gate_str = ""
        if self.x != 0:
            if self.a != 0:
                gate_str += f"Z^{{-{self.a}}}"
            gate_str += f"X^{{{self.x}}}"
            if self.a != 0:
                gate_str += f"Z^{{{self.a}}}"
        if self.z != 0:
            gate_str += f"Z^{{{self.z}}}"
        if gate_str == "":
            gate_str += "I"

        gate_str = gate_str.replace('1.0', '')

        return gate_str

    @lru_cache(maxsize=None)
    def matrix(self):
        phased_xz = cirq.PhasedXZGate(axis_phase_exponent=self.a, x_exponent=self.x, z_exponent=self.z)

        phased_xz = cirq.unitary(phased_xz)
        I = cirq.unitary(cirq.I)

        if self.q == 1:
            return np.kron(I, phased_xz)
        elif self.q == 2:
            return np.kron(phased_xz, I)
        else:
            raise NotImplementedError()


@dataclass(frozen=True)
class CZ(Gate):
    def __str__(self):
        return f"CZ"

    def matrix(self):
        return cirq.unitary(cirq.CZ)

    def gate_str(self) -> str:
        return f"CZ"


@dataclass(frozen=True)
class CNOT(Gate):
    q: int  # control qubit index

    def __str__(self):
        return f"CNOT({self.q}, {2 if self.q == 1 else 1})"

    def matrix(self):
        if self.q == 1:
            cnot = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
        elif self.q == 2:
            cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        else:
            raise NotImplementedError()

        return cnot

    def gate_str(self) -> str:
        return f"CNOT({self.q}, {2 if self.q == 1 else 1})"
