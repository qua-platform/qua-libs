from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig, LoopbackInterface
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from configuration import config, gauss, gauss_der

QMm = QuantumMachinesManager()
QM1 = QMm.open_qm(config)

n_qubits = 3  # Number of system qubits.
n_shots = 1e6  # Number of quantum measurements.
tot_qubits = n_qubits + 1  # Addition of an ancillary qubit.
ancilla_idx = n_qubits  # Index of the ancillary qubit (last position).
steps = 30  # Number of optimization steps
eta = 0.8  # Learning rate
q_delta = 0.001  # Initial spread of random quantum weights
rng_seed = 0  # Seed for random number generator
# Coefficients of the linear combination A = c_0 A_0 + c_1 A_1 ...
c = np.array([1.0, 0.2, 0.2])


def hadamard(qubits):
    """Unitary matrix rotating the ground state to the problem vector |b> = U_b |0>."""
    for qubit in qubits:
        play("hadamard", qubit)


def U_b():
    hadamard(["q0", "q1", "q2"])


def CNOT(control, target):
    play("X", target)


def CZ(control, target):
    play("X", target)


def CA(idx):
    """Controlled versions of the unitary components A_l of the problem matrix A."""
    if idx == 0:
        # Identity operation
        None
    elif idx == 1:
        CNOT(control="q_anc", target=f"q{idx}")
        CZ(control="q_anc", target=f"q{idx}")
    elif idx == 2:
        CNOT(control="q_anc", target=f"q{idx}")


def Y(theta, qubit):
    """Rotation by arbitrary angle around Y axis"""
    play("X", qubit, duration=theta)


def variational_block(weights):
    """Variational circuit mapping the ground state |0> to the ansatz state |x>."""
    # We first prepare an equal superposition of all the states of the computational basis.
    hadamard(["q0", "q1", "q2"])

    # A very minimal variational circuit.
    for idx, w in enumerate(weights):
        Y(w, f"q{idx}")


def local_hadamard_test(weights, l=None, lp=None, j=None, part=None):
    # First Hadamard gate applied to the ancillary qubit.
    hadamard("q_anc")
    # For estimating the imaginary part of the coefficient "mu", we must add a "-i"
    # phase gate.
    if part.lower() == "im":
        frame_rotation(-np.pi / 2, "q_anc")
    # Variational circuit generating a guess for the solution vector |x>
    variational_block(weights)
    # Controlled application of the unitary component A_l of the problem matrix A.
    CA(l)
    # Adjoint of the unitary U_b associated to the problem vector |b>.
    # In this specific example Adjoint(U_b) = U_b.
    U_b()
    if j != -1:
        CZ(control="q_anc", target=f"q{j}")
    # Unitary U_b associated to the problem vector |b>.
    U_b()
    # Controlled application of Adjoint(A_lp).
    # In this specific example Adjoint(A_lp) = A_lp.
    CA(lp)
    # Second Hadamard gate applied to the ancillary qubit.
    hadamard("q_anc")
    # Expectation value of Z for the ancillary qubit.

    return qml.expval(qml.PauliZ(wires=ancilla_idx))


with program() as VQLSprog:
    U_b(["q1", "q2", "q3"])

job = QM1.simulate(VQLSprog, SimulationConfig(1000))
samples = job.get_simulated_samples()
samples.con1.plot()
