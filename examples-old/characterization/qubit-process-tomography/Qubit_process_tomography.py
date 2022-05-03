"""
Qubit_process_tomography.py: Reconstruction of characteristic χ matrix for a superoperator applied on a single qubit
Author: Arthur Strauss - Quantum Machines
Created: 13/11/2020
Created on QUA version: 0.5.138
"""

# Importing the necessary from qm
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm.qua import math
from qm import LoopbackInterface
from qm import SimulationConfig
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import comb
import scipy.stats as stats
from configuration import *

π = np.pi
qmManager = QuantumMachinesManager()  # Reach OPX's IP address
qm = qmManager.open_qm(config)  # Generate a Quantum Machine based on the configuration described above
N_shots = 1  # Number of shots fixed to determine operator expectation values


# QUA macros (pulse definition of useful quantum gates)
def Hadamard(tgt):
    U2(tgt, 0, π)


def U2(tgt, 𝜙=0, 𝜆=0):
    Rz(𝜆, tgt)
    Y90(tgt)
    Rz(𝜙, tgt)


def U3(tgt, 𝜃=0, 𝜙=0, 𝜆=0):
    Rz(𝜆 - π / 2, tgt)
    X90(tgt)
    Rz(π - 𝜃, tgt)
    X90(tgt)
    Rz(𝜙 - π / 2, tgt)


def Rz(𝜆, tgt):
    frame_rotation(-𝜆, tgt)


def Rx(𝜆, tgt):
    U3(tgt, 𝜆, -π / 2, π / 2)


def Ry(𝜆, tgt):
    U3(tgt, 𝜆, 0, 0)


def X90(tgt):
    play("X90", tgt)


def Y90(tgt):
    play("Y90", tgt)


def Y180(tgt):
    play("Y180", tgt)


def Arbitrary_process(
    tgt,
):  # QUA macro for applying arbitrary process, here a X rotation of angle π/4, followed by a Y rotation of π/2
    Rx(π / 4, tgt)
    Ry(π / 2, tgt)


def state_saving(I, Q, state_estimate, stream):  # Do state estimation protocol in QUA, and save the associated state
    # Define coef a & b defining the line separating states 0 & 1 in the IQ Plane (calibration required), here a & b are arbitrary
    a = declare(fixed, value=1.0)
    b = declare(fixed, value=1.0)
    with if_(Q - a * I - b > 0):
        assign(state_estimate, 1)
    with else_():
        assign(state_estimate, 0)
    save(state_estimate, stream)


def measure_and_reset_state(tgt, RR, I, Q, A, stream_A):
    measure("meas_pulse", RR, None, ("integW1", I), ("integW2", Q))
    state_saving(I, Q, A, stream_A)
    wait(
        t1, tgt
    )  # Wait for relaxation of the qubit after the collapse of the wavefunction in case of collapsing into |1> state


def Z_tomography(tgt, RR, Iz, Qz, Z, stream_Z):
    # Generate an arbitrary process to characterize
    Arbitrary_process(tgt)
    # Begin tomography_process
    measure_and_reset_state(tgt, RR, Iz, Qz, Z, stream_Z)


def X_tomography(tgt, RR, Ix, Qx, X, stream_X):
    Arbitrary_process(tgt)
    Hadamard(tgt)
    measure_and_reset_state(tgt, RR, Ix, Qx, X, stream_X)


def Y_tomography(tgt, RR, Iy, Qy, Y, stream_Y):
    Arbitrary_process(tgt)
    Hadamard(tgt)
    frame_rotation(π / 2, "qubit")  # S-gate
    measure_and_reset_state(tgt, RR, Iy, Qy, Y, stream_Y)


with program() as process_tomography:
    stream_Z = declare_stream()
    stream_Y = declare_stream()
    stream_X = declare_stream()

    j = declare(int)  # Define necessary QUA variables to store the result of the experiments
    Iz = declare(fixed)
    Qz = declare(fixed)
    Z = declare(fixed)
    Ix = declare(fixed)
    Qx = declare(fixed)
    X = declare(fixed)
    Iy = declare(fixed)
    Qy = declare(fixed)
    Y = declare(fixed)
    t1 = declare(int, value=10)  # Assume we know the value of the relaxation time allowing to return to 0 state
    with for_(j, 0, j < N_shots, j + 1):
        # Preparing state |0>, i.e do nothing else than tomography:
        Z_tomography("qubit", "RR", Iz, Qz, Z, stream_Z)
        Y_tomography("qubit", "RR", Iy, Qy, Y, stream_Y)
        X_tomography("qubit", "RR", Ix, Qx, X, stream_X)

        # Preparing state |1>, apply a rotation of π around X axis
        Rx(π, "qubit")
        Z_tomography("qubit", "RR", Iz, Qz, Z, stream_Z)
        Y_tomography("qubit", "RR", Iy, Qy, Y, stream_Y)
        X_tomography("qubit", "RR", Ix, Qx, X, stream_X)

        # Preparing |+> state, apply Hadamard
        Hadamard("qubit")
        Z_tomography("qubit", "RR", Iz, Qz, Z, stream_Z)
        Y_tomography("qubit", "RR", Iy, Qy, Y, stream_Y)
        X_tomography("qubit", "RR", Ix, Qx, X, stream_X)

        # Preparing |-> state, apply Hadamard then S gate
        Hadamard("qubit")
        frame_rotation(-π / 2, "qubit")
        Z_tomography("qubit", "RR", Iz, Qz, Z, stream_Z)
        Y_tomography("qubit", "RR", Iy, Qy, Y, stream_Y)
        X_tomography("qubit", "RR", Ix, Qx, X, stream_X)

    with stream_processing():
        stream_Z.save_all("Z")
        stream_X.save_all("X")
        stream_Y.save_all("Y")

job = qmManager.simulate(
    config,
    process_tomography,
    SimulationConfig(int(50000), simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)])),
)  # Use LoopbackInterface to simulate the response of the qubit
time.sleep(1.0)

# Retrieving all results
my_tomography_results = job.result_handles
X = my_tomography_results.X.fetch_all()["value"]
Y = my_tomography_results.Y.fetch_all()["value"]
Z = my_tomography_results.Z.fetch_all()["value"]

# Using direct inversion for state tomography on each of the 4 prepared states
state = np.array([[None, None, None]] * 4)  # Store results associated to each of the 4 prepared states
counts_1 = np.array(
    [[None, None, None]] * 4
)  # Store number of 1s measured for each axis (X,Y,Z) for each of the 4 prepared states
counts_0 = np.array([[None, None, None]] * 4)  # Same for 0s
R_dir_inv = np.array([[0, 0, 0]] * 4)  # Bloch vectors for each of the 4 prepared states
rho_div_inv = np.array([None] * 4)  # Density matrices associated to the 4 states obtained after applying the process
ρ = np.array([None] * 4)  # Store matrices described in eq 8.173-8.176 of Box 8.5 in Nielsen & Chuang

for i in range(4):
    state[i] = [X[i::4], Y[i::4], Z[i::4]]  # Isolate results for |0>,|1>, |+> and |->
    counts_1[i] = [
        np.count_nonzero(state[i][0]),
        np.count_nonzero(state[i][1]),
        np.count_nonzero(state[i][2]),
    ]
    counts_0[i] = [
        N_shots - counts_1[i][0],
        N_shots - counts_1[i][1],
        N_shots - counts_1[i][2],
    ]
    R_dir_inv[i] = (1 / N_shots) * np.array(
        [
            (counts_0[i][0] - counts_1[i][0]),
            (counts_0[i][1] - counts_1[i][1]),
            (counts_0[i][2] - counts_1[i][2]),
        ]
    )
    rho_div_inv[i] = 0.5 * (
        np.array([[1.0, 0.0], [0.0, 1]])
        + R_dir_inv[i][0] * np.array([[0.0, 1.0], [1.0, 0.0]])
        + R_dir_inv[i][1] * np.array([[0.0, -1j], [1j, 0.0]])
        + R_dir_inv[i][2] * np.array([[1.0, 0.0], [0.0, -1.0]])
    )

ρ[0] = rho_div_inv[0]
ρ[3] = rho_div_inv[1]
ρ[1] = rho_div_inv[2] - 1j * rho_div_inv[3] - ((1 - 1j) / 2) * (rho_div_inv[0] + rho_div_inv[1])
ρ[2] = rho_div_inv[2] + 1j * rho_div_inv[3] - ((1 + 1j) / 2) * (rho_div_inv[0] + rho_div_inv[1])

Λ = 0.5 * np.array(  # Build the Λ matrix as described in eq 8.178 of Box 8.5 of Nielsen & Chuang
    [[1, 0, 0, 1], [0, 1, 1, 0], [0, 1, -1, 0], [1, 0, 0, -1]]
)

R = np.array(
    [  ##Build the "super" density matrix as shown in eq 8.179 in Box 8.5 of the book of Nielsen & Chuang
        [ρ[0][0][0], ρ[0][0][1], ρ[1][0][0], ρ[1][0][1]],
        [ρ[0][1][0], ρ[0][1][1], ρ[1][1][0], ρ[1][1][1]],
        [ρ[2][0][0], ρ[2][0][1], ρ[3][0][0], ρ[3][0][1]],
        [ρ[2][1][0], ρ[2][1][1], ρ[3][1][0], ρ[3][1][1]],
    ]
)

χ = Λ @ R @ Λ

print("Reconstruction of χ-matrix using direct inversion state tomography : ", χ)


def is_physical(
    R,
):  # Check if the reconstructed density matrix is physically valid or not.
    if np.linalg.norm(R) <= 1:
        return True
    else:
        return False


def norm(R):
    return np.linalg.norm(R)


# Might be False when considering direct inversion method.


# Bayesian Mean Estimate


def C(r):
    """
    This function implements a homogeneous prior.
    """
    return np.where(np.linalg.norm(r, axis=0) < 1, 1, 0)


def P(x, y, z, Nx1, Nx0, Ny1, Ny0, Nz1, Nz0):
    """
    Fill in here the probability of measuring the results Nxu, Nxd, Nyu, Nyd, Nzu, Nzd given a density matrix defined
    by the Bloch vector r = (x, y, z), which is a Binomial law for each axis
    """
    px = comb(Nx0 + Nx1, Nx1) * ((1 + x) * 0.5) ** Nx1 * ((1 - x) * 0.5) ** Nx0
    py = comb(Ny0 + Ny1, Ny1) * ((1 + y) * 0.5) ** Ny1 * ((1 - y) * 0.5) ** Ny0
    pz = comb(Nz0 + Nz1, Nz1) * ((1 + z) * 0.5) ** Nz1 * ((1 - z) * 0.5) ** Nz0

    return px * py * pz


def L(x, y, z, Nx1, Nx0, Ny1, Ny0, Nz1, Nz0):
    """
    Implement here the likelihood
    """
    return C([x, y, z]) * P(x, y, z, Nx1, Nx0, Ny1, Ny0, Nz1, Nz0)


"""
Implement here a Metropolis-Hasings algorithm to efficiently evaluate the Baysian mean integral. 
Help can be found here https://people.duke.edu/~ccc14/sta-663/MCMC.html and here
https://en.wikipedia.org/wiki/Monte_Carlo_integration

You can also look at the following paper: 
Blume-Kohout, Robin. "Optimal, reliable estimation of quantum states." New Journal of Physics 12.4 (2010): 043034.

Make sure that the efficiency of the algorithm is about 30%
"""


def BME_Bloch_vec(Nx1, Nx0, Ny1, Ny0, Nz1, Nz0):
    target = lambda x, y, z: L(x, y, z, Nx1, Nx0, Ny1, Ny0, Nz1, Nz0)

    r = np.array([0.0, 0.0, 0.0])
    niters = 10000
    burnin = 500
    sigma = np.diag([0.005, 0.005, 0.005])
    accepted = 0

    rs = np.zeros((niters - burnin, 3), np.float)
    for i in range(niters):
        new_r = stats.multivariate_normal(r, sigma).rvs()
        p = min(target(*new_r) / target(*r), 1)
        if np.random.rand() < p:
            r = new_r
            accepted += 1
        if i >= burnin:
            rs[i - burnin] = r

    print("Efficiency: ", accepted / niters)
    r_BME = rs.mean(axis=0)
    return r_BME


R_BME = np.array([[0, 0, 0]] * 4)  # Bloch vector reconstruction
rho_BME = np.array([None] * 4)
ρ_BME = np.array([None] * 4)
for i in range(4):
    R_BME[i] = BME_Bloch_vec(
        counts_1[i][0],
        counts_0[i][0],
        counts_1[i][1],
        counts_0[i][1],
        counts_1[i][2],
        counts_0[i][2],
    )
    rho_BME[i] = 0.5 * (
        np.array([[1.0, 0.0], [0.0, 1]])
        + R_dir_inv[i][0] * np.array([[0.0, 1.0], [1.0, 0.0]])
        + R_dir_inv[i][1] * np.array([[0.0, -1j], [1j, 0.0]])
        + R_dir_inv[i][2] * np.array([[1.0, 0.0], [0.0, -1.0]])
    )

ρ_BME[0] = rho_BME[0]
ρ_BME[3] = rho_BME[1]
ρ_BME[1] = rho_BME[2] - 1j * rho_BME[3] - ((1 - 1j) / 2) * (rho_BME[0] + rho_BME[1])
ρ_BME[2] = rho_BME[2] + 1j * rho_BME[3] - ((1 + 1j) / 2) * (rho_BME[0] + rho_BME[1])

R2 = np.array(
    [
        [ρ_BME[0][0][0], ρ_BME[0][0][1], ρ_BME[1][0][0], ρ_BME[1][0][1]],
        [ρ_BME[0][1][0], ρ_BME[0][1][1], ρ_BME[1][1][0], ρ_BME[1][1][1]],
        [ρ_BME[2][0][0], ρ_BME[2][0][1], ρ_BME[3][0][0], ρ_BME[3][0][1]],
        [ρ_BME[2][1][0], ρ_BME[2][1][1], ρ_BME[3][1][0], ρ_BME[3][1][1]],
    ]
)
χ_BME = Λ @ R2 @ Λ
print("Reconstruction of χ-matrix using Bayesian Mean Estimation tomography : ", χ_BME)
