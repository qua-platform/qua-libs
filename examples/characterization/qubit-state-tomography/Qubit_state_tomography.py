"""
qubit-state-tomography.py: Qubit state Bloch vector reconstruction
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

# Setting up the Gaussian waveform sample
gauss_pulse_len = 100  # nsec
Amp = 0.2  # Pulse Amplitude
gauss_arg = np.linspace(-3, 3, gauss_pulse_len)
gauss_wf = np.exp(-(gauss_arg ** 2) / 2)
gauss_wf = Amp * gauss_wf / np.max(gauss_wf)
## Setting up the configuration of the experimental setup
## Embedded in a Python dictionary
config = {
    "version": 1,
    "controllers": {  # Define the QM device and its outputs, in this case:
        "con1": {  # 2 analog outputs for the in-phase and out-of phase components
            "type": "opx1",  # of the qubit (I & Q), and 2 other analog outputs for the coupled readout resonator
            "analog_outputs": {
                1: {"offset": 0.032},
                2: {"offset": 0.041},
                3: {"offset": -0.024},
                4: {"offset": 0.115},
            },
            "analog_inputs": {
                1: {"offset": +0.0},
            },
        }
    },
    "elements": {  # Define the elements composing the quantum system, i.e the qubit+ readout resonator (RR)
        "qubit": {
            "mixInputs": {
                "I": ("con1", 1),  # Connect the component to one output of the OPX
                "Q": ("con1", 2),
                "lo_frequency": 5.10e7,
                "mixer": "mixer_qubit",  ##Associate a mixer entity to control the IQ mixing process
            },
            "intermediate_frequency": 5.15e7,  # Resonant frequency of the qubit
            "operations": {  # Define the set of operations doable on the qubit, each operation is related
                "gauss_pulse": "gauss_pulse_in",  # to a pulse
                "Arbitrary_Op": "state_generation_pulse",
                "Hadamard_Op": "Hadamard_pulse",
            },
        },
        "RR": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": 6.00e7,
                "mixer": "mixer_res",
            },
            "intermediate_frequency": 6.12e7,
            "operations": {
                "meas_pulse": "meas_pulse_in",
            },
            "time_of_flight": 180,  # Measurement parameters
            "smearing": 0,
            "outputs": {"out1": ("con1", 1)},
        },
    },
    "pulses": {  # Pulses definition
        "meas_pulse_in": {  # Readout pulse
            "operation": "measurement",
            "length": 200,
            "waveforms": {
                "I": "exc_wf",  # Decide what pulse to apply for each component
                "Q": "zero_wf",
            },
            "integration_weights": {
                "integW1": "integW1",
                "integW2": "integW2",
            },
            "digital_marker": "marker1",
        },
        "gauss_pulse_in": {  # Standard Gaussian pulse
            "operation": "control",
            "length": 100,
            "waveforms": {"I": "gauss_wf", "Q": "zero_wf"},
        },
        "state_generation_pulse": {
            # Pulse generating the arbitrary state, to be redefined according to the desired state
            "operation": "control",  # Could also be a sequence of pulses, cf function Arbitrary_state_generation()
            "length": 100,
            "waveforms": {"I": "gauss_wf", "Q": "zero_wf"},
        },
        "Hadamard_pulse": {
            # To be redefined according to the hardware (function Hadamard could actually be a sequence of pulses)
            "operation": "control",
            "length": 100,
            "waveforms": {"I": "gauss_wf", "Q": "zero_wf"},
        },
    },
    "waveforms": {  # Specify the envelope type of the pulses defined above
        "zero_wf": {"type": "constant", "sample": 0.0},
        "exc_wf": {"type": "constant", "sample": 0.479},
        "gauss_wf": {"type": "arbitrary", "samples": gauss_wf.tolist()},
    },
    "digital_waveforms": {"marker1": {"samples": [(1, 4), (0, 2), (1, 1), (1, 0)]}},
    "integration_weights": {  # Define integration weights for measurement demodulation
        "integW1": {
            "cosine": [
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
            ],
            "sine": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        },
        "integW2": {
            "cosine": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "sine": [
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
            ],
        },
    },
    "mixers": {  # Potential corrections to be brought related to the IQ mixing scheme
        "mixer_res": [
            {
                "intermediate_frequency": 6.12e7,
                "lo_frequency": 6.00e7,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
        ],
        "mixer_qubit": [
            {
                "intermediate_frequency": 5.15e7,
                "lo_frequency": 5.10e7,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
        ],
    },
}
qmManager = QuantumMachinesManager()  # Reach OPX's IP address
my_qm = qmManager.open_qm(
    config
)  # Generate a Quantum Machine based on the configuration described above
N_shots = 10


def Arbitrary_state_generation(
    tgt,
):  # QUA macro for generating input state we want to sample out
    play("Arbitrary_Op", tgt)


def Hadamard(tgt):  # QUA macro for applying a Hadamard gate
    play("Hadamard_Op", tgt)


def state_saving(
    I, Q, state_estimate, stream
):  # Do state estimation protocol in QUA, and save the associated state
    # Define coef a & b defining the line separating states 0 & 1 in the IQ Plane (calibration required), here a & b are arbitrary
    a = declare(fixed, value=1.0)
    b = declare(fixed, value=1.0)
    with if_(Q - a * I - b > 0):
        assign(state_estimate, 1)
    with else_():
        assign(state_estimate, 0)
    save(state_estimate, stream)


def do_tomography():
    with program() as tomography:
        stream_Ix = (
            declare_stream()
        )  # Open streams to allow data retrieval for post-processing
        stream_Qx = declare_stream()
        stream_Iy = declare_stream()
        stream_Qy = declare_stream()
        stream_Iz = declare_stream()
        stream_Qz = declare_stream()
        stream_Z = declare_stream()
        stream_Y = declare_stream()
        stream_X = declare_stream()

        j = declare(
            int
        )  # Define necessary QUA variables to store the result of the experiments
        Iz = declare(fixed)
        Qz = declare(fixed)
        Z = declare(fixed)
        Ix = declare(fixed)
        Qx = declare(fixed)
        X = declare(fixed)
        Iy = declare(fixed)
        Qy = declare(fixed)
        Y = declare(fixed)
        t1 = declare(
            int, value=10
        )  # Assume we know the value of the relaxation time allowing to return to 0 state
        with for_(j, 0, j < N_shots, j + 1):
            # Generate an arbitrary quantum state, e.g fully superposed state |+>=(|0>+|1>)/sqrt(2)
            Arbitrary_state_generation("qubit")
            # Begin tomography_process
            # Start with Pauli-Z expectation value determination : getting statistics of state measurement is enough to calculate it
            measure("meas_pulse", "RR", None, ("integW1", Iz), ("integW2", Qz))
            save(Iz, stream_Iz)  # Save the results
            save(Qz, stream_Qz)
            state_saving(Iz, Qz, Z, stream_Z)
            wait(
                t1, "qubit"
            )  # Wait for relaxation of the qubit after the collapse of the wavefunction in case of collapsing into |1> state

            # Repeat sequence for X axis
            # Generate an arbitrary quantum state, e.g fully superposed state |+>=(|0>+|1>)/sqrt(2)
            Arbitrary_state_generation("qubit")
            # Begin tomography_process
            # Determine here Pauli X-expectation value, which corresponds to applying a Hadamard gate before measurement (unitary transformation)
            Hadamard("qubit")
            measure("meas_pulse", "RR", "samples", ("integW1", Ix), ("integW2", Qx))
            save(Ix, stream_Ix)  # Save the results
            save(Qx, stream_Qx)
            state_saving(Ix, Qx, X, stream_X)
            wait(
                t1, "qubit"
            )  # Wait for relaxation of the qubit after the collapse of the wavefunction in case of collapsing into |1> state
            # Could also do active reset

            # Repeat for Y axis
            # Generate an arbitrary quantum state, e.g fully superposed state |+>=(|0>+|1>)/sqrt(2)
            Arbitrary_state_generation("qubit")
            # Begin tomography_process
            # Determine here Pauli Y-expectation value, which corresponds to applying a Hadamard gate then S-gate before measurement (unitary transformation)
            Hadamard("qubit")
            frame_rotation(np.pi / 2, "qubit")  # S-gate
            measure("meas_pulse", "RR", "samples", ("integW1", Iy), ("integW2", Qy))
            save(Iy, stream_Iy)  # Save the results
            save(Qy, stream_Qy)
            state_saving(Iy, Qy, Y, stream_Y)
            wait(
                t1, "qubit"
            )  # Wait for relaxation of the qubit after the collapse of the wavefunction in case of collapsing into |1> state
        with stream_processing():
            stream_Iz.save_all("Iz_raw")
            stream_Qz.save_all("Qz_raw")
            stream_Z.save_all("Z")
            stream_Ix.save_all("Ix_raw")
            stream_Qx.save_all("Qx_raw")
            stream_X.save_all("X")
            stream_Iy.save_all("Iy_raw")
            stream_Qy.save_all("Qy_raw")
            stream_Y.save_all("Y")

    my_job = my_qm.simulate(
        tomography,
        SimulationConfig(
            int(50000), simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)])
        ),
    )  # Use LoopbackInterface to simulate the response of the qubit
    time.sleep(1.0)
    return my_job


# Retrieving all results
my_tomography_results = do_tomography().result_handles

Ix = my_tomography_results.Ix_raw.fetch_all()["value"]
Qx = my_tomography_results.Qx_raw.fetch_all()["value"]
X = my_tomography_results.X.fetch_all()["value"]

Iy = my_tomography_results.Iy_raw.fetch_all()["value"]
Qy = my_tomography_results.Qy_raw.fetch_all()["value"]
Y = my_tomography_results.Y.fetch_all()["value"]

Iz = my_tomography_results.Iz_raw.fetch_all()["value"]
Qz = my_tomography_results.Qz_raw.fetch_all()["value"]
Z = my_tomography_results.Z.fetch_all()["value"]

total_counts = [len(X), len(Y), len(Z)]
counts_1 = [
    np.count_nonzero(X),
    np.count_nonzero(Y),
    np.count_nonzero(Z),
]  # Count the number of measurements of the |1> state for each axis
counts_0 = [
    total_counts[0] - counts_1[0],
    total_counts[1] - counts_1[1],
    total_counts[2] - counts_1[2],
]  # Same for |0>

# Perform direct inversion protocol

R_dir_inv = [0, 0, 0]  # Bloch vector reconstruction
for i in range(3):
    R_dir_inv[i] = (counts_1[i] - counts_0[i]) / total_counts[i]


def is_physical(
    R,
):  # Check if the reconstructed density matrix is physically valid or not.
    if np.linalg.norm(R) <= 1:
        return True
    else:
        return False


def norm(R):
    return np.linalg.norm(R)


print("The reconstructed Bloch vector using direct inversion is ", R_dir_inv)
print(
    "Is the associated quantum state valid?",
    is_physical(R_dir_inv),
    ". Norm: ",
    norm(R_dir_inv),
)

rho_div_inv = 0.5 * (
    np.array(([1.0, 0.0], [0.0, 1]))
    + R_dir_inv[0] * np.array(([0.0, 1.0], [1.0, 0.0]))
    + R_dir_inv[1] * np.array(([0.0, -1j], [1j, 0.0]))
    + R_dir_inv[2] * np.array(([1.0, 0.0], [0.0, -1.0]))
)
print(" Reconstructed density matrix using direct inversion : ", rho_div_inv)
print("Trace: ", np.trace(rho_div_inv))
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

target = lambda x, y, z: L(
    x,
    y,
    z,
    counts_1[0],
    counts_0[0],
    counts_1[1],
    counts_0[1],
    counts_1[2],
    counts_0[2],
)


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
print("The reconstructed Bloch vector using Bayesian Mean Estimate is ", r_BME)
print(
    "Is the associated quantum state valid?",
    is_physical(r_BME),
    ". Norm: ",
    norm(r_BME),
)
rho_BME = 0.5 * (
    np.array(([1.0, 0.0], [0.0, 1]))
    + r_BME[0] * np.array(([0.0, 1.0], [1.0, 0.0]))
    + r_BME[1] * np.array(([0.0, -1j], [1j, 0.0]))
    + r_BME[2] * np.array(([1.0, 0.0], [0.0, -1.0]))
)
print(" Reconstructed density matrix using direct inversion : ", rho_BME)
print("Trace: ", np.trace(rho_BME))
