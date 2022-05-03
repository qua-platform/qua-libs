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
from scipy.linalg import sqrtm

# Setting up the Gaussian waveform sample
gauss_pulse_len = 100  # nsec
Amp = 0.2  # Pulse Amplitude
gauss_arg = np.linspace(-3, 3, gauss_pulse_len)
gauss_wf = np.exp(-(gauss_arg**2) / 2)
gauss_wf = Amp * gauss_wf / np.max(gauss_wf)

# Setting up the configuration of the experimental setup
# Embedded in a Python dictionary
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
                "pi": "gauss_pulse_in",
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
my_qm = qmManager.open_qm(config)  # Generate a Quantum Machine based on the configuration described above
N_shots = 1000
qubit = "qubit"
rr = "RR"
n_input_states = 6


def H(q):  # QUA macro for applying a Hadamard gate
    frame_rotation(3 * np.pi / 2, q)
    play("pi" * amp(0.5), q)
    reset_frame(q)


def arb_gate(q, index):
    with switch_(index):
        with case_(0):  # X
            play("pi", q)
        with case_(1):  # Y
            frame_rotation(np.pi / 2, q)
            play("pi", q)
        with case_(2):  # X/2
            play("pi" * amp(0.5), q)
        with case_(3):  # Y/2
            frame_rotation(np.pi / 2, q)
            play("pi" * amp(0.5), q)
        with case_(4):  # -X/2
            play("pi" * amp(-0.5), q)
        with case_(5):  # -Y/2
            frame_rotation(np.pi / 2, q)
            play("pi" * amp(-0.5), q)
    reset_frame(q)


with program() as state_tomo:
    stream_state = declare_stream()
    I = declare(fixed)
    Q = declare(fixed)
    s = declare(int, value=3)
    n = declare(int)
    j = declare(int)
    th = declare(fixed, value=-5.57e-9)
    state = declare(bool)

    with for_(n, 0, n < N_shots, n + 1):
        with for_(j, 0, j < 6, j + 1):
            # X basis measurement
            arb_gate(qubit, j)
            H(qubit)
            align()
            measure("meas_pulse", rr, None, demod.full("integW1", I, "out1"))
            assign(state, I < th)
            save(state, stream_state)
            align()
            play("pi", qubit, condition=I < th)

            # Y measurement
            arb_gate(qubit, j)
            H(qubit)
            frame_rotation(np.pi / 2, qubit)
            align()
            measure("meas_pulse", rr, None, demod.full("integW1", I, "out1"))
            assign(state, I < th)
            save(state, stream_state)
            align()
            play("pi", qubit, condition=I < th)

            # Z basis measurement
            arb_gate(qubit, j)
            align()
            measure("meas_pulse", rr, None, demod.full("integW1", I, "out1"))
            assign(state, I < th)
            save(state, stream_state)
            align()
            play("pi", qubit, condition=I < th)

    with stream_processing():
        stream_state.boolean_to_int().buffer(n_input_states, 3).average().save("state")

job = qmManager.simulate(
    config, state_tomo, SimulationConfig(int(100000))
)  # Use LoopbackInterface to simulate the response of the qubit
time.sleep(1.0)

results = job.result_handles
results.wait_for_all_values()
P_1 = results.state.fetch_all()  # 2*P_1 - 1
R_dir_inv = []  # Bloch vectors for each input states
rho_dir_inv = []


def is_physical(
    R,
):  # Check if the reconstructed density matrix is physically valid or not.
    if np.linalg.norm(R) <= 1:
        return True
    else:
        return False


def norm(R):
    return np.linalg.norm(R)


for i in range(n_input_states):
    R_dir_inv.append(1 - 2 * P_1[i])
    rho_dir_inv.append(
        0.5
        * (
            np.array(([1.0, 0.0], [0.0, 1]))
            + R_dir_inv[i][0] * np.array(([0.0, 1.0], [1.0, 0.0]))
            + R_dir_inv[i][1] * np.array(([0.0, -1j], [1j, 0.0]))
            + R_dir_inv[i][2] * np.array(([1.0, 0.0], [0.0, -1.0]))
        )
    )


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
R_BME = []
rho_BME = []
for j in range(n_input_states):
    target = lambda x, y, z: L(
        x,
        y,
        z,
        N_shots * P_1[j][0],
        N_shots * (1 - P_1[j][0]),
        N_shots * P_1[j][1],
        N_shots * (1 - P_1[j][1]),
        N_shots * P_1[j][2],
        N_shots * (1 - P_1[j][2]),
    )

    r = np.array([0.0, 0.0, 0.0])
    niters = 10000
    burnin = 500
    sigma = np.diag([0.005, 0.005, 0.005])
    accepted = 0

    rs = np.zeros((niters - burnin, 3), float)
    for i in range(niters):
        new_r = stats.multivariate_normal(r, sigma).rvs()
        p = min(target(*new_r) / target(*r), 1)
        if np.random.rand() < p:
            r = new_r
            accepted += 1
        if i >= burnin:
            rs[i - burnin] = r

    # print("Efficiency: ", accepted / niters)
    R_BME.append(rs.mean(axis=0))
    # print("The reconstructed Bloch vector using Bayesian Mean Estimate is ", R_BME[j])
    # print(
    #     "Is the associated quantum state valid?",
    #     is_physical(R_BME[j]),
    #     ". Norm: ",
    #     norm(R_BME[j]),
    # )
    rho_BME.append(
        0.5
        * (
            np.array(([1.0, 0.0], [0.0, 1]))
            + R_BME[j][0] * np.array(([0.0, 1.0], [1.0, 0.0]))
            + R_BME[j][1] * np.array(([0.0, -1j], [1j, 0.0]))
            + R_BME[j][2] * np.array(([1.0, 0.0], [0.0, -1.0]))
        )
    )


# Constructing density matrices for target states
def fidelity(rho: np.ndarray, sigma: np.ndarray):
    return np.real(np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho))) ** 2)


ket0 = np.array([[1], [0]])
ket1 = np.array([[0], [1]])

th_output_states = [
    ket1,
    1j * ket1,
    1 / np.sqrt(2) * (ket0 - 1j * ket1),
    1 / np.sqrt(2) * (ket0 + ket1),
    1 / np.sqrt(2) * (ket0 + 1j * ket1),
    1 / np.sqrt(2) * (ket0 - ket1),
]

rho_th = [s_th @ s_th.conj().T for s_th in th_output_states]

for k in range(n_input_states):
    print(f"Theoretical DM {k}:", rho_th[k])
    print(f"Reconstructed DM {k} (DI)", rho_dir_inv[k])
    print(f"Fidelity {k} (DI):", fidelity(rho_th[k], rho_dir_inv[k]))
    # print(f"Is state {k} physically valid?", is_physical(R_dir_inv[k]), "Trace:", np.trace(rho_dir_inv[k]))
    print(f"Reconstructed DM {k} (BME) :", rho_BME[k])
    # print("Trace: ", np.trace(rho_BME[k]))
    print(f"Fidelity {k} (BME):", fidelity(rho_th[k], rho_BME[k]))
