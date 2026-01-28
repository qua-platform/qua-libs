"""
        PURITY RANDOMIZED BENCHMARKING (for gates >= 40ns)

Purity Randomized Benchmarking (also known as Unitarity RB) characterizes gate noise by measuring
the unitarity - a value between 0 and 1 that indicates how coherent the noise is:
    - Unitarity = 1: Purely coherent errors (calibration issues, over/under rotations)
    - Unitarity = 0: Purely incoherent errors (depolarization, decoherence)

Based on Wallman et al. "Estimating the Coherence of Noise" (arXiv:1503.07865).

Protocol Overview:
    The program plays random sequences of Clifford gates WITHOUT a recovery gate (unlike standard RB).
    For each sequence at each depth, all three Pauli operators (X, Y, Z) are measured to compute the
    shifted purity P = <X>^2 + <Y>^2 + <Z>^2, which measures the squared length of the Bloch vector.

    The purity decays as: E[P] = A * u^(m-1) + B
    where 'u' is the unitarity (decay constant), 'm' is the sequence length, and A, B account for SPAM errors.

Key Differences from Standard RB (16a_randomized_benchmarking.py):
    - No recovery gate: Standard RB requires it, Purity RB does NOT
    - Measurement: Standard RB measures Z only (survival probability), Purity RB measures X, Y, Z
    - Output metric: Standard RB gives sequence fidelity, Purity RB gives shifted purity
    - Decay fit: Standard RB extracts fidelity 'p', Purity RB extracts unitarity 'u'

Output Metrics:
    - Unitarity (u): Decay constant in [0, 1] indicating coherence of noise
    - Lower bound on optimal infidelity: R >= (d-1)/d * (1 - sqrt(u)) per Wallman Eq. 46

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - Having the qubit frequency perfectly calibrated (ramsey).
    - Having calibrated the readout for state discrimination (rotated blobs and threshold).
    - Having calibrated ge_threshold and rotation_angle for accurate state assignment.
    - Set the desired flux bias.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from macros import readout_macro
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Number of random sequences
num_of_sequences = 50
# Number of averaging loops for each random sequence
n_avg = 20
# Maximum circuit depth
max_circuit_depth = 1000
# Play each sequence with a depth step equal to 'delta_clifford' - Must be > 0
delta_clifford = 10
assert (max_circuit_depth / delta_clifford).is_integer(), "max_circuit_depth / delta_clifford must be an integer."
# Pseudo-random number generator seed
seed = 345324

# Data to save
save_data_dict = {
    "num_of_sequences": num_of_sequences,
    "n_avg": n_avg,
    "max_circuit_depth": max_circuit_depth,
    "delta_clifford": delta_clifford,
    "config": config,
}
