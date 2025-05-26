"""
This script contains useful QUA macros for the two-qubit cross-entropy benchmarking use case.

Author: Arthur Strauss - Quantum Machines
Last updated: 2024-12-08
"""

from matplotlib import pyplot as plt
from qiskit.transpiler import CouplingMap
from qiskit.circuit import QuantumCircuit, QuantumRegister
from typing import List
from qm.qua import *
from qualang_tools.addons.variables import assign_variables_to_element
import numpy as np
from quam_libs.components import TransmonPair, Transmon
from scipy import optimize
from scipy.stats import stats
from itertools import combinations


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
        assign_variables_to_element(readout_elements[i].name, I[i], Q[i])
    return I, I_st, Q, Q_st


def reset_qubit(method: str, qubit: Transmon, **kwargs):
    """
    Macro to reset the qubit state.

    If method is 'cooldown', then the variable cooldown_time (in clock cycles) must be provided as a python integer > 4.

    **Example**: reset_qubit('cooldown', cooldown_times=500)

    If method is 'active', then 3 parameters are available as listed below.

    **Example**: reset_qubit('active', threshold=-0.003, max_tries=3)

    :param method: Method the reset the qubit state. Can be either 'cooldown' or 'active'.
    :param qubit: The qubit to be addressed in QuAM
    :key cooldown_time: qubit relaxation time in clock cycle, needed if method is 'cooldown'. Must be an integer > 4.
    :key threshold: threshold to discriminate between the ground and excited state, needed if method is 'active'.
    :key max_tries: python integer for the maximum number of tries used to perform active reset,
        needed if method is 'active'. Must be an integer > 0 and default value is 1.
    :key Ig: A QUA variable for the information in the `I` quadrature used for active reset. If not given, a new
        variable will be created. Must be of type `Fixed`.
    :key pi_pulse: The pulse to play to get back to the ground state. Default is 'x180'.
    :return:
    """
    if method == "cooldown":
        # Check cooldown_time
        cooldown_time = kwargs.get("cooldown_time", None)
        if (cooldown_time is None) or (cooldown_time < 4):
            raise Exception("'cooldown_time' must be an integer > 4 clock cycles")
        # Reset qubit state
        qubit.xy.wait(cooldown_time)
    elif method == "active":
        # Check threshold
        threshold = kwargs.get("threshold", None)
        if threshold is None:
            raise Exception("'threshold' must be specified for active reset.")
        # Check max_tries
        max_tries = kwargs.get("max_tries", 1)
        if (max_tries is None) or (not float(max_tries).is_integer()) or (max_tries < 1):
            raise Exception("'max_tries' must be an integer > 0.")
        # Check Ig
        Ig = kwargs.get("Ig", None)
        pi_pulse_name = kwargs.get("pi_pulse", "x180")
        # Reset qubit state
        return active_reset(threshold, qubit, max_tries=max_tries, Ig=Ig, pi_pulse=pi_pulse_name)


# Macro for performing active reset until successful for a given number of tries.
def active_reset(threshold: float, qubit: Transmon, max_tries=1, Ig=None, pi_pulse: str = "x180"):
    """Macro for performing active reset until successful for a given number of tries.

    :param threshold: threshold for the 'I' quadrature discriminating between ground and excited state.
    :param qubit: The qubit element. Must be defined in the config.
    :param resonator: The resonator element. Must be defined in the config.
    :param max_tries: python integer for the maximum number of tries used to perform active reset. Must >= 1.
    :param Ig: A QUA variable for the information in the `I` quadrature. Should be of type `Fixed`. If not given, a new
        variable will be created
    :param pi_pulse: The pulse to play to get back to the ground state. Default is 'x180'.
    :return: A QUA variable for the information in the `I` quadrature and the number of tries after success.
    """
    if Ig is None:
        Ig = declare(fixed)
    if (max_tries < 1) or (not float(max_tries).is_integer()):
        raise Exception("max_count must be an integer >= 1.")
    # Initialize Ig to be > threshold
    assign(Ig, threshold + 2**-28)
    # Number of tries for active reset
    counter = declare(int)
    # Reset the number of tries
    assign(counter, 0)

    # Perform active feedback
    qubit.xy.align(qubit.resonator.name)
    # Use a while loop and counter for other protocols and tests
    with while_((Ig > threshold) & (counter < max_tries)):
        # Measure the resonator
        qubit.resonator.measure("readout")
        # Play a pi pulse to get back to the ground state
        qubit.xy.play(pi_pulse, condition=(Ig > threshold))
        # Increment the number of tries
        assign(counter, counter + 1)
    return Ig, counter


def align_transmon(qubit: Transmon):
    """
    Macro to align all qubit drives with the associated resonator
    """
    qubit.xy.align(qubit.resonator.name, qubit.z.name)


def align_transmon_pair(qubit_pair: TransmonPair):
    """
    Macro to align all qubit drives with the associated resonators for a given qubit pair
    """
    all_channels = ["xy", "z", "resonator"]
    all_elements = []
    for qubit in [qubit_pair.qubit_control, qubit_pair.qubit_target]:
        for channel in all_channels:
            all_elements.append(getattr(qubit, channel).name)
    align(*all_elements)


def get_parallel_gate_combinations(coupling_map: CouplingMap, direction="forward"):
    """
    Returns all possible combinations of qubit pairs for which a two-qubit gate can be applied in parallel,
    respecting the specified direction constraint.

    Parameters:
    - coupling_map: Qiskit CouplingMap object that represents the qubit connectivity.
    - direction: 'forward' or 'reverse' to indicate which direction of qubit pairs should be selected.

    Returns:
    - List of combinations where the maximum number of two-qubit gates can be applied in parallel.
    """
    # Get all possible two-qubit gate pairs
    qubit_pairs = coupling_map.get_edges()

    # Create a set to store unique pairs in the specified direction
    filtered_pairs = set()
    for q1, q2 in qubit_pairs:
        if direction == "forward":
            # Add the pair if it's in the forward direction
            if (q2, q1) not in filtered_pairs:
                filtered_pairs.add((q1, q2))
        elif direction == "reverse":
            # Add the reversed pair if the original forward pair exists
            if (q1, q2) not in filtered_pairs:
                filtered_pairs.add((q2, q1))

    # Convert the set back to a list for further processing
    qubit_pairs = list(filtered_pairs)

    max_parallel_combinations = []
    max_num_parallel_gates = 0

    # Check all possible combinations of the qubit pairs
    for r in range(1, len(qubit_pairs) + 1):
        for combo in combinations(qubit_pairs, r):
            # Check if all pairs in the combination can be applied in parallel
            used_qubits = set()
            valid = True
            for pair in combo:
                if pair[0] in used_qubits or pair[1] in used_qubits:
                    valid = False
                    break
                used_qubits.update(pair)

            if valid:
                if len(combo) > max_num_parallel_gates:
                    max_num_parallel_gates = len(combo)
                    max_parallel_combinations = [combo]
                elif len(combo) == max_num_parallel_gates:
                    max_parallel_combinations.append(combo)

    return max_parallel_combinations


def generate_circuits(xeb_config, gate_indices: np.ndarray, available_combinations) -> List[List[QuantumCircuit]]:
    two_qubit_gate_pattern = 0
    n_qubits = xeb_config.n_qubits
    circuits = []
    if all([isinstance(qubit, Transmon) for qubit in xeb_config.qubits]):
        qubit_names = [qubit.name for qubit in xeb_config.qubits]
    else:
        qubit_names = xeb_config.qubits
    for s in range(xeb_config.seqs):
        circuits.append([])
        for d_, depth in enumerate(xeb_config.depths):
            q_regs = [QuantumRegister(1, qubit_name) for qubit_name in qubit_names]
            qc = QuantumCircuit(*q_regs)
            for d in range(depth):
                for q in range(n_qubits):
                    sq_gate = xeb_config.gate_set[gate_indices[s, q, d]].gate
                    qc.append(sq_gate, [q])
                qc.barrier()
                if xeb_config.two_qb_gate is not None:
                    for i, combination in enumerate(available_combinations):
                        if i == two_qubit_gate_pattern:
                            for pair in combination:
                                qc.append(xeb_config.two_qb_gate.gate, pair)
                            qc.barrier()
                            break
                    if two_qubit_gate_pattern == len(available_combinations) - 1:
                        two_qubit_gate_pattern = 0
                    else:
                        two_qubit_gate_pattern += 1

                    # qc.append(self.xeb_config.two_qb_gate.gate, [0, 1])
            qc.measure_all()
            circuits[s].append(qc)
            two_qubit_gate_pattern = 0
    return circuits


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


def compute_log_fidelity(incoherent_dist, expected_probs, measured_probs):
    """
    Compute the log fidelity between the expected and measured distributions.

    Parameters:
    - incoherent_dist: numpy array, the incoherent distribution
    - expected_probs: numpy array, the expected probabilities
    - measured_probs: numpy array, the measured probabilities

    Returns:
    - The log fidelity between the expected and measured distributions
    """
    # Compute the cross entropy between the incoherent distribution and the expected probabilities
    xe_incoherent = cross_entropy(incoherent_dist, expected_probs)
    xe_measured = cross_entropy(measured_probs, expected_probs)
    xe_expected = cross_entropy(expected_probs, expected_probs)

    f_xeb = (xe_incoherent - xe_measured) / (xe_incoherent - xe_expected)
    return f_xeb


def evaluate_log_fidelity(f_xeb, singularity, outlier, seq, depth):
    """
    Evaluate the log fidelity and return the corresponding value.
    """
    if np.isnan(f_xeb) or np.isinf(f_xeb):
        singularity.append((seq, depth))
        return np.nan
    elif f_xeb < 0 or f_xeb > 1:
        outlier.append((seq, depth))
        return np.nan
    return f_xeb


def update_record(records, seq, depth, expected_probs, measured_probs, dim):
    """
    Update the record to compute linear fidelities (Cirq like processing).
    """
    records += [
        {
            "sequence": seq,
            "depth": depth,
            "pure_probs": expected_probs,
            "measured_probs": measured_probs,
            "e_u": np.sum(expected_probs**2),
            "u_u": np.sum(expected_probs) / dim,
            "m_u": np.sum(measured_probs * expected_probs),
        }
    ]
    return records


def update_data_frame(df):
    """
    Update the data frame to compute linear fidelities (Cirq like processing).
    """
    try:
        df["y"] = df["m_u"] - df["u_u"]
        df["x"] = df["e_u"] - df["u_u"]
        df["numerator"] = df["x"] * df["y"]
        df["denominator"] = df["x"] ** 2
        return df

    except KeyError:
        raise ValueError("The records for linear XEB are empty. Please rerun the experiment.")


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
