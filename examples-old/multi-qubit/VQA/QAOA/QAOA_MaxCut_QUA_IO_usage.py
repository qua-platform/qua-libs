"""QAOA_MaxCut_QUA_IO_usage.py: Implementation of QAOA for solving a MaxCut problem instance with real-time capabilities
Author: Arthur Strauss - Quantum Machines
Created: 06/10/2021
This script must run on QUA 0.9 or higher
"""

from Config_QAOA import *
import networkx as nx
from typing import List
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from qm.QuantumMachinesManager import QuantumMachinesManager
import time
import random as rand

# QM configuration based on IBM Yorktown Quantum Computer:
# https://github.com/Qiskit/ibmq-device-information/tree/master/backends/yorktown/V1

qmm = QuantumMachinesManager()
qm = qmm.open_qm(IBMconfig)

# Problem definition for QAOA: MaxCut instance on a graph with 4 nodes

# Generating the graph using networkx package
n = 4  # Number of nodes
V = np.arange(0, n, 1)
E = [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (3, 2, 1.0), (0, 3, 1.0), (1, 3, 1.0)]

G = nx.Graph()
G.add_nodes_from(V)
G.add_weighted_edges_from(E)
E = G.edges()

# Drawing the Graph G
colors = ["b" for node in G.nodes()]
default_axes = plt.axes(frameon=True)
pos = nx.spring_layout(G)

nx.draw_networkx(G, node_color=colors, node_size=600, alpha=1, ax=default_axes, pos=pos)

# Algorithm hyperparameters
p = 1  # depth of quantum circuit, i.e number of adiabatic evolution blocks
N_shots = 10  # Sampling number for expectation value computation
MaxCut_value = 5.0  # Value of the ideal MaxCut (set to None if unknown) to calculate approximation ratio

th = 0.0  # Threshold for state discrimination

# Introduce registers to address relevant quantum elements in config
q = [f"q{i}" for i in range(n)]
rr = [f"rr{i}" for i in range(n)]

# Initialize angles to random to launch the program
qm.set_io1_value(np.random.uniform(0, 2 * π))
qm.set_io2_value(np.random.uniform(0, π))

""" 
We use here multiple global variables, callable in every function : 
- G: graph instance modelling the MaxCut problem we're solving
- n:  number of nodes in the graph, which is also the number of qubits
- N_shots: sampling number for expectation value computation
- qm: Quantum Machine instance opened with a specific hardware configuration
- p:  number of adiabatic evolution blocks, i.e defines the depth of the quantum circuit preparing QAOA ansatz state

This script relies on built in functions.
By order of calling when using the full algorithm we have the following tree:
1. result_optimization, main organ of whole QAOA outline, returns the result of the algorithm

2. quantum_avg_computation, returns the expectation value of cost Hamiltonian
      for a specific parametrized quantum circuit ((γ,β) angles fixed),
      measurement sampling for statistics given by parameter N_shots)

3. QAOA, the QUA program 
"""

with program() as QAOA:
    rep, b = declare(int), declare(int)
    I, Q = declare(fixed), declare(fixed)
    state = declare(int, size=n)
    γ, β = declare(fixed, size=p), declare(fixed, size=p)
    Cut, Expectation_value = declare(fixed, value=0), declare(fixed, value=0.0)
    w = declare(fixed)
    # waiting = declare(bool, value=True)
    with infinite_loop_():
        pause()
        # assign(waiting, True)
        # with while_(waiting):
        #     assign(waiting, IO1)

        # Load circuit parameters using IO variables
        with for_(b, init=0, cond=b < p, update=b + 1):
            pause()
            assign(γ[b], IO1)
            assign(β[b], IO2)

        # Start running quantum circuit (repeated N_shots times)
        with for_(rep, init=0, cond=rep <= N_shots, update=rep + 1):

            for k in range(n):  # for each qubit in the config (= node in the graph G)
                Hadamard(q[k])

            with for_(b, init=0, cond=b < p, update=b + 1):  # for each block of the QAOA quantum circuit

                # Cost Hamiltonian evolution: derived here specifically for MaxCut problem
                for e in E:  # for each edge of the graph G
                    e1, e2 = int(e[0]), int(e[1])
                    ctrl, tgt = q[e1], q[e2]
                    CU1(2 * γ[b], ctrl, tgt)
                    Rz(-γ[b], ctrl)
                    Rz(-γ[b], tgt)

                align()

                # Mixing Hamiltonian evolution
                for k in range(n):  # Apply X(β) rotation on each qubit
                    Rx(-2 * β[b], q[k])

            # Measurement and state determination
            for k in range(n):  # Iterate over each resonator
                measure("meas_pulse", rr[k], None, integration.full("integW1", I))
                assign(state[k], Cast.to_int(I > th))

                # Active reset: flip the qubit (X pulse) if measured state is 1
                with if_(Cast.to_bool(state[k])):
                    Rx(π, q[k])

            # Cost function evaluation
            for e in E:  # for each edge of the graph G
                e1, e2 = int(e[0]), int(e[1])
                assign(w, G[e1][e2]["weight"])  # Retrieve weight associated to edge e
                assign(
                    Cut,
                    Cut
                    + w
                    * (
                        Cast.to_fixed(state[e1]) * (1.0 - Cast.to_fixed(state[e2]))
                        + Cast.to_fixed(state[e2]) * (1.0 - Cast.to_fixed(state[e1]))
                    ),
                )
            assign(Expectation_value, Expectation_value + Cut)
        assign(Expectation_value, Math.div(Expectation_value, N_shots))
        assign(IO1, Expectation_value)
        # save(Expectation_value, "exp_value")


# Main Python functions for launching QAOA algorithm


def encode_angles_in_IO(gamma: List[float], beta: List[float]):
    """
    Insert parameter sets in the program with IO variables
    """
    if len(gamma) != len(beta):
        raise IndexError("Angle sets are not matching desired circuit depth")
    for i in range(len(gamma)):
        while not (job.is_paused()):
            time.sleep(0.01)
        qm.set_io1_value(gamma[i])
        qm.set_io2_value(beta[i])
        job.resume()


def quantum_avg_computation(angles: List[float]):
    """
    Calculate Hamiltonian expectation value (cost function to optimize)
    """

    gamma = angles[0 : 2 * p : 2]
    beta = angles[1 : 2 * p : 2]

    # qm.set_io1_value(True)
    job.resume()

    encode_angles_in_IO(gamma, beta)

    # Wait for experiment to be completed
    while not (job.is_paused()):
        time.sleep(0.0001)

    exp_value = qm.get_io1_value()["fixed_value"]
    # results = job.result_handles
    # exp_value = results.exp_value.fetch_all()

    return -exp_value  # minus sign because optimizer finds the minimum


def SPSA_optimize(init_angles, boundaries, max_iter=100):
    """
    Use SPSA optimization scheme, source : https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF
    """

    def SPSA_calibration():
        return 0.6283185307179586, 1e-1, 0, 0.602, 0.101

    a, c, A, alpha1, gamma = SPSA_calibration()
    angles = init_angles

    for j in range(1, max_iter):
        a_k = a / (j + A) ** alpha1
        c_k = c / j**gamma
        # Vector of random variables issued from a Bernoulli distribution +1,-1, could be something else
        delta_k = 2 * np.round(np.random.uniform(0, 1, 2 * p)) - 1
        angles_plus = angles + c_k * delta_k
        angles_minus = angles - c_k * delta_k
        cost_plus = quantum_avg_computation(angles_plus)
        cost_minus = quantum_avg_computation(angles_minus)
        gradient_est = (cost_plus - cost_minus) / (2 * c_k * delta_k)
        angles = angles - a_k * gradient_est
        for i in range(len(angles)):  # Used to set angles value within the boundaries during optimization
            angles[i] = min(angles[i], boundaries[i][1])
            angles[i] = max(angles[i], boundaries[i][0])

    return angles, -quantum_avg_computation(angles)  # return optimized angles and associated expectation value


def result_optimization(optimizer: str = "Nelder-Mead", max_iter: int = 100):
    """
    Main function to retrieve results of the algorithm

    :param optimizer: name of classical optimization algorithm (based on scipy library or "SPSA")
    :param max_iter: maximum number of iterations for running SPSA
    """
    if p == 1:
        print("Optimization for", p, "block")
    else:
        print("Optimization for", p, "blocks")

    min_bound = [0.0] * (2 * p)
    max_bound = [2 * π, π] * p
    boundaries = [(min_bound[i], max_bound[i]) for i in range(2 * p)]

    # Generate random initial angles
    angles = []
    for _ in range(p):
        angles.append(rand.uniform(0, 2 * π))
        angles.append(rand.uniform(0, π))

    if optimizer == "SPSA":
        opti_angle, expectation_value = SPSA_optimize(np.array(angles), boundaries, max_iter)
    else:
        Result = minimize(
            quantum_avg_computation,
            np.array(angles),
            method=optimizer,
            bounds=boundaries,
        )
        opti_angle = Result.x
        expectation_value = quantum_avg_computation(opti_angles)

    # Print results
    if MaxCut_value is not None:
        ratio = expectation_value / MaxCut_value
        print("Approximation ratio : ", ratio)
    print("Average cost value obtained: ", expectation_value)
    print(
        "Optimized angles set: γ:",
        opti_angle[0 : 2 * p : 2],
        "β:",
        opti_angle[1 : 2 * p : 2],
    )

    return expectation_value, opti_angle


# Run the algorithm

job = qm.execute(QAOA)
exp, opti_angles = result_optimization()
job.halt()
