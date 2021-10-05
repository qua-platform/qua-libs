"""QAOA_MaxCut_QUA_IO_usage.py: Implementation of QAOA for solving a MaxCut problem instance with real-time capabilities
Author: Arthur Strauss - Quantum Machines
Created: 25/11/2020
This script must run on QUA 0.7 or higher
"""

# QM imports
from Config_QAOA import *
import networkx as nx
from typing import List
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from qm.QuantumMachinesManager import QuantumMachinesManager
import time
import random as rand


# QM configuration based on IBM Quantum Computer :
# Yorktown device https://github.com/Qiskit/ibmq-device-information/tree/master/backends/yorktown/V1
qm1 = QuantumMachinesManager()
QM = qm1.open_qm(IBMconfig)

# MaxCut problem definition for QAOA

# Generating the connectivity graph with 5 nodes
n = 4
V = np.arange(0, n, 1)
E = [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (3, 2, 1.0), (0, 3, 1.0), (1, 3, 1.0)]

G = nx.Graph()
G.add_nodes_from(V)
G.add_weighted_edges_from(E)
E = G.edges()
MaxCut_value = 5.0  # Value of the ideal MaxCut (set to None if unknown)
# Drawing the Graph G
colors = ['b' for node in G.nodes()]
default_axes = plt.axes(frameon=True)
pos = nx.spring_layout(G)

nx.draw_networkx(G, node_color=colors, node_size=600, alpha=1, ax=default_axes, pos=pos)

# Algorithm hyperparameters
p = 1  # depth of quantum circuit, i.e number of adiabatic evolution blocks
N_shots = 10  # Sampling number for expectation value computation

# Initialize angles to random to launch the program
QM.set_io1_value(np.random.uniform(0, 2*π))
QM.set_io2_value(np.random.uniform(0, π))

""" 
We use here multiple global variables, callable in every function : 
- G: graph instance modelling the MaxCut problem we're solving
- n:  number of nodes in the graph, which is also the number of qubits
- N_shots: sampling number for expectation value computation
- QM: Quantum Machine instance opened with a specific hardware configuration
- p:  number of adiabatic evolution blocks, i.e defines the depth of the quantum circuit preparing QAOA ansatz state

This script relies on built in functions.
By order of calling when using the full algorithm we have the following tree:
1. result_optimization, main organ of whole QAOA outline, returns the result of the algorithm

2. quantum_avg_computation, returns the expectation value of cost Hamiltonian
      for a specific parametrized quantum circuit ((γ,β) angles fixed),
      measurement sampling for statistics given by parameter N_shots)

3. QAOA, the QUA program 
"""

th = 0.  # Threshold for state discrimination
# Introduce registers to address relevant quantum elements in config
q = [f"q{i}" for i in range(n)]
rr = [f"rr{i}" for i in range(n)]

with program() as QAOA:
    rep = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    state = declare(int, size=n)
    γ = declare(fixed)
    β = declare(fixed)
    block = declare(int)
    γ_set = declare(fixed, size=p)
    β_set = declare(fixed, size=p)
    Cut = declare(fixed, value=0)
    Expectation_value = declare(fixed, value=0.)
    w = declare(fixed)
    var = declare(fixed)

    with infinite_loop_():
        pause()

        # Load circuit parameters using IO variables
        with for_(block, init=0, cond=block < p, update=block + 1):
            pause()
            assign(γ_set[block], IO1)
            assign(β_set[block], IO2)

        # Start running quantum circuit (repeated N_shots times)
        with for_(rep, init=0, cond=rep <= N_shots, update=rep + 1):
            # Apply Hadamard on each qubit
            for k in range(n):  # Run what's inside for qubit0, qubit1, ..., qubit4
                Hadamard(q[k])

            with for_each_((γ, β), (γ_set, β_set)):
                #  Cost Hamiltonian evolution
                for edge in E:  # Iterate over the whole connectivity of Graph G
                    ctrl, tgt = q[edge[0]], q[edge[1]]
                    CU1(2 * γ, ctrl, tgt)
                    Rz(-γ, ctrl)
                    Rz(-γ, tgt)

                # Mixing Hamiltonian evolution
                for k in range(n):  # Apply X(β) rotation on each qubit
                    Rx(-2 * β, q[k])

            align()

            # Measurement and state determination
            for k in range(n):  # Iterate over each resonator
                measure("meas_pulse", rr[k], None, integration.full("integW1", I))
                assign(state[k], Cast.to_int(I > th))

                # Active reset, flip the qubit is measured state is 1
                with if_(Cast.to_bool(state[k])):
                    Rx(π, q[k])

            # Cost function evaluation
            for e in E:  # for each edge in the graph
                e1 = int(e[0])
                e2 = int(e[1])
                assign(w, G[e1][e2]["weight"])  # Retrieve weight associated to edge e
                assign(Cut, Cut +
                       w *
                       (Cast.to_fixed(state[e1]) * (1. - Cast.to_fixed(state[e2])) +
                        Cast.to_fixed(state[e2]) * (1. - Cast.to_fixed(state[e1]))
                        )
                       )
            assign(Expectation_value, Expectation_value + Math.div(Cut, N_shots))

        save(Expectation_value, "exp_value")


# Main Python functions for launching QAOA algorithm

def encode_angles_in_IO(gamma: List[float], beta: List[float]):
    """
    Insert parameters using IO variables by keeping track of where we are in the QUA program
    """
    if len(gamma) != len(beta):
        raise IndexError("Angle sets are not matching desired circuit depth")
    for i in range(len(gamma)):
        while not (job.is_paused()):
            time.sleep(0.01)
        QM.set_io1_value(gamma[i])
        QM.set_io2_value(beta[i])
        job.resume()


def quantum_avg_computation(angles: List[float]):  # Calculate Hamiltonian expectation value (cost function to optimize)

    gamma = angles[0: 2 * p: 2]
    beta = angles[1: 2 * p: 2]
    job.resume()

    encode_angles_in_IO(gamma, beta)

    # Wait for experiment to be completed
    while not (job.is_paused()):
        time.sleep(0.0001)
    results = job.result_handles
    exp_value = results.exp_value.fetch_all()

    return -exp_value  # minus sign because optimizer finds the minimum


def SPSA_optimize(init_angles, boundaries, max_iter=100):
    """
    Use SPSA optimization scheme, source : https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF
    """

    a, c, A, alpha, gamma = SPSA_calibration()
    angles = init_angles

    for j in range(1, max_iter):
        a_k = a / (j + A) ** alpha
        c_k = c / j ** gamma
        delta_k = (
                2 * np.round(np.random.uniform(0, 1, 2 * p)) - 1
        )  # Vector of random variables issued from a Bernoulli distribution +1,-1, could be something else
        angles_plus = angles + c_k * delta_k
        angles_minus = angles - c_k * delta_k
        cost_plus = quantum_avg_computation(angles_plus)
        cost_minus = quantum_avg_computation(angles_minus)
        gradient_est = (cost_plus - cost_minus) / (2 * c_k * delta_k)
        angles = angles - a_k * gradient_est
        for i in range(
                len(angles)
        ):  # Used to set angles value within the boundaries during optimization
            angles[i] = min(angles[i], boundaries[i][1])
            angles[i] = max(angles[i], boundaries[i][0])

    return angles, -quantum_avg_computation(
        angles
    )  # return optimized angles and associated expectation value


def SPSA_calibration():
    return 1.0, 1.0, 1.0, 1.0, 1.0


# Run QAOA procedure from here, for various adiabatic evolution block numbers
def result_optimization(optimizer: str = 'Nelder-Mead', max_iter: int = 100):
    if p == 1:
        print("Optimization for", p, "block")
    else:
        print("Optimization for", p, "blocks")

    min_bound = [0.] * (2 * p)
    max_bound = [2*π, π] * p
    boundaries = [(min_bound[i], max_bound[i]) for i in range(2 * p)]

    # Generate random initial angles
    angles = []
    for i in range(p):
        angles.append(rand.uniform(0, 2 * π))
        angles.append(rand.uniform(0, π))

    if optimizer == "SPSA":
        opti_angle, expectation_value = SPSA_optimize(np.array(angles), boundaries, max_iter)
    else:
        Result = minimize(quantum_avg_computation, np.array(angles), method=optimizer, bounds=boundaries)
        opti_angle = Result.x
        expectation_value = quantum_avg_computation(opti_angles)

    # Print results
    if MaxCut_value is not None:
        ratio = expectation_value / MaxCut_value
        print("Approximation ratio : ", ratio)
    print("Average cost value obtained: ", expectation_value)
    print("Optimized angles set: γ:", opti_angle[0: 2*p: 2], "β:", opti_angle[1: 2 * p: 2])

    return expectation_value, opti_angle


# Run the algorithm

job = QM.execute(QAOA)
exp, opti_angles = result_optimization()
job.halt()
