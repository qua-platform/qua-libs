"""QAOA_MaxCut_QUA_IO_usage.py: Implementing QAOA using QUA to perform quantum computation and IO values for optimization
Author: Arthur Strauss - Quantum Machines
Created: 25/11/2020
This script must run on QUA 0.7 or higher
"""

# QM imports
from Config_QAOA import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import QuantumMachine
from qm.qua import *


import random as rand
import math
import numpy as np
import time

# We import the tools to handle general Graphs
import networkx as nx
from scipy.optimize import minimize, differential_evolution, brute

# We import plotting tools
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import rc

π = np.pi

# QM configuration based on IBM Quantum Computer :
# Yorktown device https://github.com/Qiskit/ibmq-device-information/tree/master/backends/yorktown/V1
qm1 = QuantumMachinesManager("192.168.116.129")
QM = qm1.open_qm(IBMconfig)

# ## MaxCut problem definition for QAOA

# Generating the connectivity graph with 5 nodes
n = 4
V = np.arange(0, n, 1)
E = [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (3, 2, 1.0), (0, 3, 1.0), (1, 3, 1.0)]

G = nx.Graph()
G.add_nodes_from(V)
G.add_weighted_edges_from(E)
MaxCut_value = 5.0  # Value of the ideal MaxCut

# Drawing the Graph G
"""colors = ['b' for node in G.nodes()]
default_axes = plt.axes(frameon=True)
pos = nx.spring_layout(G)

nx.draw_networkx(G, node_color=colors, node_size=600, alpha=1, ax=default_axes, pos=pos)"""
p = 1  # Define the depth of the quantum circuit, aka number of adiabatic evolution blocks
Nshots = 10  # Define number of shots necessary to compute expectation value of cost Hamiltonian

# Initialize angles to random to launch the program
QM.set_io1_value(π)
QM.set_io2_value(1)

""" 
We use here multiple global variables, callable in every function : 
- G the graph instance modelling the MaxCut problem we're solving
- n the number of nodes in the graph, which is also the number of qubits in the config (named "qubit0", ..,"qubit(n-1)"
- Nshots, number of shots desired for one expectation value evaluation on the QC (number of times you do the sampling of the QC)
- QM, the Quantum Machine instance opened with a specific hardware configuration
- p, the number of blocks of adiabatic evolution, i.e defines the depth of the quantum circuit preparing QAOA ansatz state 
This script relies on built in functions.
By order of calling when using the full algorithm we have the following tree:
1. result_optimization, main organ of whole QAOA outline, returns the result of the algorithm

2. quantum_avg_computation, returns the expectation value of cost Hamiltonian
      for a fixed parametrized quantum circuit ((γ,β) angles fixed), measurement sampling for statistics given by parameter Nshots)
    In this function, processing of data obtained from QUA program is done to return the value to be fed back in the classical optimizer

3. QAOA, the QUA program applying the quantum circuit on the hardware,
      and ensuring the saving of the data using stream-processing features
4. Other smaller functions, which are QUA macros, meant to be modified according to specific hardware implementation """

# Auxiliary Python functions
def generate_binary(
    n,
):  # Define a function to generate a list of binary strings (Python function, not related to QUA)

    # 2^(n-1)  2^n - 1 inclusive
    bin_arr = range(0, int(math.pow(2, n)))
    bin_arr = [bin(i)[2:] for i in bin_arr]

    # Prepending 0's to binary strings
    max_len = len(max(bin_arr, key=len))
    bin_arr = [i.zfill(max_len) for i in bin_arr]

    return bin_arr


def cost_function_C(
    x, G
):  # Cost function for MaxCut problem, needs to be adapted to the considered optimization problem

    E = G.edges()
    if len(x) != len(G.nodes()):
        return np.nan

    C = 0
    for edge in E:
        e1 = edge[0]
        e2 = edge[1]
        w = G[e1][e2]["weight"]
        C = C + w * x[e1] * (1 - x[e2]) + w * x[e2] * (1 - x[e1])

    return C


# QUA macros (pulse definition of quantum gates)
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


def CR(
    ctrl, tgt
):  # gate created based on the implementation on IBM in the following paper : https://arxiv.org/abs/2004.06755
    return None


def CNOT(ctrl, tgt):  # To be defined
    frame_rotation(π / 2, ctrl)
    X90(tgt)
    CR(tgt, ctrl)


def Rz(𝜆, tgt):
    frame_rotation(-𝜆, tgt)


def Ry(𝜆, tgt):
    U3(tgt, 𝜆, 0, 0)


def CU1(𝜆, ctrl, tgt):
    Rz(𝜆 / 2.0, ctrl)
    CNOT(ctrl, tgt)
    Rz(-𝜆 / 2.0, tgt)
    CNOT(ctrl, tgt)
    Rz(𝜆 / 2.0, tgt)


def Rx(𝜆, tgt):
    U3(tgt, 𝜆, -π / 2, π / 2)


def X90(tgt):
    play("X90", tgt)


def Y90(tgt):
    play("Y90", tgt)


def Y180(tgt):
    play("Y180", tgt)


def SWAP(qubit1, qubit2):
    CNOT(qubit1, qubit2)
    CNOT(qubit2, qubit1)
    CNOT(qubit1, qubit2)


def measurement(RR, I, Q):  # Simple measurement command, could be changed/generalized
    measure("meas_pulse", RR, None, ("integW1", I), ("integW2", Q))


# Stream processing QUA macros
def raw_saving(I, Q, I_stream, Q_stream):
    # Saving command
    save(I, I_stream)
    save(Q, Q_stream)


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


# Main Python functions for launching QAOA algorithm


def encode_angles_in_IO(
    γ, β
):  # Insert angles values using IO1 by keeping track of where we are in the QUA program
    if len(γ) != len(β):
        raise IndexError
    for i in range(len(γ)):
        while not (job.is_paused()):
            time.sleep(0.01)
        QM.set_io1_value(γ[i])
        QM.set_io2_value(β[i])
        job.resume()


def quantum_avg_computation(
    angles,
):  # Calculate Hamiltonian expectation value (cost function to optimize)

    γ = angles[0 : 2 * p : 2]
    β = angles[1 : 2 * p : 2]
    job.resume()

    encode_angles_in_IO(γ, β)

    # Implement routine here to set IO values and resume the program, and adapt prior part (or remove it)

    while not (job.is_paused()):
        time.sleep(0.0001)
    results = job.result_handles
    # time.sleep()
    output_states = []
    # Those commands do retrieve the results in generic variables called state#i with #i being a number between 0 and n-1
    # Those results are then stored in a bigger array called output_states
    timing = results.timing.fetch_all()["value"]
    print(timing[-1])
    for d in range(n):
        output_states.append(results.get("state" + str(d)).fetch_all())

    counts = (
        {}
    )  # Dictionary containing statistics of measurement of each bitstring obtained
    for i in range(2 ** n):
        counts[generate_binary(n)[i]] = 0

    for i in range(
        Nshots
    ):  # Here we build in the statistics by picking line by line the bistrings obtained in measurements
        bitstring = ""
        for j in range(len(output_states)):
            bitstring += str(output_states[j][i])
        counts[bitstring] += 1

    avr_C = 0

    for bitstr in list(
        counts.keys()
    ):  # Computation of expectation value according to simulation results

        # use sampled bit string x to compute C(x)
        x = [int(num) for num in list(bitstr)]
        tmp_eng = cost_function_C(x, G)

        # compute the expectation value and energy distribution
        avr_C = avr_C + counts[bitstr] * tmp_eng

    Mp_sampled = avr_C / Nshots

    return -Mp_sampled  # minus sign because optimizer will minimize the function,
    # to get the maximum value one takes the absolute value of the yielded result of  optimization


def SPSA_optimize(
    init_angles, boundaries, max_iter=100
):  # Use SPSA optimization scheme, source : https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF

    a, c, A, alpha, gamma = SPSA_calibration()
    angles = init_angles

    for k in range(1, max_iter):
        a_k = a / (k + A) ** alpha
        c_k = c / k ** gamma
        delta_k = (
            2 * np.round(np.random.uniform(0, 1, 2 * p)) - 1
        )  # Vector of random variables issued from a Bernouilli distribution +1,-1, could be something else
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


# Run the QAOA procedure from here, for various adiabatic evolution block numbers
def result_optimization(max_iter=100):

    if p == 1:
        print("Optimization for", p, "block")
    else:
        print("Optimization for", p, "blocks")
    angles = []
    min_bound = []
    max_bound = []

    for i in range(p):  # Generate random initial angles
        angles.append(rand.uniform(0, 2 * π))
        angles.append(rand.uniform(0, π))
        min_bound.append(0.0)
        min_bound.append(0.0)
        max_bound.append(2 * π)
        max_bound.append(π)

    boundaries = []

    for i in range(2 * p):  # Generate boundaries for each variable during optimization
        boundaries.append((min_bound[i], max_bound[i]))

    opti_angle, expectation_value = SPSA_optimize(
        np.array(angles), boundaries, max_iter
    )

    ratio = expectation_value / MaxCut_value
    print("Approximation ratio : ", ratio)
    print("Average cost value obtained: ", expectation_value)
    print("Optimized angles set: ", opti_angle)

    return (
        ratio,
        expectation_value,
        opti_angle,
    )  # Returns costs, optimized angles, associated approximation ratio.


# To finish the program and yield the solution, one might run a quantum_avg_computation once again with optimized angles,
# and retrieve most frequent bitstrings, one of them should correspond to the optimal solution of the problem


# QUA program

with program() as QAOA:
    rep = declare(
        int
    )  # Iterator for sampling (necessary to have good statistics for expectation value estimation)
    t = declare(int, value=10)  # Assume knowledge of relaxation time
    timing = declare(int, value=0)

    I = declare(fixed)
    Q = declare(fixed)
    state_estimate = declare(int)

    state_streams, I_streams, Q_streams = [None] * n, [None] * n, [None] * n
    state_strings, I_strings, Q_strings = [None] * n, [None] * n, [None] * n
    for i in range(n):
        state_strings[i] = "state" + str(i)
        I_strings[i] = "I" + str(i)
        Q_strings[i] = "Q" + str(i)
        state_streams[i], I_streams[i], Q_streams[i] = (
            declare_stream(),
            declare_stream(),
            declare_stream(),
        )

    with infinite_loop_():
        pause()
        γ = declare(fixed)
        β = declare(fixed)
        block = declare(int)
        γ_set = declare(fixed, size=p)
        β_set = declare(fixed, size=p)

        with for_(block, init=0, cond=block < p, update=i + 1):
            pause()
            assign(γ_set[block], IO1)
            assign(β_set[block], IO2)

        with for_(
            rep, init=0, cond=rep <= Nshots, update=rep + 1
        ):  # Do Nshots times the same quantum circuit

            # Apply Hadamard on each qubit
            for k in range(n):  # Run what's inside for qubit0, qubit1, ..., qubit4
                q = "qubit" + str(k)
                Hadamard(q)

            with for_each_((γ, β), (γ_set, β_set)):
                #  Cost Hamiltonian evolution
                for edge in G.edges():  # Iterate over the whole connectivity of Graph G
                    i = edge[0]
                    j = edge[1]
                    qc = "qubit" + str(
                        i
                    )  # Define string corresponding to targeted quantum element in the config
                    qt = "qubit" + str(j)
                    CU1(2 * γ, qc, qt)
                    Rz(-γ, qc)
                    Rz(-γ, qt)

                    # Mixing Hamiltonian evolution

                for k in range(n):  # Apply X(β) rotation on each qubit
                    q = "qubit" + str(k)
                    Rx(-2 * β, q)
                save(γ, "gamma")
                save(β, "beta")

            # Measurement and state determination

            for k in range(n):  # Iterate over each resonator 'CPW' in the config file
                RR = "CPW" + str(k)
                q = "qubit" + str(k)
                measurement(RR, I, Q)

                state_saving(
                    I, Q, state_estimate, state_streams[k]
                )  # Do state discrimination based on each IQ point and save it in a variable called state0,1,..
                # Active reset, flip the qubit is measured state is 1
                with if_(state_estimate == 1):
                    Rx(π, q)

                raw_saving(
                    I, Q, I_streams[k], Q_streams[k]
                )  # Save I & Q variables, if deemed necessary by the user

        assign(timing, timing + 1)
        save(timing, "timing")
    with stream_processing():
        for i in range(n):
            state_streams[i].buffer(Nshots).save(state_strings[i])
            I_streams[i].buffer(Nshots).save(I_strings[i])
            Q_streams[i].buffer(Nshots).save(Q_strings[i])


job = QM.execute(QAOA)

# Try out the QAOA for the problem defined by the Maxcut for graph G
ratio, exp, opti_angles = result_optimization()


job.halt()
