"""QAOA_MaxCut_QUA_IO_usage.py: Implementing QAOA using QUA to perform quantum computation and IO values for optimization
Author: Arthur Strauss - Quantum Machines
Created: 25/11/2020
Created on QUA version: 0.5.138
"""

# QM imports
from Config_QAOA import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import QuantumMachine
from qm import SimulationConfig
from qm import LoopbackInterface
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

# import math tools
π = np.pi
qm1 = QuantumMachinesManager()
# QM configuration based on IBM Quantum Computer :
# Yorktown device https://github.com/Qiskit/ibmq-device-information/tree/master/backends/yorktown/V1

QM = qm1.open_qm(IBMconfig)
sim_args = {"simulate": SimulationConfig(int(5e3))}  # Run on simulation interface

# ## MaxCut problem definition for QAOA

# Generating the connectivity graph with 5 nodes
n = 5
V = np.arange(0, n, 1)
E = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0), (2, 3, 1.0), (2, 4, 1.0), (3, 4, 1.0)]

G = nx.Graph()
G.add_nodes_from(V)
G.add_weighted_edges_from(E)
MaxCut_value = 9.0  # Value of the ideal MaxCut

# Drawing the Graph G
# colors = ['b' for node in G.nodes()]
# default_axes = plt.axes(frameon=True)
# pos = nx.spring_layout(G)
#
# nx.draw_networkx(G, node_color=colors, node_size=600, alpha=1, ax=default_axes, pos=pos)

Nrep = 3  # Define number of shots necessary to compute expectation value of cost Hamiltonian

# Initialize angles to random to launch the program
QM.set_io1_value(π)
QM.set_io2_value(1)

""" 
We use here multiple global variables, callable in every function : 
- G the graph instance modelling the MaxCut problem we're solving
- n the number of nodes in the graph, which is also the number of qubits in the config (named "qubit0", ..,"qubit(n-1)"
- Nrep, number of shots desired for one expectation value evaluation on the QC (number of times you do the sampling of the QC)
- QM, the Quantum Machine instance opened with a specific hardware configuration

This script relies on built in functions.
By order of calling when using the full algorithm we have the following tree:
1. result_optimization, main organ of whole QAOA outline, returns the result of the algorithm

2. quantum_avg_computation, returns the expectation value of cost Hamiltonian
      for a fixed parametrized quantum circuit ((γ,β) angles fixed), measurement sampling for statistics given by parameter Nrep)
    In this function, processing of data obtained from QUA program is done to return the value to be fed back in the classical optimizer

3. QAOA, the QUA program applying the quantum circuit on the hardware,
      and ensuring the saving of the data using stream-processing features
4. Other smaller functions, which are QUA macros, meant to be modified according to specific hardware implementation """


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
    Rz(𝜙 - π / 2)


def CNOT(ctrl, tgt):  # To be defined
    return None


def Rz(𝜆, tgt):
    frame_rotation(-𝜆, tgt)


def CU1(𝜆, ctrl, tgt):
    Rz(𝜆 / 2.0, ctrl)
    CNOT(ctrl, tgt)
    Rz(-𝜆 / 2.0, tgt)
    CNOT(ctrl, tgt)
    Rz(𝜆 / 2.0, tgt)


def Rx(𝜆, tgt):
    U3(𝜆, -π / 2, π / 2)


def X90(tgt):
    play("Drag_Op_I", tgt)


def Y90(tgt):
    play("Drag_Op_Q", tgt)


def measurement(RR, I, Q):  # Simple measurement command, could be changed/generalized
    measure("meas_pulse", RR, None, ("integW1", I), ("integW2", Q))


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


def SWAP(qubit1, qubit2):
    CNOT(qubit1, qubit2)
    CNOT(qubit2, qubit1)
    CNOT(qubit1, qubit2)


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


def encode_angles_in_IO(
    γ, β
):  # Insert angles values using IO1 by keeping track of where we are in the QUA program
    if len(γ) != len(β):
        raise IndexError
    for i in range(len(γ)):
        while not (job.is_paused()):
            time.sleep(0.1)
        QM.set_io1_value(γ[i])
        job.resume()
        while not (job.is_paused()):
            time.sleep(0.1)
        QM.set_io1_value(β[i])
        job.resume()


def quantum_avg_computation(
    angles,
):  # Calculate Hamiltonian expectation value (cost function to optimize)

    l = len(angles)
    γ = angles[0:l:2]
    β = angles[1:l:2]
    QM.set_io2_value(int(l / 2))

    encode_angles_in_IO(γ, β)

    # Implement routine here to set IO values and resume the program, and adapt prior part (or remove it)

    while not (job.is_paused()):
        time.sleep(0.2)
    time.sleep(0.4)
    results = job.result_handles
    output_states = []
    # Those commands do retrieve the results in generic variables called state#i with #i being a number between 0 and n-1
    # Those results are then stored in a bigger array called output_states
    for d in range(n):
        output_states.append(results.get("state" + str(d)).fetch_all())

    counts = (
        {}
    )  # Dictionary containing statistics of measurement of each bitstring obtained
    for i in range(len(generate_binary(n))):
        counts[generate_binary(n)[i]] = 0

    for i in range(
        Nrep
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

    Mp_sampled = avr_C / Nrep

    return -Mp_sampled  # minus sign because optimizer will minimize the function,
    # to get the maximum value one takes the absolute value of the yielded result of  optimization


def SPSA_optimize(
    init_angles, boundaries, max_iter=100
):  # Use SPSA optimization scheme, source : https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF

    a, c, A, alpha, gamma = SPSA_calibration()
    angles = init_angles

    for k in range(max_iter):
        a_k = a / (k + A) ** alpha
        c_k = c / k ** gamma
        delta_k = (
            2 * np.round(np.random.uniform(0, 1, max_iter)) - 1
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

    return angles, quantum_avg_computation(
        angles
    )  # return optimized angles and associated expectation value


# Worth it to define new version of quantum_avg to return also
# most probable bistring


def SPSA_calibration():
    return 1.0, 1.0, 1.0, 1.0, 1.0


# Run the QAOA procedure from here, for various adiabatic evolution block numbers
def result_optimization(p_min=1, p_max=5, max_iter=100):
    p = []
    opti_angles = []
    ratio = []
    exp = []

    for j in range(p_min, p_max + 1):  # Iterate over the number of adiabatic blocks
        if j == 1:
            print("Optimization for", j, "block")
        else:
            print("Optimization for", j, "blocks")
        angles = []
        min_bound = []
        max_bound = []

        for i in range(j):  # Generate random initial angles
            angles.append(rand.uniform(0, 2 * π))
            angles.append(rand.uniform(0, π))
            min_bound.append(0.0)
            min_bound.append(0.0)
            max_bound.append(2 * π)
            max_bound.append(π)

        p.append(j)
        boundaries = []

        for i in range(
            2 * j
        ):  # Generate boundaries for each variable during optimization
            boundaries.append((min_bound[i], max_bound[i]))

        opti_angle, expectation_value = SPSA_optimize(angles, boundaries, max_iter)
        opti_angles.append(opti_angle)
        exp.append(expectation_value)
        ratio.append(expectation_value / MaxCut_value)

    return (
        p,
        ratio,
        exp,
        opti_angles,
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
    with infinite_loop_():
        pause()
        p = declare(int, value=IO2)
        γ = declare(fixed)
        β = declare(fixed)
        block = declare(int)
        with for_(
            rep, init=0, cond=rep <= Nrep, update=rep + 1
        ):  # Do Nrep times the same quantum circuit

            # Apply Hadamard on each qubit
            for k in range(n):  # Run what's inside for qubit0, qubit1, ..., qubit4
                q = "qubit" + str(k)
                Hadamard(q)

            with for_(block, init=0, cond=block < p, update=block + 1):
                #  Cost Hamiltonian evolution
                pause()
                assign(γ, IO1)

                for edge in G.edges():  # Iterate over the whole connectivity of Graph G
                    i = edge[0]
                    j = edge[1]
                    node1 = "qubit" + str(
                        i
                    )  # Define string corresponding to targeted quantum element in the config
                    node2 = "qubit" + str(j)
                    CU1(2 * γ, node1, node2)
                    Rz(-γ, node1)
                    Rz(-γ, node2)

                    # Mixing Hamiltonian evolution
                pause()
                assign(β, IO1)
                for k in range(n):  # Apply X(β) rotation on each qubit
                    q = "qubit" + str(k)
                    Rx(-2 * β, q)

            # Measurement and state determination
            state_streams, I_streams, Q_streams = [None] * n, [None] * n, [None] * n
            state_strings, I_strings, Q_strings = [None] * n, [None] * n, [None] * n
            for i in range(n):
                state_strings[i] = "state" + str(i)
                I_strings[i] = "I" + str(i)
                Q_strings[i] = "Q" + str(i)

            for k in range(n):  # Iterate over each resonator 'CPW' in the config file
                I = declare(fixed)  #
                Q = declare(fixed)
                state_estimate = declare(int)
                RR = "CPW" + str(k)
                measurement(RR, I, Q)

                # Stream processing

                state_streams[k] = declare_stream()
                I_streams[k] = declare_stream()
                Q_streams[k] = declare_stream()
                raw_saving(
                    I, Q, I_streams[k], Q_streams[k]
                )  # Save I & Q variables, if deemed necessary by the user
                state_saving(
                    I, Q, state_estimate, state_streams[k]
                )  # Do state discrimination based on each IQ point and save it in a variable called state0,1,..
                wait(t, RR)  # Reset qubit state by waiting for relaxation
                wait(t, q)
        assign(timing, timing + 1)
        save(timing, "timing")
    with stream_processing():
        for i in range(n):
            state_streams[i].buffer(Nrep).save(state_strings[i])
            I_streams[i].buffer(Nrep).save(I_strings[i])
            Q_streams[i].buffer(Nrep).save(Q_strings[i])

job = QM.execute(QAOA)

# Try out the QAOA for the problem defined by the Maxcut for graph G
p, ratio, exp, opti_angles = result_optimization(p_min=1, p_max=1)
print(ratio, opti_angles)
# Plot out performance of QAOA in terms of number of adiabatic evolution blocks
rc("font", **{"family": "sans-serif", "sans-serif": ["Futura"]})
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc("text", usetex=True)
fig1, ax = plt.subplots()
ax.plot(p, ratio, "x r")
ax.set(
    xlabel="\# of blocks p",
    ylabel="$F_p/C_{max}$",
    title="Best expectation value obtained after optimization of angles",
)
ax.grid()
