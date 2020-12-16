"""QAOA_MaxCut_QUA.py: Implementing QAOA using QUA to perform quantum computation
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

# We import the tools to handle general Graphs
import networkx as nx
from scipy.optimize import minimize, differential_evolution, brute
# We import plotting tools
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import rc


π = np.pi

qm1 = QuantumMachinesManager("3.122.60.129")
# QM configuration based on IBM Quantum Computer :
# Yorktown device https://github.com/Qiskit/ibmq-device-information/tree/master/backends/yorktown/V1


# QM = qm1.open_qm(IBMconfig)
sim_args = {
    'simulate': SimulationConfig(int(5e3))}  # Run on simulation interface

# ## MaxCut problem definition for QAOA

# Generating the connectivity graph with 5 nodes
n = 4
V = np.arange(0, n, 1)
E = [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (3, 2, 1.0), (0, 3, 1.0), (1, 3, 1.0)]

G = nx.Graph()
G.add_nodes_from(V)
G.add_weighted_edges_from(E)
MaxCut_value = 5.  # Value of the ideal MaxCut

# Drawing the Graph G
# colors = ['b' for node in G.nodes()]
# default_axes = plt.axes(frameon=True)
# pos = nx.spring_layout(G)
#
# nx.draw_networkx(G, node_color=colors, node_size=600, alpha=1, ax=default_axes, pos=pos)


""" This script relies on built in functions
By order of calling when using the full algorithm we have the following tree:
1. result_optimization, main organ of whole QAOA outline, returns the result of the algorithm

2. quantum_avg_computation, returns the expectation value of cost Hamiltonian
      for a fixed parametrized quantum circuit ((γ,β) angles fixed), measurement sampling for statistics given by parameter Nrep)
    In this function, processing of data obtained from QUA program is done to return the value to be fed back in the classical optimizer

3. Launch_QAOA, returns the QUA program applying the quantum circuit on the hardware,
      and ensuring the saving of the data using stream-processing features
4. Other smaller functions, which are QUA macros, meant to be modified according to specific hardware implementation """


def generate_binary(n):  # Define a function to generate a list of binary strings (Python function, not related to QUA)

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
    Rz(𝜆-π/2, tgt)
    X90(tgt)
    Rz(π-𝜃, tgt)
    X90(tgt)
    Rz(𝜙-π/2, tgt)



def CNOT(ctrl, tgt):  # To be defined
    return None


def Rz(𝜆, tgt):
    frame_rotation(-𝜆, tgt)


def CU1(𝜆, ctrl, tgt):
    Rz(𝜆 / 2., ctrl)
    CNOT(ctrl, tgt)
    Rz(-𝜆 / 2., tgt)
    CNOT(ctrl, tgt)
    Rz(𝜆 / 2., tgt)


def Rx(𝜆, tgt):
    U3(tgt, 𝜆, -π / 2, π / 2)

def X90(tgt):
    play('X90', tgt)

def Y90(tgt):
    play('Y90', tgt)

def Y180(tgt):
    play('Y180', tgt)



def measurement(resonator, I, Q):  # Simple measurement command, could be changed/generalized
    measure('meas_pulse', resonator, None, ('integW1', I), ('integW2', Q))


def raw_saving(I, Q, k):
    # Saving command
    I_name = 'I' + str(k)  # Variables I1,I2,I3, ...
    Q_name = 'Q' + str(k)
    save(I, I_name)
    save(Q, Q_name)


def state_saving(I, Q, state_estimate, k):  # Do state estimation protocol in QUA, and save the associated state
    # Define coef a & b defining the line separating states 0 & 1 in the IQ Plane (calibration required), here a & b are arbitrary
    a = declare(fixed, value=1.)
    b = declare(fixed, value=1.)
    with if_(Q - a * I - b > 0):
        assign(state_estimate, 1)
    with else_():
        assign(state_estimate, 0)
    # Save the variable into a generic stream defined as state0,1,2 ...
    state_name = 'state' + str(k)
    save(state_estimate, state_name)


def SWAP(qubit1, qubit2):
    CNOT(qubit1, qubit2)
    CNOT(qubit2, qubit1)
    CNOT(qubit1, qubit2)


def Launch_QAOA(γ_set, β_set, G,
                N_rep):  # γ,β are arrays containing angle values, G is a Graph embedding the connectivity of the MaxCut instance
    n = len(list(G.nodes()))  # Number of qubits, equivalent to number of nodes in the Graph
    if (len(γ_set) != len(β_set)):
        raise Error
    p = len(
        γ_set)  # Indicates the number of adiabatic blocks to be called, yields also the number of parameters to be optimized classically

    with program() as QAOA:
        rep = declare(int)  # Iterator for sampling (necessary to have good statistics for expectation value estimation)
        t = declare(int, value=10)  # Assume knowledge of relaxation time

        with for_(rep, init=0, cond=rep <= N_rep, update=rep + 1):  # Do N_rep times the same quantum circuit
            γ = declare(fixed)
            β = declare(fixed)
            if p > 1:
                γ_set_QUA = declare(fixed,
                                    value=γ_set)  # QUA arrays initialized according to input params to use QUA for_ loops
                β_set_QUA = declare(fixed, value=β_set)
            else:
                assign(γ, γ_set[0])
                assign(β, β_set[0])

            # Apply Hadamard on each qubit
            for k in range(n):  # Run what's inside for qubit0, qubit1, ..., qubit4
                q = 'qubit' + str(k)
                Hadamard(q)

            # Apply cost Hamiltonian evolution
            if p > 1:  # If statement necessary because QUA for_each_ does not tolerate an array of size 1
                with for_each_((γ, β), (
                γ_set_QUA, β_set_QUA)):  # Iteration over the set of angles, p parameter is the length of the angle sets

                    for edge in G.edges():  # Iterate over the whole connectivity of Graph G
                        i = edge[0]
                        j = edge[1]
                        node1 = 'qubit' + str(
                            i)  # Define string corresponding to targeted quantum element in the config
                        node2 = 'qubit' + str(j)
                        CU1(2 * γ, node1, node2)
                        Rz(-γ, node1)
                        Rz(-γ, node2)

                        # Mixing Hamiltonian evolution

                    for k in range(n):  # Apply X(β) rotation on each qubit
                        q = 'qubit' + str(k)
                        Rx(-2 * β, q)

            else:  # If γ_set & β_set both have size 1
                for edge in G.edges():
                    i = edge[0]
                    j = edge[1]
                    node1 = 'qubit' + str(i)
                    node2 = 'qubit' + str(j)
                    CU1(2 * γ, node1, node2)
                    Rz(-γ, node1)
                    Rz(-γ, node2)

                    # Mixing Hamiltonian evolution

                for k in range(n):
                    q = 'qubit' + str(k)
                    Rx(-2 * β, q)

            # Measurement and state determination
            for k in range(n):  # Iterate over each resonator 'CPW' in the config file
                I = declare(fixed)  #
                Q = declare(fixed)
                state_estimate = declare(int)
                RR = 'CPW' + str(k)
                measurement(RR, I, Q)
                raw_saving(I, Q, k)  # Save I & Q variables, if deemed necessary by the user
                state_saving(I, Q, state_estimate,
                             k)  # Do state discrimination based on each IQ point and save it in a variable called state0,1,..
                wait(t, RR)  # Reset qubit state by waiting for relaxation
                wait(t, q)

    return QAOA


def cost_function_Max_Cut(x, G):  # Cost function for MaxCut problem, needs to be adapted to the considered optimization problem

    E = G.edges()
    if len(x) != len(G.nodes()):
        return np.nan

    C = 0
    for edge in E:
        e1 = edge[0]
        e2 = edge[1]
        w = G[e1][e2]['weight']
        C = C + w * x[e1] * (1 - x[e2]) + w * x[e2] * (1 - x[e1])

    return C


γ_test = [2]
β_test = [1.5]
γ_test2 = [2., 3]
β_test2 = [1.5, 2]

my_job = qm1.simulate(IBMconfig,Launch_QAOA(γ_test, β_test, G, N_rep=1),
                     SimulationConfig(int(10000), simulation_interface=LoopbackInterface(
                         [("con1", 1, "con1",
                           1)])))

# prog = Launch_QAOA(γ_test2, β_test2, G, N_rep=1)
# print(prog)
# my_job2 = qm1.simulate(IBMconfig, prog,
#                        SimulationConfig(int(10000), simulation_interface=LoopbackInterface(
#                            [("con1", 1, "con1",
#                              1)])))


def quantum_avg_computation(angles, G, Nrep, QM,
                            real_device=False):  # Calculate Hamiltonian expectation value (cost function to optimize)
    n = len(list(G.nodes()))
    l = len(angles)
    γ = angles[0:l:2]
    β = angles[1:l:2]
    if real_device:
        # job = QM.execute(Launch_QAOA(γ, β, G, Nrep))
        print(real_device)
    else:
        job = QM.simulate(IBMconfig, Launch_QAOA(γ, β, G, Nrep),
                          SimulationConfig(int(5000), simulation_interface=LoopbackInterface(
                              [("con1", 1, "con1",
                                1)])))

    results = job.result_handles
    output_states = []
    # Those commands do retrieve the results in generic variables called state#i with #i being a number between 0 and n-1
    # Those results are then stored in a bigger array called output_states
    list(map(exec, [f"state{d}=results.state{d}.fetch_all()['value']" for d in range(n)]))
    list(map(exec, [f"output_states.append(state{d})" for d in range(n)]))

    counts = {}  # Dictionary containing statistics of measurement of each bitstring obtained
    for i in range(len(generate_binary(n))):
        counts[generate_binary(n)[i]] = 0

    for i in range(
            Nrep):  # Here we build in the statistics by picking line by line the bistrings obtained in measurements
        bitstring = ""
        for j in range(len(output_states)):
            bitstring += str(output_states[j][i])
        counts[bitstring] += 1

    avr_C = 0

    for sample in list(counts.keys()):  # Computation of expectation value according to simulation results

        # use sampled bit string x to compute C(x)
        x = [int(num) for num in list(sample)]
        tmp_eng = cost_function_Max_Cut(x, G)

        # compute the expectation value and energy distribution
        avr_C = avr_C + counts[sample] * tmp_eng

    Mp_sampled = avr_C / Nrep

    return -Mp_sampled  # minus sign because optimizer will minimize the function,
    # to get the maximum value one takes the absolute value of the yielded result of  optimization


# Run the QAOA procedure from here, for various adiabatic evolution block numbers
def result_optimization(G, Nrep, QM, real_device=False, optimizer='COBYLA', p_min=1, p_max=5, MaxCut_value=9.0):
    nb_dp = 3  # Choose how many QAOA algorithm you want to run, with different values of p
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
            min_bound.append(0.)
            min_bound.append(0.)
            max_bound.append(2 * π)
            max_bound.append(π)

        p.append(j)
        boundaries = []

        for i in range(2 * j):  # Generate boundaries for each variable during optimization
            boundaries.append((min_bound[i], max_bound[i]))

        if (optimizer == 'brute'):  # Choose optimizer kind (derived from scipy)
            opti_angles1, expminus = brute(func=quantum_avg_computation, ranges=boundaries, args=(G, Nrep, QM, False),
                                           Ns=10,
                                           finish=fmin)
            exp.append(-expminus)
            ratio.append(-expminus / MaxCut_value)
            opti_angles.append(opti_angles1)
        elif (optimizer == 'differential_evolution'):
            Opti_Result = differential_evolution(quantum_avg_computation, bounds=boundaries, args=(G, Nrep, QM, False),
                                                 tol=0.1)
            opti_angles.append(Opti_Result.x)
            exp.append(-Opti_Result.fun)
            ratio.append(-Opti_Result.fun / MaxCut_value)
        else:
            Opti_Result = minimize(fun=quantum_avg_computation, x0=angles, args=(G, Nrep, QM, False), method=optimizer,
                                   bounds=boundaries)
            opti_angles.append(Opti_Result.x)
            exp.append(-Opti_Result.fun)
            ratio.append(-Opti_Result.fun / MaxCut_value)

        print("Optimization done")

    return p, ratio, exp, opti_angles  # Returns costs, optimized angles, associated approximation ratio.

# To finish the program and yield the solution, one might run a quantum_avg_computation once again with optimized angles,
# and retrieve most frequent bitstrings, one of them should correspond to the optimal solution of the problem
test_angles=[1.5, 2.3]
print(quantum_avg_computation(test_angles, G, 1,qm1))