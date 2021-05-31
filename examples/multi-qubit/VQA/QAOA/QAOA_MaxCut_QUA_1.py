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

qm1 = QuantumMachinesManager()
# QM configuration based on IBM Quantum Computer :
# Yorktown device https://github.com/Qiskit/ibmq-device-information/tree/master/backends/yorktown/V1


# QM = qm1.open_qm(IBMconfig)
sim_args = {"simulate": SimulationConfig(int(5e3))}  # Run on simulation interface

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


def Launch_QAOA(
    γ_set, β_set, G, N_rep
):  # γ,β are arrays containing angle values, G is a Graph embedding the connectivity of the MaxCut instance
    n = len(
        list(G.nodes())
    )  # Number of qubits, equivalent to number of nodes in the Graph
    if len(γ_set) != len(β_set):
        raise IndexError("angles provided do not match length of the circuit")
    p = len(
        γ_set
    )  # Indicates the number of adiabatic blocks to be called, yields also the number of parameters to be optimized classically

    with program() as QAOA:
        rep = declare(
            int
        )  # Iterator for sampling (necessary to have good statistics for expectation value estimation)
        t = declare(int, value=10)  # Assume knowledge of relaxation time

        with for_(
            rep, init=0, cond=rep <= N_rep, update=rep + 1
        ):  # Do N_rep times the same quantum circuit
            γ = declare(fixed)
            β = declare(fixed)
            if p > 1:
                γ_set_QUA = declare(
                    fixed, value=γ_set
                )  # QUA arrays initialized according to input params to use QUA for_ loops
                β_set_QUA = declare(fixed, value=β_set)
            else:
                assign(γ, γ_set[0])
                assign(β, β_set[0])

            # Apply Hadamard on each qubit
            for k in range(n):  # Run what's inside for qubit0, qubit1, ..., qubit4
                q = "qubit" + str(k)
                Hadamard(q)

            # Apply cost Hamiltonian evolution
            if (
                p > 1
            ):  # If statement necessary because QUA for_each_ does not tolerate an array of size 1
                with for_each_(
                    (γ, β), (γ_set_QUA, β_set_QUA)
                ):  # Iteration over the set of angles, p parameter is the length of the angle sets

                    for (
                        edge
                    ) in G.edges():  # Iterate over the whole connectivity of Graph G
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

                    for k in range(n):  # Apply X(β) rotation on each qubit
                        q = "qubit" + str(k)
                        Rx(-2 * β, q)

            else:  # If γ_set & β_set both have size 1
                for edge in G.edges():
                    i = edge[0]
                    j = edge[1]
                    node1 = "qubit" + str(i)
                    node2 = "qubit" + str(j)
                    CU1(2 * γ, node1, node2)
                    Rz(-γ, node1)
                    Rz(-γ, node2)

                    # Mixing Hamiltonian evolution

                for k in range(n):
                    q = "qubit" + str(k)
                    Rx(-2 * β, q)

            # Measurement and state determination
            for k in range(n):  # Iterate over each resonator 'CPW' in the config file
                I = declare(fixed)  #
                Q = declare(fixed)
                state_estimate = declare(int)
                RR = "CPW" + str(k)
                measurement(RR, I, Q)
                raw_saving(
                    I, Q, k
                )  # Save I & Q variables, if deemed necessary by the user
                state_saving(
                    I, Q, state_estimate, k
                )  # Do state discrimination based on each IQ point and save it in a variable called state0,1,..
                wait(t, RR)  # Reset qubit state by waiting for relaxation
                wait(t, q)

    return QAOA


γ_test = [2]
β_test = [1.5]
γ_test2 = [2.0, 3]
β_test2 = [1.5, 2]

my_job = qm1.simulate(
    IBMconfig,
    Launch_QAOA(γ_test, β_test, G, N_rep=1),
    SimulationConfig(
        int(10000), simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)])
    ),
)

# prog = Launch_QAOA(γ_test2, β_test2, G, N_rep=1)
# print(prog)
# my_job2 = qm1.simulate(IBMconfig, prog,
#                        SimulationConfig(int(10000), simulation_interface=LoopbackInterface(
#                            [("con1", 1, "con1",
#                              1)])))


def quantum_avg_computation_classic(
    angles, G, Nrep, QM, real_device=False
):  # Calculate Hamiltonian expectation value (cost function to optimize)
    n = len(list(G.nodes()))
    l = len(angles)
    γ = angles[0:l:2]
    β = angles[1:l:2]
    if real_device:
        # job = QM.execute(Launch_QAOA(γ, β, G, Nrep))
        print(real_device)
    else:
        job = QM.simulate(
            IBMconfig,
            Launch_QAOA(γ, β, G, Nrep),
            SimulationConfig(
                int(5000),
                simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)]),
            ),
        )

    results = job.result_handles
    output_states = []
    # Those commands do retrieve the results in generic variables called state#i with #i being a number between 0 and n-1
    # Those results are then stored in a bigger array called output_states
    for d in range(n):
        output_states.append(results.get("state" + str(d)).fetch_all())

    counts = (
        {}
    )  # Dictionary containing statistics of measurement of each bitstring obtained
    for i in range(2 ** n):
        counts[generate_binary(n)[i]] = 0

    for i in range(
        Nrep
    ):  # Here we build in the statistics by picking line by line the bistrings
        # obtained in measurements
        bitstring = ""
        for j in range(len(output_states)):
            bitstring += str(output_states[j][i])
        counts[bitstring] += 1

    avr_C = 0

    for sample in list(
        counts.keys()
    ):  # Computation of expectation value according to simulation results

        # use sampled bit string x to compute C(x)
        x = [int(num) for num in list(sample)]
        tmp_eng = cost_function_C(x, G)

        # compute the expectation value and energy distribution
        avr_C = avr_C + counts[sample] * tmp_eng

    Mp_sampled = avr_C / Nrep

    return -Mp_sampled  # minus sign because optimizer will minimize the function,
    # to get the maximum value one takes the absolute value of the yielded result of  optimization


# Run the QAOA procedure from here, for various adiabatic evolution block numbers
def result_optimization(
    G,
    Nrep,
    QM,
    real_device=False,
    optimizer="COBYLA",
    p_min=1,
    p_max=5,
    MaxCut_value=9.0,
):
    nb_dp = (
        3  # Choose how many QAOA algorithm you want to run, with different values of p
    )
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

        if optimizer == "brute":  # Choose optimizer kind (derived from scipy)
            opti_angles1, expminus = brute(
                func=quantum_avg_computation_classic,
                ranges=boundaries,
                args=(G, Nrep, QM, False),
                Ns=10,
            )
            exp.append(-expminus)
            ratio.append(-expminus / MaxCut_value)
            opti_angles.append(opti_angles1)
        elif optimizer == "differential_evolution":
            Opti_Result = differential_evolution(
                quantum_avg_computation_classic,
                bounds=boundaries,
                args=(G, Nrep, QM, False),
                tol=0.1,
            )
            opti_angles.append(Opti_Result.x)
            exp.append(-Opti_Result.fun)
            ratio.append(-Opti_Result.fun / MaxCut_value)
        else:
            Opti_Result = minimize(
                fun=quantum_avg_computation_classic,
                x0=angles,
                args=(G, Nrep, QM, False),
                method=optimizer,
                bounds=boundaries,
            )
            opti_angles.append(Opti_Result.x)
            exp.append(-Opti_Result.fun)
            ratio.append(-Opti_Result.fun / MaxCut_value)

        print("Optimization done")

    return (
        p,
        ratio,
        exp,
        opti_angles,
    )  # Returns costs, optimized angles, associated approximation ratio.


# To finish the program and yield the solution, one might run a quantum_avg_computation once again with optimized angles,
# and retrieve most frequent bitstrings, one of them should correspond to the optimal solution of the problem
test_angles = [1.5, 2.3]
print(quantum_avg_computation_classic(test_angles, G, 1, qm1))
