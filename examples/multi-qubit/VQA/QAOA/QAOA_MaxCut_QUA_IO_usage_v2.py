"""QAOA_MaxCut_QUA_IO_usage.py: Implementing QAOA using QUA to perform quantum computation and IO values for optimization
Author: Arthur Strauss - Quantum Machines
Created: 25/11/2020
This script must run on QUA 0.7 or higher
"""

# QM imports
from Config_QAOA import *
import networkx as nx
from scipy.optimize import minimize, differential_evolution, brute

# We import plotting tools
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import rc

# QM configuration based on IBM Quantum Computer :
# Yorktown device https://github.com/Qiskit/ibmq-device-information/tree/master/backends/yorktown/V1
qm1 = QuantumMachinesManager()
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
    for d in range(n):
        output_states.append(results.get("state" + str(d)).fetch_all())
    timing = results.timing.fetch_all()["value"]
    print(timing[-1])

    counts = {} # Dictionary containing statistics of measurement of each bitstring obtained
    for i in range(2 ** n):
        counts[generate_binary(n)[i]] = 0

    for i in range(Nshots):  # build statistics by picking line by line the bistrings obtained in measurements
        bitstring = ""
        for j in range(len(output_states)):
            bitstring += str(output_states[j][i])
        counts[bitstring] += 1

    avr_C = 0

    for bitstr in list(counts.keys()):  # Computation of expectation value according to simulation results

        # use sampled bit string x to compute C(x)
        x = [int(num) for num in list(bitstr)]
        tmp_eng = cost_function_C(x, G)

        # compute the expectation value and energy distribution
        avr_C = avr_C + counts[bitstr] * tmp_eng

    Mp_sampled = avr_C / Nshots

    return -Mp_sampled  # minus sign because optimizer will minimize the function,
    # to get the maximum value one takes the absolute value of the yielded result of  optimization


# QUA program


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


th = 0.  # Threshold for state discrimination
state_strings = [f"state{i}" for i in range(n)]
I_strings = [f"I{i}" for i in range(n)]
Q_strings = [f"Q{i}" for i in range(n)]
q = [f"q{i}" for i in range(n)]
rr = [f"rr{i}" for i in range(n)]

with program() as QAOA:
    rep = declare(int)  # Iterator for sampling (necessary to have good statistics for expectation value estimation)
    I = declare(fixed)
    Q = declare(fixed)
    state = declare(bool, size=n)
    γ = declare(fixed)
    β = declare(fixed)
    block = declare(int)
    γ_set = declare(fixed, size=p)
    β_set = declare(fixed, size=p)
    Cut = declare(int)
    bitstring_stream = declare_stream()
    with infinite_loop_():
        pause()

        with for_(block, init=0, cond=block < p, update=i + 1):
            pause()
            assign(γ_set[block], IO1)
            assign(β_set[block], IO2)

        with for_(rep, init=0, cond=rep <= Nshots, update=rep + 1):
            # Apply Hadamard on each qubit
            for k in range(n):  # Run what's inside for qubit0, qubit1, ..., qubit4
                Hadamard(q[k])

            with for_each_((γ, β), (γ_set, β_set)):
                #  Cost Hamiltonian evolution
                for edge in G.edges():  # Iterate over the whole connectivity of Graph G
                    ctrl, tgt = q[edge[0]], q[edge[1]]
                    CU1(2 * γ, ctrl, tgt)
                    Rz(-γ, ctrl)
                    Rz(-γ, tgt)

                # Mixing Hamiltonian evolution
                for k in range(n):  # Apply X(β) rotation on each qubit
                    Rx(-2 * β, q[k])

            # Measurement and state determination

            for k in range(n):  # Iterate over each resonator 'CPW' in the config file
                measure("meas_pulse", rr[k], None, dual_demod.full(("integW1", I), ("integW2", Q)))
                assign(state[k], I > th)

                # Active reset, flip the qubit is measured state is 1
                with if_(state[k]):
                    Rx(π, q)

            cost_function_C(Cut, G)

    with stream_processing():
        for i in range(n):
            state_streams[i].buffer(Nshots).save(state_strings[i])


job = QM.execute(QAOA)

# Try out the QAOA for the problem defined by the Maxcut for graph G
ratio, exp, opti_angles = result_optimization()


job.halt()
