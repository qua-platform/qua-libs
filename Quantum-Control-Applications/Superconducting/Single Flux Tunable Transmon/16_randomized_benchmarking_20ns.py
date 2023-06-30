"""
Performs a 1 qubit randomized benchmarking to measure the 1 qubit gate fidelity (works for gates between 20ns and 40ns).
The 40ns limitation is due to the latency of the switch/case statements. To overcome it, the idea is to convert the
random sequence made of Clifford operations into a sequence of single qubit gates (X, Y, X/2...) and play them by pairs.
this way we can have gap-less RB with gates as short as 20ns.
The drawback is that the max depth is currently limited to 2600 due to data memory.
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from scipy.optimize import curve_fit
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from macros import readout_macro

# Single qubit Clifford operations
c1_ops = [  # Clifford operations
    ("I",),
    ("x180",),
    ("y180",),
    ("y180", "x180"),
    ("x90", "y90"),
    ("x90", "-y90"),
    ("-x90", "y90"),
    ("-x90", "-y90"),
    ("y90", "x90"),
    ("y90", "-x90"),
    ("-y90", "x90"),
    ("-y90", "-x90"),
    ("x90",),
    ("-x90",),
    ("y90",),
    ("-y90",),
    ("-x90", "y90", "x90"),
    ("-x90", "-y90", "x90"),
    ("x180", "y90"),
    ("x180", "-y90"),
    ("y180", "x90"),
    ("y180", "-x90"),
    ("x90", "y90", "x90"),
    ("-x90", "y90", "-x90"),
]
# Single qubit gates
single_qubit_gates = [
    "I",
    "x180",
    "x90",
    "-x90",
    "y180",
    "y90",
    "-y90",
]
# Pairs of single qubit gates
single_qubit_gate_pairs = []
for i in range(len(single_qubit_gates)):
    for j in range(len(single_qubit_gates)):
        single_qubit_gate_pairs.append(((single_qubit_gates[i],) + (single_qubit_gates[j],)))

##############################
# Program-specific variables #
##############################
state_discrimination = False
inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]
max_circuit_depth = (
    2600  # Data memory limit is 16k and we need total vector size of 6*max_circuit_depth --> max_circuit_depth<=2600
)
num_of_sequences = 100
n_avg = 100
seed = 345323
cooldown_time = 5 * qubit_T1 // 4
delta_clifford = 50  # must be > 1
assert (max_circuit_depth / delta_clifford).is_integer(), "max_circuit_depth / delta_clifford must be an integer."


def power_law(power, a, b, p):
    return a * (p**power) + b


def single_gate_indices_from_clifford(clifford_index):
    """
    Return the indices of the single-qubit gates used in the specified Clifford gate.

    :param clifford_index: index of the Clifford gate
    :return: list containing the indices of the single-qubit gates
    """
    out = []
    for gate in c1_ops[clifford_index]:
        out.append(single_qubit_gates.index(gate))
    return out


def pairs_of_gate_indices_from_single(ind1, ind2):
    """
    Returns the index of the single-qubit-gate pait corresponding to the two single-qubit gate indices provided.

    :param ind1: index of the first single-qubit gate
    :param ind2: index of the second single-qubit gate
    :return: index of the corresponding single-qubit-gate pair
    """
    for pair in single_qubit_gate_pairs:
        if pair[0] == single_qubit_gates[ind1] and pair[1] == single_qubit_gates[ind2]:
            return single_qubit_gate_pairs.index(pair)


def from_clifford_to_single(clifford_seq, single_qubit_seq, single_qubit_ind):
    """
    Convert a sequence of single-qubit Clifford gates into a sequence of single-qubit gates.

    :param clifford_seq: a QUA array for the sequence of clifford gates.
    :param single_qubit_seq: a QUA array for the sequence of single-qubit gates.
    :param single_qubit_ind: a QUA int for the length of the single-qubit-gate sequence.
    :return: The single-qubit-gate sequence and its length
    """
    with switch_(clifford_seq):
        for clifford_index in [0, 1, 2, 12, 13, 14, 15]:
            with case_(clifford_index):
                assign(single_qubit_seq[single_qubit_ind], single_gate_indices_from_clifford(clifford_index)[0])
                assign(single_qubit_ind, single_qubit_ind + 1)
        for clifford_index in [3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21]:
            with case_(clifford_index):
                assign(single_qubit_seq[single_qubit_ind], single_gate_indices_from_clifford(clifford_index)[0])
                assign(single_qubit_seq[single_qubit_ind + 1], single_gate_indices_from_clifford(clifford_index)[1])
                assign(single_qubit_ind, single_qubit_ind + 2)
        for clifford_index in [16, 17, 22, 23]:
            with case_(clifford_index):
                assign(single_qubit_seq[single_qubit_ind], single_gate_indices_from_clifford(clifford_index)[0])
                assign(single_qubit_seq[single_qubit_ind + 1], single_gate_indices_from_clifford(clifford_index)[1])
                assign(single_qubit_seq[single_qubit_ind + 2], single_gate_indices_from_clifford(clifford_index)[2])
                assign(single_qubit_ind, single_qubit_ind + 3)

    return single_qubit_seq, single_qubit_ind


def generate_sequence():
    """
    Generates the random sequence of single-qubit Clifford gates, convert it as a sequence of single-qubit gates and regroup these gates by pairs.
    """
    print("Compute sequence...")
    # Random single-qubit Clifford gate sequence variables
    cayley = declare(int, value=list(c1_table.flatten()))  # Load the Cayley table in QUA
    inv_list = declare(int, value=inv_gates)  # Load the list of invert gates in QUA
    sequence = declare(int, size=max_circuit_depth + 1)  # The random Clifford sequence
    inv_gate = declare(
        int, size=2 * max_circuit_depth + 1
    )  # The list containing the recovery gates to play at each depth
    rand = Random(seed=seed)

    current_state = declare(int)
    step = declare(int)
    i = declare(int)

    # Conversion to single-qubit gate sequence variables
    sequence_single = declare(int, size=2 * max_circuit_depth + 1)  # The sequence of single-qubit gates
    sequence_single_len = declare(int)  # Length of the current single-qubit gate sequence
    sequence_single_len_prev = declare(int)  # Length of the current single-qubit gate sequence

    # Conversion to pairs single-qubit gate sequence variables
    # sequence_pairs = declare(int, size=max_circuit_depth) --> Replaced by sequence to save memory
    sequence_pairs_lengths = declare(int, size=max_circuit_depth)
    even = declare(bool)
    sequence_pairs_len = declare(int)
    last = declare(int)
    first = declare(int)
    end_point = declare(int)
    ii = declare(int)

    # Recovery gate
    recovery_single = declare(int, size=4)  # Single-qubit gates defining the recovery sequence
    recovery_index_single = declare(int)  # Current index of the recovery sequence made of single-qubit gates
    # recovery_pairs = declare(int, size=2 * max_circuit_depth)  # Pairs of single-qubit gates defining the recovery sequence  --> Replaced by inv_gate to save memory
    recovery_index_pairs = declare(int)  # Current index of the recovery sequence made of pairs of single-qubit gates
    assign(recovery_index_pairs, 0)

    assign(current_state, 0)
    assign(sequence_single_len, 0)
    assign(sequence_pairs_len, 0)
    assign(even, True)
    with for_(i, 0, i < max_circuit_depth, i + 1):
        ## Get the random sequence of Clifford gates
        # ------------------------------------------
        assign(step, rand.rand_int(24))  # Get a random number
        assign(
            current_state, cayley[current_state * 24 + step]
        )  # Find the current state based on the random index and the Cayley table
        assign(sequence[i], step)  # Update the Clifford sequence with the random index
        assign(inv_gate[2 * i], inv_list[current_state])  # Update the recovery gate to play after this random gate

        ## Split this sequence into single-qubit gates
        # --------------------------------------------
        sequence_single, sequence_single_len = from_clifford_to_single(
            sequence[i], sequence_single, sequence_single_len
        )

        ## Regroup the single-qubit gates into pairs
        # ------------------------------------------
        # Get the index of the first single-qubit gate of the sequence
        with if_(i == 0):
            assign(first, 0)
        with else_():
            assign(first, sequence_single_len_prev)
        # Get the index of the last single-qubit gate of the sequence.
        # If the sequence is odd, then the last gate will be played with the recovery gate
        assign(end_point, (sequence_single_len >> 1) << 1)

        # If previous sequence was odd, then start by adding a pair with the last gate (last gate, first new gate)
        with if_(~even):
            for j in range(len(single_qubit_gates)):
                with if_(last == j):
                    for k in range(len(single_qubit_gates)):
                        with if_(sequence_single[sequence_single_len_prev] == k):
                            assign(sequence[sequence_pairs_len], pairs_of_gate_indices_from_single(j, k))
                            assign(sequence_pairs_len, sequence_pairs_len + 1)
            assign(first, sequence_single_len_prev + 1)  # Shift the index of the first gate by one

        # Check if we are dealing with an odd or even number of single-qubit gates
        with if_(sequence_single_len == (sequence_single_len >> 1) << 1):
            assign(even, True)
        with else_():
            assign(even, False)
            # Get the last single-qubit gate of the current sequence which will be the first gate of the next sequence
            assign(last, sequence_single[sequence_single_len - 1])

        # Construct the sequence made of pairs of single-qubit gates
        with for_(ii, first, ii < end_point, ii + 2):  # Loop over the single-qubit-gate sequence
            for j in range(len(single_qubit_gates)):
                with if_(sequence_single[ii] == j):
                    for k in range(len(single_qubit_gates)):
                        with if_(sequence_single[ii + 1] == k):  # Get the corresponding pair as (j, k)
                            assign(sequence[sequence_pairs_len], pairs_of_gate_indices_from_single(j, k))
                            assign(sequence_pairs_len, sequence_pairs_len + 1)
        # Get the length of the sequence of pairs of single-qubit gates for each depth
        assign(sequence_pairs_lengths[i], sequence_pairs_len)

        #### Recovery gate
        # Initialize the indices and sequences
        assign(recovery_index_single, 0)
        assign(recovery_single[0], 0)
        assign(recovery_single[1], 0)
        assign(recovery_single[2], 0)
        assign(recovery_single[3], 0)
        # Convert the recovery sequence from Clifford gates to single-qubit gates
        recovery_single, recovery_index_single = from_clifford_to_single(
            inv_gate[2 * i], recovery_single, recovery_index_single
        )
        # If the current sequence has an even number of gates, then add a new pair for the recovery sequence
        with if_(even):
            with for_(ii, 0, ii < recovery_index_single, ii + 2):
                for j in range(len(single_qubit_gates)):
                    with if_(recovery_single[ii] == j):
                        for k in range(len(single_qubit_gates)):
                            with if_(recovery_single[ii + 1] == k):
                                assign(inv_gate[recovery_index_pairs], pairs_of_gate_indices_from_single(j, k))
                                assign(recovery_index_pairs, recovery_index_pairs + 1)
        # If the current sequence has an odd number of gates, then start the recovery sequence with the last gate
        with else_():
            for j in range(len(single_qubit_gates)):
                with if_(last == j):
                    for k in range(len(single_qubit_gates)):
                        with if_(recovery_single[0] == k):
                            assign(inv_gate[recovery_index_pairs], pairs_of_gate_indices_from_single(j, k))
                            assign(recovery_index_pairs, recovery_index_pairs + 1)
            with for_(ii, 1, ii < recovery_index_single, ii + 2):
                for j in range(len(single_qubit_gates)):
                    with if_(recovery_single[ii] == j):
                        for k in range(len(single_qubit_gates)):
                            with if_(recovery_single[ii + 1] == k):
                                assign(inv_gate[recovery_index_pairs], pairs_of_gate_indices_from_single(j, k))
                                assign(recovery_index_pairs, recovery_index_pairs + 1)
        # Make sure that the recovery sequence always contains two pairs of gates
        with if_(recovery_index_pairs > ((recovery_index_pairs >> 1) << 1)):
            assign(inv_gate[recovery_index_pairs], 0)
            assign(recovery_index_pairs, recovery_index_pairs + 1)

        assign(sequence_single_len_prev, sequence_single_len)
    return sequence, sequence_pairs_lengths, inv_gate


def play_sequence(sequence_list, number_of_gates):
    i = declare(int)
    with for_(i, 0, i < number_of_gates, i + 1):
        with switch_(sequence_list[i], unsafe=True):
            for ii in range(len(single_qubit_gate_pairs)):
                with case_(ii):
                    for iii in range(len(single_qubit_gate_pairs[ii])):
                        if single_qubit_gate_pairs[ii][iii] == "I":
                            wait(x180_len // 4, "qubit")
                        else:
                            play(single_qubit_gate_pairs[ii][iii], "qubit")


# ###################
# # The QUA program #
# ###################
with program() as rb:
    depth = declare(int)  # Current depth
    depth_target = declare(int)  # Play the sequence every depth target
    saved_gates = declare(int, size=2)  # Gates that will be replaced by the recovery gates for each depth
    random_sequence_index = declare(int)  # Index of the current random sequence
    n = declare(int)  # Averaging index
    I = declare(fixed)  # I quadrature from the measurement
    Q = declare(fixed)  # Q quadrature from the measurement
    state = declare(bool)  # Qubit state if state discrimination
    seq_length = declare(int)  # Length of the current sequence
    # Declare the streams
    rnd_seq_ind_st = declare_stream()
    if state_discrimination:
        state_st = declare_stream()
    else:
        I_st = declare_stream()
        Q_st = declare_stream()

    # Loop over the different random sequences
    with for_(random_sequence_index, 0, random_sequence_index < num_of_sequences, random_sequence_index + 1):
        save(random_sequence_index, rnd_seq_ind_st)
        # Get the random sequence
        sequence_pairs, sequence_pairs_lengths, recovery_pairs = generate_sequence()
        # Initialize depth_target
        assign(depth_target, 0)
        # Loop over the circuit depths
        with for_(depth, 1, depth <= max_circuit_depth, depth + 1):
            # Play the sequence every depth target
            with if_((depth == 1) | (depth == depth_target)):
                # Replacing the last gates in the sequence with the sequence's inverse gates
                # The original gates are saved in 'saved_gates' and are being restored at the end
                assign(saved_gates[0], sequence_pairs[sequence_pairs_lengths[depth - 1]])
                assign(saved_gates[1], sequence_pairs[sequence_pairs_lengths[depth - 1] + 1])

                # Get the recovery gate to play at the end of this sequence which can be made of 1, 2 or 3 single-qubit gates
                assign(sequence_pairs[sequence_pairs_lengths[depth - 1]], recovery_pairs[2 * depth - 2])
                assign(sequence_pairs[sequence_pairs_lengths[depth - 1] + 1], recovery_pairs[2 * depth - 1])
                assign(seq_length, sequence_pairs_lengths[depth - 1] + 2)

                # Averaging loop
                with for_(n, 0, n < n_avg, n + 1):
                    # Can replace by active reset
                    wait(cooldown_time, "resonator", "qubit")
                    with strict_timing_():
                        play_sequence(sequence_pairs, seq_length)
                    align("resonator", "qubit")
                    # Make sure you updated the ge_threshold
                    state, I, Q = readout_macro(threshold=ge_threshold, state=state, I=I, Q=Q)
                    if state_discrimination:
                        save(state, state_st)
                    else:
                        save(I, I_st)
                        save(Q, Q_st)

                # Update the depth target to the next desired value
                assign(depth_target, depth_target + delta_clifford)
                # Restore the original gates
                assign(sequence_pairs[sequence_pairs_lengths[depth - 1]], saved_gates[0])
                assign(sequence_pairs[sequence_pairs_lengths[depth - 1] + 1], saved_gates[1])

    with stream_processing():
        rnd_seq_ind_st.save("iteration")
        # return a 1D array of averaged random pulse sequences vs circuit depth
        if state_discrimination:
            state_st.boolean_to_int().buffer(n_avg).map(FUNCTIONS.average()).buffer(
                max_circuit_depth // delta_clifford + 1
            ).average().save("state_avg")
        else:
            I_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(max_circuit_depth // delta_clifford + 1).average().save(
                "I"
            )
            Q_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(max_circuit_depth // delta_clifford + 1).average().save(
                "Q"
            )


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip, qop_port)

simulate = False

if simulate:
    simulation_config = SimulationConfig(duration=100_000)  # in clock cycles
    job = qmm.simulate(config, rb, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(rb)
    # Get results from QUA program
    if state_discrimination:
        results = fetching_tool(job, data_list=["state_avg", "iteration"], mode="live")
    else:
        results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    # data analysis
    x = np.arange(0, max_circuit_depth + 0.1, delta_clifford)
    x[0] = 1  # to set the first value of 'x' to be depth = 1 as in the experiment
    while results.is_processing():
        # data analysis
        if state_discrimination:
            state_avg, iteration = results.fetch_all()
            value_avg = 1 - state_avg
            error_avg = np.std(state_avg)
        else:
            I, Q, iteration = results.fetch_all()
            value_avg = np.sqrt(I**2 + Q**2)
            error_avg = np.std(np.sqrt(I**2 + Q**2))

        # Progress bar
        progress_counter(iteration, num_of_sequences, start_time=results.get_start_time())
        # Plot results
        plt.cla()
        plt.errorbar(x, value_avg, yerr=error_avg, marker=".")
        pars, cov = curve_fit(
            f=power_law,
            xdata=x,
            ydata=value_avg,
            p0=[0.5, 0.5, 0.9],
            bounds=(-np.inf, np.inf),
            maxfev=2000,
        )
        plt.plot(x, power_law(x, *pars), linestyle="--", linewidth=2)
        plt.xlabel("Number of Clifford gates")
        plt.ylabel("Sequence Fidelity")
        plt.title("Single qubit RB")
        plt.pause(0.1)

        stdevs = np.sqrt(np.diag(cov))

    print("#########################")
    print("### Fitted Parameters ###")
    print("#########################")
    print(f"A = {pars[0]:.3} ({stdevs[0]:.1}), B = {pars[1]:.3} ({stdevs[1]:.1}), p = {pars[2]:.3} ({stdevs[2]:.1})")
    print("Covariance Matrix")
    print(cov)

    one_minus_p = 1 - pars[2]
    r_c = one_minus_p * (1 - 1 / 2**1)
    r_g = r_c / 1.875  # 1.875 is the average number of gates in clifford operation
    r_c_std = stdevs[2] * (1 - 1 / 2**1)
    r_g_std = r_c_std / 1.875

    print("#########################")
    print("### Useful Parameters ###")
    print("#########################")
    print(
        f"Error rate: 1-p = {np.format_float_scientific(one_minus_p, precision=2)} ({stdevs[2]:.1})\n"
        f"Clifford set infidelity: r_c = {np.format_float_scientific(r_c, precision=2)} ({r_c_std:.1})\n"
        f"Gate infidelity: r_g = {np.format_float_scientific(r_g, precision=2)}  ({r_g_std:.1})"
    )

    # np.savez("rb_values", value)
