"""
        SINGLE QUBIT INTERLEAVED RANDOMIZED BENCHMARKING (for gates >= 20ns and < 40ns)
Because of the latency of the switch/case commands (40ns), we cannot play a random sequence of gates shorter than 40ns
using the standard RB script.
The trick is to convert the random sequence made of Clifford operations into a sequence of single qubit gates
(X, Y, X/2...) and play them by pairs. This way we can have gap-less RB with gates as short as 20ns, because 20ns+20ns=40ns.
The drawback is that the max depth is currently limited to 1000 Clifford gates due to data memory.

Here again, each random sequence is derived on the FPGA for the maximum depth (specified as an input) and played for each depth
asked by the user (the sequence is truncated to the desired depth). Each truncated sequence ends with the recovery gate,
found at each step thanks to a preloaded lookup table (Cayley table), that will bring the qubit back to its ground state.
In this version, a Clifford gate chosen by the user is interleaved between each random gate in the sequence. This allows
to characterize the fidelity of a specific gate.

If the readout has been calibrated and is good enough, then state discrimination can be applied to only return the state
of the qubit. Otherwise, the 'I' and 'Q' quadratures are returned.
Each sequence is played n_avg times for averaging. A second averaging is performed by playing different random sequences.

The data is then post-processed to extract the single-qubit gate fidelity and error per gate
.
Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - Having the qubit frequency perfectly calibrated (ramsey).
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Set the desired flux bias.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from macros import readout_macro
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


##############################
# Program-specific variables #
##############################
num_of_sequences = 50  # Number of random sequences
n_avg = 20  # Number of averaging loops for each random sequence
max_circuit_depth = 1000  # Maximum circuit depth < 1000
delta_clifford = 10  #  Play each sequence with a depth step equals to 'delta_clifford - Must be > 1
assert (max_circuit_depth / delta_clifford).is_integer(), "max_circuit_depth / delta_clifford must be an integer."
seed = 345324  # Pseudo-random number generator seed
# Flag to enable state discrimination if the readout has been calibrated (rotated blobs and threshold)
state_discrimination = False
# List of recovery gates from the lookup table
inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]
# index of the gate to interleave from the play_sequence() function defined below
# Correspondence table:
#  0: identity |  1: x180 |  2: y180
# 12: x90      | 13: -x90 | 14: y90 | 15: -y90 |
interleaved_gate_index = 2


###################################
# Helper functions and QUA macros #
###################################
# Single qubit Clifford operations
c1_ops = [
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


def get_interleaved_gate(gate_index):
    if gate_index == 0:
        return "I"
    elif gate_index == 1:
        return "x180"
    elif gate_index == 2:
        return "y180"
    elif gate_index == 12:
        return "x90"
    elif gate_index == 13:
        return "-x90"
    elif gate_index == 14:
        return "y90"
    elif gate_index == 15:
        return "-y90"


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


def generate_sequence(interleaved_gate_index):
    """
    Generates the random sequence of single-qubit Clifford gates, convert it as a sequence of single-qubit gates and regroup these gates by pairs.
    """
    print("Compute sequence...")
    # Random single-qubit Clifford gate sequence variables
    cayley = declare(int, value=list(c1_table.flatten()))  # Load the Cayley table in QUA
    inv_list = declare(int, value=inv_gates)  # Load the list of invert gates in QUA
    sequence = declare(int, size=2 * max_circuit_depth + 1)  # The random Clifford sequence
    inv_gate = declare(
        int, size=2 * max_circuit_depth + 1
    )  # The list containing the recovery gates to play at each depth
    rand = Random(seed=seed)

    current_state = declare(int)
    step = declare(int)
    i = declare(int)

    # Conversion to single-qubit gate sequence variables
    sequence_single = declare(int, size=2 * 2 * max_circuit_depth + 1)  # The sequence of single-qubit gates
    sequence_single_len = declare(int)  # Length of the current single-qubit gate sequence
    sequence_single_len_prev = declare(int)  # Length of the current single-qubit gate sequence

    # Conversion to pairs single-qubit gate sequence variables
    sequence_pairs_lengths = declare(int, size=2 * max_circuit_depth)
    even = declare(bool)
    sequence_pairs_len = declare(int)
    last = declare(int)
    first = declare(int)
    end_point = declare(int)
    ii = declare(int)

    # Recovery gate
    recovery_single = declare(int, size=4)  # Single-qubit gates defining the recovery sequence
    recovery_index_single = declare(int)  # Current index of the recovery sequence made of single-qubit gates
    recovery_pairs = declare(
        int, size=2 * 2 * max_circuit_depth
    )  # Pairs of single-qubit gates defining the recovery sequence
    recovery_index_pairs = declare(int)  # Current index of the recovery sequence made of pairs of single-qubit gates
    assign(recovery_index_pairs, 0)

    assign(current_state, 0)
    assign(sequence_single_len, 0)
    assign(sequence_pairs_len, 0)
    assign(even, True)
    with for_(i, 0, i < 2 * max_circuit_depth, i + 2):
        ## Get the random sequence of Clifford gates
        # ------------------------------------------
        assign(step, rand.rand_int(24))  # Get a random number
        assign(
            current_state, cayley[current_state * 24 + step]
        )  # Find the current state based on the random index and the Cayley table
        assign(sequence[i], step)  # Update the Clifford sequence with the random index
        assign(inv_gate[i], inv_list[current_state])  # Update the recovery gate to play after this random gate
        # interleaved gate
        assign(step, interleaved_gate_index)
        assign(current_state, cayley[current_state * 24 + step])
        assign(sequence[i + 1], step)
        assign(inv_gate[i + 1], inv_list[current_state])

    with for_(i, 0, i < 2 * max_circuit_depth, i + 1):
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
            inv_gate[i], recovery_single, recovery_index_single
        )
        # If the current sequence has an even number of gates, then add a new pair for the recovery sequence
        with if_(even):
            with for_(ii, 0, ii < recovery_index_single, ii + 2):
                for j in range(len(single_qubit_gates)):
                    with if_(recovery_single[ii] == j):
                        for k in range(len(single_qubit_gates)):
                            with if_(recovery_single[ii + 1] == k):
                                assign(recovery_pairs[recovery_index_pairs], pairs_of_gate_indices_from_single(j, k))
                                assign(recovery_index_pairs, recovery_index_pairs + 1)
        # If the current sequence has an odd number of gates, then start the recovery sequence with the last gate
        with else_():
            for j in range(len(single_qubit_gates)):
                with if_(last == j):
                    for k in range(len(single_qubit_gates)):
                        with if_(recovery_single[0] == k):
                            assign(recovery_pairs[recovery_index_pairs], pairs_of_gate_indices_from_single(j, k))
                            assign(recovery_index_pairs, recovery_index_pairs + 1)
            with for_(ii, 1, ii < recovery_index_single, ii + 2):
                for j in range(len(single_qubit_gates)):
                    with if_(recovery_single[ii] == j):
                        for k in range(len(single_qubit_gates)):
                            with if_(recovery_single[ii + 1] == k):
                                assign(recovery_pairs[recovery_index_pairs], pairs_of_gate_indices_from_single(j, k))
                                assign(recovery_index_pairs, recovery_index_pairs + 1)
        # Make sure that the recovery sequence always contains two pairs of gates
        with if_(recovery_index_pairs > ((recovery_index_pairs >> 1) << 1)):
            assign(recovery_pairs[recovery_index_pairs], 0)
            assign(recovery_index_pairs, recovery_index_pairs + 1)

        assign(sequence_single_len_prev, sequence_single_len)
    return sequence, sequence_pairs_lengths, recovery_pairs


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


###################
# The QUA program #
###################
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
        sequence_pairs, sequence_pairs_lengths, recovery_pairs = generate_sequence(interleaved_gate_index)
        # Initialize depth_target
        assign(depth_target, 0)
        # Loop over the circuit depths
        with for_(depth, 1, depth <= 2 * max_circuit_depth, depth + 1):
            # Play the sequence every depth target
            with if_((depth == 2) | (depth == depth_target)):
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
                    # Can be replaced by active reset
                    wait(thermalization_time * u.ns, "resonator")
                    # Align the two elements to play the sequence after qubit initialization
                    align("resonator", "qubit")
                    # The strict_timing ensures that the sequence will be played without gaps
                    with strict_timing_():
                        play_sequence(sequence_pairs, seq_length)
                    # Align the two elements to measure after playing the circuit.
                    align("qubit", "resonator")
                    # Make sure you updated the ge_threshold and angle if you want to use state discrimination
                    state, I, Q = readout_macro(threshold=ge_threshold, state=state, I=I, Q=Q)
                    # Save the results to their respective streams
                    if state_discrimination:
                        save(state, state_st)
                    else:
                        save(I, I_st)
                        save(Q, Q_st)

                # Update the depth target to the next desired value
                assign(depth_target, depth_target + 2 * delta_clifford)
                # Restore the original gates
                assign(sequence_pairs[sequence_pairs_lengths[depth - 1]], saved_gates[0])
                assign(sequence_pairs[sequence_pairs_lengths[depth - 1] + 1], saved_gates[1])

    with stream_processing():
        rnd_seq_ind_st.save("iteration")
        if state_discrimination:
            # saves a 2D array of depth and random pulse sequences in order to get error bars along the random sequences
            state_st.boolean_to_int().buffer(n_avg).map(FUNCTIONS.average()).buffer(
                max_circuit_depth / delta_clifford + 1
            ).buffer(num_of_sequences).save("state")
            # returns a 1D array of averaged random pulse sequences vs depth of circuit for live plotting
            state_st.boolean_to_int().buffer(n_avg).map(FUNCTIONS.average()).buffer(
                max_circuit_depth / delta_clifford + 1
            ).average().save("state_avg")
        else:
            I_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(max_circuit_depth / delta_clifford + 1).buffer(
                num_of_sequences
            ).save("I")
            Q_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(max_circuit_depth / delta_clifford + 1).buffer(
                num_of_sequences
            ).save("Q")
            I_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(max_circuit_depth / delta_clifford + 1).average().save(
                "I_avg"
            )
            Q_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(max_circuit_depth / delta_clifford + 1).average().save(
                "Q_avg"
            )


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=100_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, rb, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(rb)
    # Get results from QUA program
    if state_discrimination:
        results = fetching_tool(job, data_list=["state_avg", "iteration"], mode="live")
    else:
        results = fetching_tool(job, data_list=["I_avg", "Q_avg", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    # data analysis
    x = np.arange(0, 2 * max_circuit_depth + 0.1, 2 * delta_clifford)
    x[0] = 1  # to set the first value of 'x' to be depth = 1 as in the experiment
    while results.is_processing():
        # data analysis
        if state_discrimination:
            state_avg, iteration = results.fetch_all()
            value_avg = state_avg
        else:
            I, Q, iteration = results.fetch_all()
            value_avg = I

        # Progress bar
        progress_counter(iteration, num_of_sequences, start_time=results.get_start_time())
        # Plot averaged values
        plt.cla()
        plt.plot(x, value_avg, marker=".")
        plt.xlabel("Number of Clifford gates")
        plt.ylabel("Sequence Fidelity")
        plt.title(f"Single qubit interleaved RB {get_interleaved_gate(interleaved_gate_index)}")
        plt.pause(0.1)

    # At the end of the program, fetch the non-averaged results to get the error-bars
    if state_discrimination:
        results = fetching_tool(job, data_list=["state"])
        state = results.fetch_all()[0]
        value_avg = np.mean(state, axis=0)
        error_avg = np.std(state, axis=0)
    else:
        results = fetching_tool(job, data_list=["I", "Q"])
        I, Q = results.fetch_all()
        value_avg = np.mean(I, axis=0)
        error_avg = np.std(I, axis=0)
    # data analysis
    pars, cov = curve_fit(
        f=power_law,
        xdata=x,
        ydata=value_avg,
        p0=[0.5, 0.5, 0.9],
        bounds=(-np.inf, np.inf),
        maxfev=2000,
    )
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

    # Plots
    plt.figure()
    plt.errorbar(x, value_avg, yerr=error_avg, marker=".")
    plt.plot(x, power_law(x, *pars), linestyle="--", linewidth=2)
    plt.xlabel("Number of Clifford gates")
    plt.ylabel("Sequence Fidelity")
    plt.title(f"Single qubit interleaved RB {get_interleaved_gate(interleaved_gate_index)}")

    # np.savez("rb_values", value)
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
