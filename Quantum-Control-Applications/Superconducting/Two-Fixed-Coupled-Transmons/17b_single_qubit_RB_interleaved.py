"""
        SINGLE QUBIT INTERLEAVED RANDOMIZED BENCHMARKING
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
from macros import multiplexed_readout, qua_declaration, active_reset
import math
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
num_of_sequences = 10  # Number of random sequences
n_avg = 3  # Number of averaging loops for each random sequence
max_circuit_depth = 20  # Maximum circuit depth
delta_clifford = 2  #  Play each sequence with a depth step equals to 'delta_clifford - Must be > 1
seed = 345324  # Pseudo-random number generator seed
# List of recovery gates from the lookup table
inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]
# index of the gate to interleave from the play_sequence() function defined below
# Correspondence table:
#  0: identity |  1: x180 |  2: y180
# 12: x90      | 13: -x90 | 14: y90 | 15: -y90 |
interleaved_gate_index = 0

# Assertion
assert (max_circuit_depth / delta_clifford).is_integer(), "max_circuit_depth / delta_clifford must be an integer."

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "config": config,
    "num_of_sequences": num_of_sequences,
    "delta_clifford": delta_clifford,
    "max_circuit_depth": max_circuit_depth,
    "interleaved_gate_index": interleaved_gate_index,
}


###################################
# Helper functions and QUA macros #
###################################
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


def generate_sequence(interleaved_gate_index):
    cayley = declare(int, value=c1_table.flatten().tolist())
    inv_list = declare(int, value=inv_gates)
    current_state = declare(int)
    step = declare(int)
    sequence = declare(int, size=2 * max_circuit_depth + 1)
    inv_gate = declare(int, size=2 * max_circuit_depth + 1)
    i = declare(int)
    rand = Random(seed=seed)

    assign(current_state, 0)
    with for_(i, 0, i < 2 * max_circuit_depth, i + 2):
        assign(step, rand.rand_int(24))
        assign(current_state, cayley[current_state * 24 + step])
        assign(sequence[i], step)
        assign(inv_gate[i], inv_list[current_state])
        # interleaved gate
        assign(step, interleaved_gate_index)
        assign(current_state, cayley[current_state * 24 + step])
        assign(sequence[i + 1], step)
        assign(inv_gate[i + 1], inv_list[current_state])

    return sequence, inv_gate


def play_sequence(sequence_list, depth, qb):
    i = declare(int)
    with for_(i, 0, i <= depth, i + 1):
        with switch_(sequence_list[i], unsafe=True):
            with case_(0):
                wait(pi_len >> 2, qb)
            with case_(1):
                play("x180", qb)
            with case_(2):
                play("y180", qb)
            with case_(3):
                play("y180", qb)
                play("x180", qb)
            with case_(4):
                play("x90", qb)
                play("y90", qb)
            with case_(5):
                play("x90", qb)
                play("-y90", qb)
            with case_(6):
                play("-x90", qb)
                play("y90", qb)
            with case_(7):
                play("-x90", qb)
                play("-y90", qb)
            with case_(8):
                play("y90", qb)
                play("x90", qb)
            with case_(9):
                play("y90", qb)
                play("-x90", qb)
            with case_(10):
                play("-y90", qb)
                play("x90", qb)
            with case_(11):
                play("-y90", qb)
                play("-x90", qb)
            with case_(12):
                play("x90", qb)
            with case_(13):
                play("-x90", qb)
            with case_(14):
                play("y90", qb)
            with case_(15):
                play("-y90", qb)
            with case_(16):
                play("-x90", qb)
                play("y90", qb)
                play("x90", qb)
            with case_(17):
                play("-x90", qb)
                play("-y90", qb)
                play("x90", qb)
            with case_(18):
                play("x180", qb)
                play("y90", qb)
            with case_(19):
                play("x180", qb)
                play("-y90", qb)
            with case_(20):
                play("y180", qb)
                play("x90", qb)
            with case_(21):
                play("y180", qb)
                play("-x90", qb)
            with case_(22):
                play("x90", qb)
                play("y90", qb)
                play("x90", qb)
            with case_(23):
                play("-x90", qb)
                play("y90", qb)
                play("-x90", qb)


###################
#   QUA Program   #
###################

with program() as PROGRAM:
    depth = declare(int)  # QUA variable for the varying depth
    depth_target = declare(int)  # QUA variable for the current depth (changes in steps of delta_clifford)
    # QUA variable to store the last Clifford gate of the current sequence which is replaced by the recovery gate
    saved_gate = declare(int)
    m = declare(int)  # QUA variable for the loop over random sequences
    m_st = declare_stream()
    I, I_st, Q, Q_st, n, _ = qua_declaration(nb_of_qubits=2)
    state = [declare(bool) for _ in range(2)]
    state_st = [declare_stream() for _ in range(2)]

    with for_(m, 0, m < num_of_sequences, m + 1):  # QUA for_ loop over the random sequences
        # Save the counter for the progress bar
        save(m, m_st)
        # Generates the RB sequence with a gate interleaved after each Clifford
        sequence_list, inv_gate_list = generate_sequence(interleaved_gate_index=interleaved_gate_index)
        # Depth_target is used to always play the gates by pairs [(random_gate-interleaved_gate)^depth/2-inv_gate]
        assign(depth_target, 0)

        with for_(depth, 1, depth <= 2 * max_circuit_depth, depth + 1):
            # Replacing the last gate in the sequence with the sequence's inverse gate
            # The original gate is saved in 'saved_gate' and is being restored at the end
            assign(saved_gate, sequence_list[depth])
            assign(sequence_list[depth], inv_gate_list[depth - 1])
            # Only played the depth corresponding to target_depth

            with if_((depth == 2) | (depth == depth_target)):

                with for_(n, 0, n < n_avg, n + 1):  # Averaging loop
                    # Can replace by active reset
                    wait(thermalization_time * u.ns)
                    # Align the two elements to play the sequence after qubit initialization
                    align()
                    # The strict_timing ensures that the sequence will be played without gaps
                    play_sequence(sequence_list, depth, "q1_xy")
                    play_sequence(sequence_list, depth, "q2_xy")
                    # Align the two elements to measure after playing the circuit.
                    align()
                    multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2], weights="rotated_")
                # always play the random gate followed by the interleaved gate. The factor of 2 is there to always
                # play the gates by pairs [(random_gate-interleaved_gate)^depth/2-inv_gate]
                assign(depth_target, depth_target + 2 * delta_clifford)
            # Reset the last gate of the sequence back to the original Clifford gate
            # (that was replaced by the recovery gate at the beginning)
            assign(sequence_list[depth], saved_gate)

    with stream_processing():
        m_st.save("iteration")
        for ind in range(2):
            I_st[ind].buffer(n_avg).map(FUNCTIONS.average()).buffer(
                max_circuit_depth / delta_clifford + 1
            ).average().save(f"I{ind+1}")
            Q_st[ind].buffer(n_avg).map(FUNCTIONS.average()).buffer(
                max_circuit_depth / delta_clifford + 1
            ).average().save(f"Q{ind+1}")

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
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, PROGRAM, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.plot(block=False)
else:
    try:  # Open the quantum machine
        qm = qmm.open_qm(config)

        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(PROGRAM, flags=["not-strict-timing"])

        # Prepare the figure for live plotting
        fig = plt.figure()
        interrupt_on_close(fig, job)
        # Tool to easily fetch results from the OPX (results_handle used in it)
        results = fetching_tool(job, ["iteration", "I1", "Q1", "I2", "Q2"], mode="live")
        # Live plotting

        x = np.arange(0, 2 * max_circuit_depth + 0.1, 2 * delta_clifford)
        x[0] = 1  # to set the first value of 'x' to be depth = 1 as in the experiment

        while results.is_processing():

            # Fetch results
            res = results.fetch_all()
            # Progress bar
            progress_counter(res[0], num_of_sequences, start_time=results.start_time)

            plt.suptitle("interleaved-RB")
            for ind in range(2):

                S = res[2 * ind + 1]
                # Plot
                plt.subplot(1, 2, ind + 1)
                plt.cla()
                plt.plot(x, S, marker=".", color="r")
                plt.xlabel("Number of Clifford gates")
                plt.ylabel("Sequence Fidelity")
                plt.title(f"Qb - {ind}")

            plt.tight_layout()
            plt.pause(2)

        for ind in range(2):
            save_data_dict[f"I{ind}"] = res[2 * ind + 1]
            save_data_dict[f"Q{ind}"] = res[2 * ind + 2]

        # Save results
        script_name = Path(__file__).name
        data_handler = DataHandler(root_data_folder=save_dir)
        save_data_dict.update({"fig_live": fig})
        data_handler.additional_files = {script_name: script_name, **default_additional_files}
        data_handler.save_data(data=save_data_dict, name="rb_interleaved")

    except Exception as e:
        print(f"An exception occurred: {e}")

    finally:
        qm.close()
        print("Experiment QM is now closed")
        plt.show(block=True)
