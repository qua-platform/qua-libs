"""
SINGLE QUBIT RANDOMIZED BENCHMARKING
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
from scipy.optimize import curve_fit
import matplotlib
import time

matplotlib.use("TkAgg")

##################
#   Parameters   #
##################
# Parameters Definition
num_of_sequences = 50  # Number of random sequences
n_avg = 10  # Number of averaging loops for each random sequence
max_circuit_depth = 200  # Maximum circuit depth
delta_clifford = 20  #  Play each sequence with a depth step equals to 'delta_clifford - Must be > 1
seed = 345324  # Pseudo-random number generator seed
# List of recovery gates from the lookup table
inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]

# Assertion
assert (max_circuit_depth / delta_clifford).is_integer(), "max_circuit_depth / delta_clifford must be an integer."

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "num_of_sequences": num_of_sequences,
    "delta_clifford": delta_clifford,
    "max_circuit_depth": max_circuit_depth,
    "config": config,
}


###################################
# Helper functions and QUA macros #
###################################
def power_law(power, a, b, p):
    return a * (p**power) + b


def generate_sequence():
    cayley = declare(int, value=c1_table.flatten().tolist())
    inv_list = declare(int, value=inv_gates)
    current_state = declare(int)
    step = declare(int)
    sequence = declare(int, size=max_circuit_depth + 1)
    inv_gate = declare(int, size=max_circuit_depth + 1)
    i = declare(int)
    rand = Random(seed=seed)

    assign(current_state, 0)
    with for_(i, 0, i < max_circuit_depth, i + 1):
        assign(step, rand.rand_int(24))
        assign(current_state, cayley[current_state * 24 + step])
        assign(sequence[i], step)
        assign(inv_gate[i], inv_list[current_state])

    return sequence, inv_gate


def play_sequence(sequence_list, depth, qb):
    i = declare(int)
    with for_(i, 0, i <= depth, i + 1):
        with switch_(sequence_list[i], unsafe=True):
            with case_(0):
                wait(pi_len // 4, qb)
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
# The QUA program #
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
        sequence_list, inv_gate_list = generate_sequence()  # Generate the random sequence of length max_circuit_depth

        assign(depth_target, 0)  # Initialize the current depth to 0

        with for_(depth, 1, depth <= max_circuit_depth, depth + 1):  # Loop over the depths
            # Replacing the last gate in the sequence with the sequence's inverse gate
            # The original gate is saved in 'saved_gate' and is being restored at the end
            assign(saved_gate, sequence_list[depth])
            assign(sequence_list[depth], inv_gate_list[depth - 1])
            # Only played the depth corresponding to target_depth

            with if_((depth == 1) | (depth == depth_target)):
                with for_(n, 0, n < n_avg, n + 1):  # Averaging loop
                    wait(thermalization_time * u.ns)
                    # Align the two elements to play the sequence after qubit initialization
                    align()
                    # The strict_timing ensures that the sequence will be played without gaps
                    # Play the random sequence of desired depth
                    play_sequence(sequence_list, depth, "q1_xy")
                    play_sequence(sequence_list, depth, "q2_xy")
                    # Align the two elements to measure after playing the circuit.
                    align()
                    multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2], weights="rotated_")

                # Go to the next depth
                assign(depth_target, depth_target + delta_clifford)

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
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, PROGRAM, simulation_config)
    # Get the simulated samples
    samples = job.get_simulated_samples()
    # Plot the simulated samples
    samples.con1.plot()
    # Get the waveform report object
    waveform_report = job.get_simulated_waveform_report()
    # Cast the waveform report to a python dictionary
    waveform_dict = waveform_report.to_dict()
    # Visualize and save the waveform report
    waveform_report.create_plot(samples, plot=True, save_path=str(Path(__file__).resolve()))
else:
    try:
        # Open the quantum machine
        qm = qmm.open_qm(config)

        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(PROGRAM, flags=["not-strict-timing"])

        # Prepare the figure for live plotting
        fig = plt.figure()
        interrupt_on_close(fig, job)
        # Tool to easily fetch results from the OPX (results_handle used in it)
        results = fetching_tool(job, ["iteration", "I1", "Q1", "I2", "Q2"], mode="live")
        # Live plotting

        x = np.arange(0, max_circuit_depth + 0.1, delta_clifford)
        x[0] = 1  # to set the first value of 'x' to be depth = 1 as in the experiment

        while results.is_processing():
            # Fetch results
            res = results.fetch_all()

            # Progress bar
            progress_counter(res[0], num_of_sequences, start_time=results.start_time)

            plt.suptitle("RB")

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
                save_data_dict[f"I{ind + 1}"] = res[2 * ind + 1]
                save_data_dict[f"Q{ind + 1}"] = res[2 * ind + 2]

        # Save results
        script_name = Path(__file__).name
        data_handler = DataHandler(root_data_folder=save_dir)
        save_data_dict.update({"fig_live": fig})
        data_handler.additional_files = {script_name: script_name, **default_additional_files}
        data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])

    except Exception as e:
        print(f"An exception occurred: {e}")

    finally:
        qm.close()
        print("Experiment QM is now closed")
        plt.show(block=True)
